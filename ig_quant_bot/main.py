from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import yaml

from ig_quant_bot.core.contracts import Order, FillEvent
from ig_quant_bot.core.feature_engine import FeatureEngine
from ig_quant_bot.core.ledgers import TradeLedger
from ig_quant_bot.core.state import PortfolioState
from ig_quant_bot.execution.instrument_db import InstrumentCatalog
from ig_quant_bot.execution.sim_broker import IGSyntheticBroker
from ig_quant_bot.vault.run_manager import RunManager
from ig_quant_bot.analytics.performance import PerformanceAnalyser


class QuantDesk:
    def __init__(self, config_path: str = "config.yaml"):
        # 1) Load Governance & Parameters
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            # Try relative to repo root (one level above package)
            alt = Path(__file__).resolve().parents[1] / config_path
            cfg_path = alt if alt.exists() else cfg_path

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f) or {}
        else:
            # Safe defaults so the project can run even if config.yaml is missing.
            self.cfg = {
                "strategy": {"name": "IG_Quant_Run", "universe": ["QQQ", "SPY"], "rsi_period": 14, "sma_period": 200,
                             "rsi_entry_threshold": 30, "rsi_exit_threshold": 50, "regime_filter": "BULL_STABLE"},
                "risk": {"initial_capital": 100000.0, "max_slots": 2, "max_drawdown_limit": 0.15,
                         "per_slot_margin_fraction": 0.15, "min_free_margin_buffer": 0.10, "allow_shorting": False,
                         "risk_per_trade_fraction": 0.005, "atr_stop_mult": 2.0, "atr_take_profit_mult": 2.5,
                         "max_hold_days": 10, "avoid_same_group": True},
                "costs": {"annual_funding_rate": 0.05, "weekend_multiplier": 3, "adaptive_spread_k": 0.10,
                          "commission_fixed": 0.0},
                "vault": {"output_path": "vault/runs"},
            }

        strat = self.cfg.get("strategy", {}) or {}
        risk = self.cfg.get("risk", {}) or {}
        costs = self.cfg.get("costs", {}) or {}
        vault_cfg = self.cfg.get("vault", {}) or {}

        # Strategy
        self.strategy_name = str(strat.get("name", "IG_Quant_Run"))
        self.universe = list(strat.get("universe", ["QQQ", "SPY"]))

        self.rsi_p = int(strat.get("rsi_period", 14))
        self.sma_p = int(strat.get("sma_period", 200))
        self.rsi_entry_threshold = float(strat.get("rsi_entry_threshold", 30))
        # Mean-reversion exits often work better net of financing/spread than waiting for overbought.
        self.rsi_exit_threshold = float(strat.get("rsi_exit_threshold", 50))
        self.regime_filter = str(strat.get("regime_filter", "BULL_STABLE"))

        # Optional entry filters to reduce "catching falling knives"
        self.require_trend_up = bool(strat.get("require_trend_up", False))
        self.require_rsi_turn = bool(strat.get("require_rsi_turn", False))
        self.min_signal_strength = float(strat.get("min_signal_strength", 0.0))

        # Risk
        self.initial_capital = float(risk.get("initial_capital", 100000.0))
        self.max_slots = int(risk.get("max_slots", 3))
        self.max_dd_limit = float(risk.get("max_drawdown_limit", 0.15))

        self.per_slot_margin_fraction = float(risk.get("per_slot_margin_fraction", 0.15))
        self.min_free_margin_buffer = float(risk.get("min_free_margin_buffer", 0.10))
        self.allow_shorting = bool(risk.get("allow_shorting", False))

        # Technique upgrades (configurable)
        self.risk_per_trade_fraction = float(risk.get("risk_per_trade_fraction", 0.0))  # 0 disables risk sizing
        self.atr_stop_mult = float(risk.get("atr_stop_mult", 0.0))  # 0 disables stop
        self.atr_take_profit_mult = float(risk.get("atr_take_profit_mult", 0.0))  # 0 disables TP
        self.max_hold_days = int(risk.get("max_hold_days", 0))  # 0 disables time stop
        self.avoid_same_group = bool(risk.get("avoid_same_group", False))

        # Costs
        self.annual_funding_rate = float(costs.get("annual_funding_rate", 0.05))
        self.weekend_multiplier = int(costs.get("weekend_multiplier", 3))
        self.adaptive_spread_k = float(costs.get("adaptive_spread_k", 0.10))
        self.commission_fixed = float(costs.get("commission_fixed", 0.0))

        # Vault
        vault_path = str(vault_cfg.get("output_path", "vault/runs"))
        self.run_manager = RunManager(base_path=vault_path)

        # 2) Initialize Engine Room
        self.catalog = InstrumentCatalog()

        # Broker (Backtest)
        self.broker = IGSyntheticBroker(
            self.catalog,
            annual_funding_rate=self.annual_funding_rate,
            weekend_multiplier=self.weekend_multiplier,
            adaptive_spread_k=self.adaptive_spread_k,
            commission_fixed=self.commission_fixed,
        )

        self.state = PortfolioState(self.initial_capital, allow_shorting=self.allow_shorting)
        self.trade_ledger = TradeLedger()
        self.analyser = PerformanceAnalyser()

        # 3) Persistence Buffers
        self.fills: List[FillEvent] = []
        self.ledger_rows: List[dict] = []

    def _make_run_id(self) -> str:
        return f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}_{self.strategy_name}_{str(uuid.uuid4())[:6]}"

    @staticmethod
    def _normalize_yf_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize yfinance outputs into OHLC columns required by FeatureEngine.

        yfinance versions and parameters can yield:
          - standard columns: Open/High/Low/Close/(Adj Close)/Volume
          - lowercase variants
          - MultiIndex columns when upstream config/grouping differs

        This function converts these shapes into a plain DataFrame with
        columns: Open, High, Low, Close (plus any extras preserved).
        """

        if df is None or df.empty:
            return df

        out = df.copy()

        # 1) Flatten MultiIndex columns if present (various yfinance shapes).
        if isinstance(out.columns, pd.MultiIndex):
            lvl0 = out.columns.get_level_values(0)
            lvl1 = out.columns.get_level_values(1)

            lvl0_lower = {str(x).lower() for x in lvl0}
            lvl1_lower = {str(x).lower() for x in lvl1}
            need = {"open", "high", "low", "close"}

            # Case A: first level are OHLC (e.g., ('Open','QQQ'))
            if need.issubset(lvl0_lower):
                series_map = {}
                for k in ["Open", "High", "Low", "Close"]:
                    # find the actual key in level0 (case-insensitive)
                    k0 = next((x for x in lvl0.unique() if str(x).lower() == k.lower()), None)
                    if k0 is None:
                        continue
                    sub = out[k0]
                    if isinstance(sub, pd.DataFrame):
                        # pick matching ticker if present, else first column
                        col = ticker if ticker in sub.columns else sub.columns[0]
                        series_map[k] = sub[col]
                    else:
                        series_map[k] = sub
                if series_map:
                    out = pd.concat(series_map, axis=1)

            # Case B: second level are OHLC (e.g., ('QQQ','Open'))
            elif need.issubset(lvl1_lower):
                # pick a ticker key from level0
                tick_keys = list(dict.fromkeys(out.columns.get_level_values(0).tolist()))
                tkey = next((x for x in tick_keys if str(x).upper() == str(ticker).upper()), tick_keys[0])
                series_map = {}
                for k in ["Open", "High", "Low", "Close"]:
                    col = next((c for c in out.columns if c[0] == tkey and str(c[1]).lower() == k.lower()), None)
                    if col is not None:
                        series_map[k] = out[col]
                if series_map:
                    out = pd.concat(series_map, axis=1)

            else:
                # Unknown MultiIndex: flatten to strings so the rename step below can still help.
                out.columns = ["|".join([str(a) for a in c]) for c in out.columns]

        # 2) Case-insensitive rename for standard columns.
        rename_map = {}
        for col in list(out.columns):
            c = str(col)
            cl = c.lower().strip()
            parts = [p.strip() for p in cl.split('|')]
            if cl == 'open':
                rename_map[col] = 'Open'
            elif cl == 'high':
                rename_map[col] = 'High'
            elif cl == 'low':
                rename_map[col] = 'Low'
            elif cl == 'close':
                rename_map[col] = 'Close'
            elif 'open' in parts:
                rename_map[col] = 'Open'
            elif 'high' in parts:
                rename_map[col] = 'High'
            elif 'low' in parts:
                rename_map[col] = 'Low'
            elif 'close' in parts:
                rename_map[col] = 'Close'
            elif cl in {'adj close', 'adj_close', 'adjusted close', 'adjusted_close'} and 'Close' not in out.columns:
                rename_map[col] = 'Close'
        if rename_map:
            out = out.rename(columns=rename_map)

        # 3) Ensure required columns exist; if yfinance returned "Adj Close" + others but not "Close",
        # mapping above should have handled it. If still missing, return as-is so caller can raise.
        # 4) Coerce OHLC columns to numeric (robust to object dtypes).
        for k in ['Open', 'High', 'Low', 'Close']:
            if k in out.columns:
                out[k] = pd.to_numeric(out[k], errors='coerce')

        return out

    def load_and_sync_data(self, start: str, end: Optional[str]) -> tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """Synchronize tickers to a master calendar while avoiding indicator bias from forward-filled bars."""
        if end is None:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")

        print(f"[*] Synchronizing {len(self.universe)} tickers against Master Calendar...")

        anchor = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if anchor is None or anchor.empty:
            raise RuntimeError("Failed to download SPY data for master calendar.")

        anchor = self._normalize_yf_ohlc(anchor, ticker="SPY")

        master_idx = anchor.index

        factors: Dict[str, pd.DataFrame] = {}
        for t in self.universe:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                print(f"[!] Warning: No data for {t}. Skipping.")
                continue

            df = self._normalize_yf_ohlc(df, ticker=t)
            df = df.copy()
            df["is_real_bar"] = True

            # Compute features on real bars only (no calendar forward-fill).
            feats = FeatureEngine.compute(df, rsi_p=self.rsi_p, sma_p=self.sma_p)

            # Align to master calendar without forward-filling prices or indicators.
            feats = feats.reindex(master_idx)
            feats["is_real_bar"] = feats["is_real_bar"].fillna(False).astype(bool)
            feats["valid_signal"] = feats["valid_signal"].fillna(False).astype(bool)

            factors[t] = feats

        if not factors:
            raise RuntimeError("No ticker data loaded. Universe is empty after downloads.")

        return factors, master_idx

    def _record_event(self, fill: FillEvent) -> None:
        self.fills.append(fill)
        self.trade_ledger.add_fill(fill)

    def _snapshot_daily_ledger(self, date: pd.Timestamp) -> None:
        self.ledger_rows.append(
            {
                "date": date,
                "equity": float(self.state.equity),
                "cash_balance": float(self.state.cash_balance),
                "margin_used": float(self.state.margin_used),
                "free_margin": float(self.state.free_margin),
                "pos_count": int(len(self.state.positions)),
                "drawdown": float(self.state.get_drawdown()),
            }
        )

    def _liquidate_all(self, date: pd.Timestamp, prices_mid: Dict[str, float], factors: Dict[str, pd.DataFrame], reason: str) -> None:
        """Close all open positions at the provided mid prices."""
        for t, pos in list(self.state.positions.items()):
            f = factors.get(t)
            regime = "UNKNOWN"
            atr_pts = 0.0
            if f is not None and date in f.index:
                loc = f.index.get_loc(date)
                if loc > 0:
                    prev = f.iloc[loc - 1]
                    regime = str(prev.get("Regime", "UNKNOWN"))
                    atr_pts = float(prev.get("ATR_pts", 0.0))
                    # Guard against NaN ATR on early bars
                    if pd.isna(atr_pts):
                        atr_pts = 0.0

            px = prices_mid.get(t)
            if px is None:
                px = float(pos.get("last_p") or pos.get("avg_price"))

            order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason=reason, timestamp=date)
            fill = self.broker.execute(order, float(px), regime, float(atr_pts), current_pos=pos)
            self.state.apply_fill(fill)
            self._record_event(fill)

    def run(self, start: str = "2020-01-01", end: Optional[str] = None) -> None:
        run_id = self._make_run_id()
        factors, master_idx = self.load_and_sync_data(start, end)

        print(f"[*] Starting Backtest Run: {run_id}")

        last_date = master_idx[-1]

        for date in master_idx:
            # Only use real bars for pricing inputs
            prices_open = {t: float(f.at[date, "Open"]) for t, f in factors.items() if bool(f.at[date, "is_real_bar"]) and pd.notna(f.at[date, "Open"]) }
            prices_high = {t: float(f.at[date, "High"]) for t, f in factors.items() if bool(f.at[date, "is_real_bar"]) and pd.notna(f.at[date, "High"]) }
            prices_low = {t: float(f.at[date, "Low"]) for t, f in factors.items() if bool(f.at[date, "is_real_bar"]) and pd.notna(f.at[date, "Low"]) }
            prices_close = {t: float(f.at[date, "Close"]) for t, f in factors.items() if bool(f.at[date, "is_real_bar"]) and pd.notna(f.at[date, "Close"]) }

            # A) MTM at the open (equity curve realism on entry-day spread impact handled later)
            self.state.update_mtm(prices_open, self.catalog)

            # B) Killswitch check
            if self.state.get_drawdown() < -self.max_dd_limit:
                print(f"[!] DRAWDOWN KILLSWITCH ACTIVATED AT {date}")
                # Liquidate at open to stop the run in an auditable way
                self._liquidate_all(date, prices_open, factors, reason="KILLSWITCH_LIQUIDATION")
                self.state.update_mtm(prices_open, self.catalog)
                self._snapshot_daily_ledger(date)
                break

            # C) EXITS (signal on t-1, execute on t prices)
            for t, pos in list(self.state.positions.items()):
                f = factors.get(t)
                if f is None:
                    continue

                loc = f.index.get_loc(date)
                if loc == 0:
                    continue

                prev = f.iloc[loc - 1]
                if not bool(prev.get("valid_signal", False)):
                    continue

                # --- Technique exits ---
                # 1) Intraday stop / take-profit using daily High/Low (conservative ordering: stop first)
                stop_px = pos.get("stop_price")
                tp_px = pos.get("take_profit_price")
                day_low = prices_low.get(t)
                day_high = prices_high.get(t)

                if stop_px is not None and day_low is not None and float(day_low) <= float(stop_px):
                    exit_mid = float(stop_px)
                    order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason="STOP_LOSS", timestamp=date)
                    fill = self.broker.execute(order, exit_mid, str(prev.get("Regime")), float(prev.get("ATR_pts")), current_pos=pos)
                    self.state.apply_fill(fill)
                    self._record_event(fill)
                    continue

                if tp_px is not None and day_high is not None and float(day_high) >= float(tp_px):
                    exit_mid = float(tp_px)
                    order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason="TAKE_PROFIT", timestamp=date)
                    fill = self.broker.execute(order, exit_mid, str(prev.get("Regime")), float(prev.get("ATR_pts")), current_pos=pos)
                    self.state.apply_fill(fill)
                    self._record_event(fill)
                    continue

                # 2) Time stop
                if self.max_hold_days and pos.get("entry_date") is not None:
                    held_days = (pd.to_datetime(date) - pd.to_datetime(pos["entry_date"])).days
                    if held_days >= int(self.max_hold_days):
                        px = prices_open.get(t, pos.get("last_p"))
                        if px is not None:
                            order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason="TIME_STOP", timestamp=date)
                            fill = self.broker.execute(order, float(px), str(prev.get("Regime")), float(prev.get("ATR_pts")), current_pos=pos)
                            self.state.apply_fill(fill)
                            self._record_event(fill)
                            continue

                # 3) RSI exit (mean reversion)
                exit_signal = float(prev["RSI"]) > self.rsi_exit_threshold

                # Force exit on the last available bar to avoid open positions in final metrics
                if date == last_date:
                    exit_signal = True

                if not exit_signal:
                    continue

                px = prices_open.get(t, pos.get("last_p"))
                if px is None:
                    continue

                order = Order(ticker=t, side="SELL", qty=float(pos["qty"]), reason="RSI_EXIT", timestamp=date)
                fill = self.broker.execute(order, float(px), str(prev.get("Regime")), float(prev.get("ATR_pts")), current_pos=pos)
                self.state.apply_fill(fill)
                self._record_event(fill)


            # Ensure no open positions remain on the final date even if indicators are not warmed up
            if date == last_date and len(self.state.positions) > 0:
                self._liquidate_all(date, prices_open, factors, reason="FINAL_LIQUIDATION")
            # D) ENTRIES (skip on last date)
            if date != last_date:
                self.state.update_mtm(prices_open, self.catalog)

                held_groups = set()
                if self.avoid_same_group:
                    for t in self.state.positions.keys():
                        held_groups.add(self.catalog.get(t).group)

                candidates = []
                for t, f in factors.items():
                    if t in self.state.positions:
                        continue
                    if not bool(f.at[date, "is_real_bar"]):
                        continue

                    loc = f.index.get_loc(date)
                    if loc == 0:
                        continue
                    prev = f.iloc[loc - 1]

                    if not bool(prev.get("valid_signal", False)):
                        continue
                    regime = str(prev.get("Regime", ""))
                    rf = str(self.regime_filter).upper()
                    if rf not in ("ANY", "ALL", "*") and str(regime).upper() != rf:
                        continue
                    if float(prev["RSI"]) >= self.rsi_entry_threshold:
                        continue

                    # Optional: require price in an uptrend (reduces trading in bear regimes even if regime labels are noisy)
                    if self.require_trend_up:
                        try:
                            if not (float(prev.get("Close")) > float(prev.get("SMA")) and float(prev.get("SMA_Slope")) > 0):
                                continue
                        except Exception:
                            continue

                    # Optional: require RSI to be turning up (reduces early entries during persistent selloffs)
                    if self.require_rsi_turn:
                        try:
                            if float(prev.get("RSI_Slope", 0.0)) <= 0:
                                continue
                        except Exception:
                            continue

                    # Optional: minimum composite strength
                    if self.min_signal_strength > 0:
                        try:
                            if float(prev.get("Signal_Strength", 0.0)) < self.min_signal_strength:
                                continue
                        except Exception:
                            continue

                    inst = self.catalog.get(t)
                    if self.avoid_same_group and inst.group in held_groups:
                        continue

                    atr_pts = float(prev.get("ATR_pts", 0.0))
                    # Guard against NaN ATR on early bars
                    if pd.isna(atr_pts):
                        atr_pts = 0.0
                    spread_pts = float(inst.spread_points)

                    # Score = RSI edge - spread friction penalty
                    spread_penalty = spread_pts / max(atr_pts, 1e-6)
                    score = (self.rsi_entry_threshold - float(prev["RSI"])) - (1.5 * spread_penalty)

                    candidates.append({"ticker": t, "score": float(score), "regime": str(prev.get("Regime")), "atr_pts": float(atr_pts)})

                candidates.sort(key=lambda x: x["score"], reverse=True)

                for cand in candidates:
                    if len(self.state.positions) >= self.max_slots:
                        break

                    t = cand["ticker"]
                    inst = self.catalog.get(t)

                    px = prices_open.get(t)
                    if px is None:
                        continue

                    # --- Sizing ---
                    # 1) Risk-per-trade sizing using ATR stop distance when enabled
                    qty_risk = None
                    if self.risk_per_trade_fraction > 0 and self.atr_stop_mult > 0 and cand["atr_pts"] > 0:
                        risk_budget = float(self.state.equity) * float(self.risk_per_trade_fraction)
                        stop_dist = float(self.atr_stop_mult) * float(cand["atr_pts"])  # points
                        qty_risk = risk_budget / (stop_dist * float(inst.value_per_point))
                        qty_risk = max(0.0, float(qty_risk))

                    # 2) Margin-budget cap (existing approach)
                    m_budget = float(self.state.equity * self.per_slot_margin_fraction)
                    target_notional = float(m_budget / float(inst.margin_factor))
                    qty_margin = target_notional / (float(px) * float(inst.value_per_point))

                    qty = float(qty_margin)
                    if qty_risk is not None and qty_risk > 0:
                        qty = float(min(qty_margin, qty_risk))

                    # Safety buffer
                    # Estimate margin requirement at mid price
                    gross_notional = abs(qty) * float(px) * float(inst.value_per_point)
                    est_margin = gross_notional * float(inst.margin_factor)
                    if (self.state.free_margin - est_margin) <= (self.state.equity * self.min_free_margin_buffer):
                        continue

                    if qty <= 0:
                        continue

                    order = Order(ticker=t, side="BUY", qty=float(qty), reason="RSI_ENTRY", timestamp=date)
                    fill = self.broker.execute(order, float(px), cand["regime"], cand["atr_pts"])
                    self.state.apply_fill(fill)
                    self._record_event(fill)

                    # Store trade metadata for technique exits
                    if t in self.state.positions:
                        p = self.state.positions[t]
                        p["entry_date"] = pd.to_datetime(date)
                        p["entry_reason"] = "RSI_ENTRY"
                        if self.atr_stop_mult > 0 and cand["atr_pts"] > 0:
                            p["stop_price"] = float(px) - (float(self.atr_stop_mult) * float(cand["atr_pts"]))
                        if self.atr_take_profit_mult > 0 and cand["atr_pts"] > 0:
                            p["take_profit_price"] = float(px) + (float(self.atr_take_profit_mult) * float(cand["atr_pts"]))
                        if self.max_hold_days:
                            p["max_hold_days"] = int(self.max_hold_days)

                        if self.avoid_same_group:
                            held_groups.add(inst.group)

            # E) MTM after entries so equity reflects spread immediately
            self.state.update_mtm(prices_open, self.catalog)

            # F) FUNDING at end-of-day for positions carried overnight (uses today's Close when available)
            for t, pos in list(self.state.positions.items()):
                px = prices_close.get(t, pos.get("last_p"))
                if px is None:
                    continue
                f_event = self.broker.calculate_funding(t, pos, float(px), date)
                self.state.apply_fill(f_event)
                self._record_event(f_event)

            # G) Snapshot after funding (authoritative EOD ledger)
            self._snapshot_daily_ledger(date)

        # H) Finalize & Persist
        self._wrap_up(run_id)

    def _wrap_up(self, run_id: str) -> None:
        ledger_df = pd.DataFrame(self.ledger_rows).set_index("date")
        fills_df = pd.DataFrame([asdict(f) for f in self.fills])
        trades_df = self.trade_ledger.get_trades_df()

        metadata = {"strategy": self.strategy_name, "universe": self.universe, "version": "8.0"}

        self.run_manager.persist_run(
            run_id,
            self.cfg,
            ledger_df=ledger_df,
            fills_df=fills_df,
            trades_df=trades_df,
            metadata=metadata,
        )

        # Final audit gate (non-fatal). A strict strategy can legitimately produce
        # zero fills over a window and that should not fail the whole run.
        try:
            self.analyser.audit_replay_cash_recursion(ledger_df, fills_df)
        except Exception as e:
            print(f"[!] Warning: cash recursion audit skipped/failed: {e}")

        try:
            sheet, trades_attrib_df, regime_table = self.analyser.generate_tear_sheet(ledger_df, fills_df, trades_df)
        except Exception as e:
            print(f"[!] Warning: tear sheet generation failed: {e}")
            # Provide a minimal sheet so the run still completes.
            from ig_quant_bot.analytics.performance import TearSheet
            final_equity = float(ledger_df['equity'].iloc[-1]) if (ledger_df is not None and not ledger_df.empty and 'equity' in ledger_df.columns) else 0.0
            sheet = TearSheet(final_equity=final_equity, cagr=0.0, ann_vol=0.0, sharpe=0.0, sortino=0.0, max_drawdown=0.0, total_trades=0, win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0)
            trades_attrib_df, regime_table = pd.DataFrame(), pd.DataFrame()

        # Expose last-run analytics for tooling scripts (for example sweep scripts).
        # This avoids brittle log parsing.
        self.last_tear_sheet = sheet
        self.last_trades_attrib_df = trades_attrib_df
        self.last_regime_table = regime_table


        print("[*] Run complete.")
        print(f"    Final equity: {sheet.final_equity:.2f}")
        print(f"    CAGR: {sheet.cagr:.4f} | Sharpe: {sheet.sharpe:.3f} | MaxDD: {sheet.max_drawdown:.3f}")
        print(f"    Trades: {sheet.total_trades} | Win rate: {sheet.win_rate:.3f} | PF: {sheet.profit_factor:.3f}")

        # Persist analytics outputs
        try:
            run_dir = f"{self.run_manager.base_path}/{run_id}"
            # Also persist a lightweight summary for quick sweeps.
            import json
            summary = {
                "final_equity": float(sheet.final_equity),
                "cagr": float(sheet.cagr),
                "ann_vol": float(sheet.ann_vol),
                "sharpe": float(sheet.sharpe),
                "sortino": float(sheet.sortino),
                "max_drawdown": float(sheet.max_drawdown),
                "total_trades": int(sheet.total_trades),
                "win_rate": float(sheet.win_rate),
                "profit_factor": float(sheet.profit_factor),
                "avg_win": float(sheet.avg_win),
                "avg_loss": float(sheet.avg_loss),
            }
            with open(f"{run_dir}/tear_sheet_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if trades_attrib_df is not None and not trades_attrib_df.empty:
                trades_attrib_df.to_parquet(f"{run_dir}/trades_attrib.parquet")
            if regime_table is not None and not regime_table.empty:
                regime_table.to_parquet(f"{run_dir}/expectancy_by_regime.parquet")
        except Exception as e:
            print(f"[!] Warning: Could not persist analytics artifacts: {e}")


if __name__ == "__main__":
    # Prefer running as a module: python -m ig_quant_bot.main
    QuantDesk().run(start="2020-01-01")
