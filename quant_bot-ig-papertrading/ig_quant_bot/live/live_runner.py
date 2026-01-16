from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ig_quant_bot.analytics.performance import PerformanceAnalyser
from ig_quant_bot.core.contracts import Order
from ig_quant_bot.core.feature_engine import FeatureEngine
from ig_quant_bot.core.ledgers import TradeLedger
from ig_quant_bot.core.state import PortfolioState
from ig_quant_bot.execution.instrument_db import InstrumentCatalog
from ig_quant_bot.execution.ig_live_adapter import IGLiveAdapter
from ig_quant_bot.observability.decision_logger import DecisionLogger
from ig_quant_bot.observability.logging_setup import setup_logging
from ig_quant_bot.vault.run_manager import RunManager
from ig_quant_bot.utils.config_loader import load_config
from ig_quant_bot.live.ig_data import prices_to_ohlc_df, extract_quote_mid
from ig_quant_bot.live.state_store import load_state, save_state


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


@dataclass
class LiveConfig:
    universe: List[str]
    resolution: str
    lookback_points: int
    state_path: str
    dry_run: bool


class LiveQuantDesk:
    """Daily live paper-trading runner for IG (DEMO recommended).

    Design goals:
      - No lookahead: signals are computed on the last *completed* bar, execution is "now".
      - Safety-first: idempotency keys, drawdown kill-switch, dry-run mode.
      - Audit-first: decision_log.csv and run.log are produced every run.
    """

    def __init__(self, config_path: str = "config.yaml", *, connect: bool = True):
        self.cfg = load_config(config_path)
        self._connect_on_init = bool(connect)

        self.catalog = InstrumentCatalog()

        strat = self.cfg.get("strategy", {}) or {}
        risk = self.cfg.get("risk", {}) or {}
        live = self.cfg.get("live", {}) or {}

        self.universe = list(strat.get("universe", []) or [])
        self.rsi_period = int(strat.get("rsi_period", 14))
        self.sma_period = int(strat.get("sma_period", 200))
        self.rsi_entry = float(strat.get("rsi_entry_threshold", 30))
        self.rsi_exit = float(strat.get("rsi_exit_threshold", 50))
        self.regime_filter = str(strat.get("regime_filter", "BULL_STABLE"))

        self.initial_capital = float(risk.get("initial_capital", 100000.0))
        self.max_slots = int(risk.get("max_slots", 2))
        self.max_dd_limit = float(risk.get("max_drawdown_limit", 0.15))
        self.per_slot_margin_fraction = float(risk.get("per_slot_margin_fraction", 0.15))
        self.min_free_margin_buffer = float(risk.get("min_free_margin_buffer", 0.10))
        self.allow_shorting = bool(risk.get("allow_shorting", False))

        self.risk_per_trade_fraction = float(risk.get("risk_per_trade_fraction", 0.0))
        self.atr_stop_mult = float(risk.get("atr_stop_mult", 0.0))
        self.atr_tp_mult = float(risk.get("atr_take_profit_mult", 0.0))
        self.max_hold_days = int(risk.get("max_hold_days", 0))
        self.avoid_same_group = bool(risk.get("avoid_same_group", True))

        self.live_resolution = str(live.get("resolution", "DAY")).upper()
        self.lookback_points = int(live.get("lookback_points", 450))
        self.state_path = str(live.get("state_path", "vault/live_state/state.json"))
        self.dry_run = bool(live.get("dry_run", True))

        self.output_path = str((self.cfg.get("vault", {}) or {}).get("output_path", "vault/runs"))
        self.run_manager = RunManager(base_path=self.output_path)
        self.live_cfg = live

        # Adapter initialises an IG session unless connect is disabled (used by smoke tests).
        self.adapter: Optional[IGLiveAdapter] = None
        if self._connect_on_init:
            self.adapter = IGLiveAdapter(live, self.catalog)
    def run_smoke(self) -> str:
        """Offline smoke test: validates config loading and logging without connecting to IG."""
        run_id = f"smoke_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = Path(self.output_path) / run_id
        logger = setup_logging(run_dir=run_dir, logger_name='ig_quant_bot.live')
        decision_logger = DecisionLogger(run_dir / 'decision_log.csv', run_id=run_id)
        decision_logger.log(
            date=pd.Timestamp.utcnow(),
            ticker='SYSTEM',
            phase='SYSTEM',
            action='INFO',
            reason_code='SMOKE_OK',
            details={
                'dry_run': bool(self.dry_run),
                'universe': list(self.universe),
                'live_resolution': str(self.live_resolution),
                'lookback_points': int(self.lookback_points),
            },
        )
        decision_logger.close()
        logger.info(f'Smoke OK: {run_id}')
        return run_id

    def run_once(self) -> str:

        run_id = f"live_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = Path(self.output_path) / run_id
        logger = setup_logging(run_dir=run_dir, logger_name="ig_quant_bot.live")
        decision_logger = DecisionLogger(run_dir / "decision_log.csv", run_id=run_id)

        state_file = Path(self.state_path)
        live_state = load_state(state_file)
        order_keys_seen = set(live_state.get('order_keys', []) or [])

        if self.adapter is None:
            self.adapter = IGLiveAdapter(self.live_cfg, self.catalog)

        try:
            # --- Guardrails ---
            acc_type = str((self.cfg.get("live", {}) or {}).get("acc_type", "DEMO")).upper()
            if acc_type == "LIVE" and not bool((self.cfg.get("live", {}) or {}).get("allow_live", False)):
                raise RuntimeError(
                    "LIVE account blocked. Set live.allow_live: true in config.yaml only when you are ready." 
                )

            # --- Reconcile open positions and account snapshot ---
            ps = PortfolioState(
                initial_cash=self.initial_capital,
                allow_shorting=self.allow_shorting,
                strict_reconciliation=False,
            )

            account_snapshot = self._fetch_account_snapshot(logger)
            if account_snapshot.get("equity") is not None:
                ps.cash_balance = float(account_snapshot.get("cash", ps.cash_balance))
                ps.equity = float(account_snapshot.get("equity", ps.equity))
                ps.peak_equity = max(float(live_state.get("peak_equity", 0.0)), float(ps.equity))

            open_pos = self._fetch_open_positions(logger)
            self._apply_reconciled_positions(ps, open_pos, live_state, logger)

            # Update MTM using current quotes
            prices_now = {}
            for t in list(ps.positions.keys()):
                epic = self.catalog.get(t).epic
                q = self.adapter.ig.fetch_market_by_epic(epic)
                mid = extract_quote_mid(q)
                if mid is not None:
                    prices_now[t] = float(mid)
            ps.update_mtm(prices_now, self.catalog)

            # Drawdown killswitch
            live_state["peak_equity"] = float(max(float(live_state.get("peak_equity", 0.0)), float(ps.equity)))
            dd = (float(ps.equity) / float(live_state["peak_equity"]) - 1.0) if float(live_state["peak_equity"]) > 0 else 0.0
            if dd <= -abs(float(self.max_dd_limit)):
                logger.warning(f"Killswitch triggered: drawdown={dd:.2%} <= -{self.max_dd_limit:.2%}. Closing all.")
                decision_logger.log(
                    date=pd.Timestamp.utcnow(),
                    ticker="ALL",
                    phase="RISK",
                    action="ORDER",
                    side="SELL",
                    reason_code="DRAWDOWN_KILLSWITCH",
                    equity=ps.equity,
                    free_margin=ps.free_margin,
                    margin_used=ps.margin_used,
                    details={"drawdown": dd, "limit": self.max_dd_limit},
                )
                if not self.dry_run:
                    self._close_all(ps, decision_logger, logger)
                if not self.dry_run:
                    save_state(state_file, live_state)
                decision_logger.close()
                return run_id

            # --- Compute signals from last completed bar for each instrument ---
            features_map: Dict[str, pd.DataFrame] = {}
            signal_row: Dict[str, pd.Series] = {}

            today_london = pd.Timestamp.now(tz="Europe/London").date()

            for t in self.universe:
                inst = self.catalog.get(t)
                try:
                    resp = self.adapter.ig.fetch_historical_prices_by_epic(
                        inst.epic, resolution=self.live_resolution, num_points=self.lookback_points
                    )
                    ohlc = prices_to_ohlc_df(resp, tz="UTC")
                    if ohlc.empty or len(ohlc) < max(self.sma_period, 60):
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="SYSTEM",
                            action="SKIP",
                            reason_code="INSUFFICIENT_HISTORY",
                        )
                        continue

                    f = FeatureEngine.compute(ohlc, rsi_p=self.rsi_period, sma_p=self.sma_period)

                    # Choose last completed daily bar (avoid using partial current day)
                    idx = f.index
                    last_dt = pd.to_datetime(idx[-1]).date()
                    use_i = -2 if last_dt == today_london and len(f) >= 2 else -1
                    row = f.iloc[use_i]

                    if not bool(row.get("valid_signal", False)):
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="SYSTEM",
                            action="SKIP",
                            reason_code="INVALID_SIGNAL_WARMUP",
                        )
                        continue

                    features_map[t] = f
                    signal_row[t] = row

                except Exception as e:
                    logger.exception(f"Signal build failed for {t}: {e}")
                    decision_logger.log(
                        date=pd.Timestamp.utcnow(),
                        ticker=t,
                        phase="SYSTEM",
                        action="SKIP",
                        reason_code="DATA_FETCH_OR_FEATURE_ERROR",
                        details={"error": str(e)},
                    )

            # --- Exit logic first ---
            ledger = TradeLedger()

            for t, pos in list(ps.positions.items()):
                row = signal_row.get(t)
                if row is None:
                    continue

                rsi = float(row.get("RSI", 0.0))
                regime = str(row.get("Regime", "UNKNOWN"))
                atr_pts = float(row.get("ATR_pts", 0.0)) if pd.notna(row.get("ATR_pts")) else 0.0

                # Time stop (max_hold_days)
                deal_id = str(pos.get("deal_id") or "")
                meta = (live_state.get("positions", {}) or {}).get(deal_id, {})
                entry_time = pd.to_datetime(meta.get("entry_time"), errors="coerce")

                time_stop = False
                if self.max_hold_days and pd.notna(entry_time):
                    held_days = int((pd.Timestamp.utcnow().normalize() - entry_time.normalize()).days)
                    if held_days >= int(self.max_hold_days):
                        time_stop = True

                exit_signal = (rsi >= float(self.rsi_exit)) or time_stop

                if not exit_signal:
                    decision_logger.log(
                        date=pd.Timestamp.utcnow(),
                        ticker=t,
                        phase="EXIT",
                        action="SKIP",
                        reason_code="NO_EXIT_SIGNAL",
                        rsi=rsi,
                        atr_pts=atr_pts,
                        regime=regime,
                        equity=ps.equity,
                        free_margin=ps.free_margin,
                        margin_used=ps.margin_used,
                    )
                    continue

                order_key = f"{pd.Timestamp.utcnow().date().isoformat()}|{t}|SELL|EXIT"
                if order_key in order_keys_seen:
                    decision_logger.log(
                        date=pd.Timestamp.utcnow(),
                        ticker=t,
                        phase="EXIT",
                        action="SKIP",
                        reason_code="IDEMPOTENT_ALREADY_DONE",
                        idempotency_key=order_key,
                    )
                    continue

                # Quote mid for sizing (closing uses current stake)
                epic = self.catalog.get(t).epic
                q = self.adapter.ig.fetch_market_by_epic(epic)
                mid = extract_quote_mid(q)

                decision_logger.log(
                    date=pd.Timestamp.utcnow(),
                    ticker=t,
                    phase="EXIT",
                    action="ORDER",
                    side="SELL",
                    qty=float(pos.get("qty", 0.0)),
                    price_mid=float(mid or 0.0),
                    regime=regime,
                    rsi=rsi,
                    atr_pts=atr_pts,
                    equity=ps.equity,
                    free_margin=ps.free_margin,
                    margin_used=ps.margin_used,
                    reason_code="EXIT_SIGNAL",
                    idempotency_key=order_key,
                    details={"time_stop": time_stop},
                )

                if not self.dry_run:
                    fill = self.adapter.execute(
                        Order(ticker=t, side="SELL", qty=float(pos.get("qty")), reason="EXIT"),
                        price=float(mid or pos.get("last_p") or pos.get("avg_price")),
                        regime=regime,
                        atr_pts=atr_pts,
                        current_pos=pos,
                    )
                    ps.apply_fill(fill)
                    ledger.add_fill(fill)

                order_keys_seen.add(order_key)
                if not self.dry_run:
                    live_state.setdefault("order_keys", []).append(order_key)

            # --- Entry logic ---
            open_count = int(len(ps.positions))
            slots_available = max(0, int(self.max_slots) - open_count)
            if slots_available <= 0:
                decision_logger.log(
                    date=pd.Timestamp.utcnow(),
                    ticker="ALL",
                    phase="ENTRY",
                    action="SKIP",
                    reason_code="NO_SLOTS_AVAILABLE",
                    equity=ps.equity,
                    free_margin=ps.free_margin,
                    margin_used=ps.margin_used,
                )
            else:
                candidates = []
                held_groups = {self.catalog.get(t).group for t in ps.positions.keys()}

                for t, row in signal_row.items():
                    if t in ps.positions:
                        continue

                    inst = self.catalog.get(t)
                    if self.avoid_same_group and inst.group in held_groups:
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="GROUP_CONFLICT",
                            details={"group": inst.group},
                        )
                        continue

                    rsi = float(row.get("RSI", 0.0))
                    regime = str(row.get("Regime", "UNKNOWN"))
                    if regime != self.regime_filter:
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="REGIME_MISMATCH",
                            regime=regime,
                        )
                        continue

                    if not (rsi < float(self.rsi_entry)):
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="RSI_NOT_LOW_ENOUGH",
                            rsi=rsi,
                            regime=regime,
                        )
                        continue

                    atr_pts = float(row.get("ATR_pts", 0.0)) if pd.notna(row.get("ATR_pts")) else 0.0
                    score = (float(self.rsi_entry) - rsi)
                    candidates.append((score, t, regime, rsi, atr_pts))

                candidates.sort(reverse=True)

                for _, t, regime, rsi, atr_pts in candidates[:slots_available]:
                    inst = self.catalog.get(t)

                    # Quote
                    q = self.adapter.ig.fetch_market_by_epic(inst.epic)
                    mid = extract_quote_mid(q)
                    if mid is None or mid <= 0:
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="NO_QUOTE",
                        )
                        continue

                    # Sizing (margin-based cap + optional risk-per-trade cap)
                    max_margin_for_slot = float(ps.equity) * float(self.per_slot_margin_fraction)
                    stake_by_margin = max_margin_for_slot / (float(mid) * float(inst.value_per_point) * float(inst.margin_factor))

                    # Optional risk-based sizing (stake = risk_cash / (stop_distance * value_per_point))
                    stake_by_risk = float("inf")
                    stop_level = None
                    limit_level = None

                    if self.risk_per_trade_fraction > 0 and self.atr_stop_mult > 0 and atr_pts > 0:
                        risk_cash = float(ps.equity) * float(self.risk_per_trade_fraction)
                        stop_dist = float(self.atr_stop_mult) * float(atr_pts)
                        stake_by_risk = risk_cash / (stop_dist * float(inst.value_per_point))

                    stake = float(min(stake_by_margin, stake_by_risk))
                    stake = max(0.5, round(stake, 2))  # IG min stake differs by market, keep a conservative floor

                    # Free margin buffer gate
                    projected_margin = stake * float(mid) * float(inst.value_per_point) * float(inst.margin_factor)
                    projected_free_margin = float(ps.free_margin) - projected_margin
                    if projected_free_margin < (float(ps.equity) * float(self.min_free_margin_buffer)):
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="FREE_MARGIN_BUFFER",
                            rsi=rsi,
                            atr_pts=atr_pts,
                            regime=regime,
                            equity=ps.equity,
                            free_margin=ps.free_margin,
                            margin_used=ps.margin_used,
                            details={"projected_free_margin": projected_free_margin, "stake": stake},
                        )
                        continue

                    # Attach stop/limit if enabled
                    if self.atr_stop_mult > 0 and atr_pts > 0:
                        stop_level = float(mid) - float(self.atr_stop_mult) * float(atr_pts)
                    if self.atr_tp_mult > 0 and atr_pts > 0:
                        limit_level = float(mid) + float(self.atr_tp_mult) * float(atr_pts)

                    order_key = f"{pd.Timestamp.utcnow().date().isoformat()}|{t}|BUY|ENTRY"
                    if order_key in order_keys_seen:
                        decision_logger.log(
                            date=pd.Timestamp.utcnow(),
                            ticker=t,
                            phase="ENTRY",
                            action="SKIP",
                            reason_code="IDEMPOTENT_ALREADY_DONE",
                            idempotency_key=order_key,
                        )
                        continue

                    decision_logger.log(
                        date=pd.Timestamp.utcnow(),
                        ticker=t,
                        phase="ENTRY",
                        action="ORDER",
                        side="BUY",
                        qty=stake,
                        price_mid=float(mid),
                        stop_level=stop_level,
                        limit_level=limit_level,
                        regime=regime,
                        rsi=rsi,
                        atr_pts=atr_pts,
                        equity=ps.equity,
                        free_margin=ps.free_margin,
                        margin_used=ps.margin_used,
                        reason_code="ENTRY_SIGNAL",
                        idempotency_key=order_key,
                    )

                    if not self.dry_run:
                        fill = self.adapter.execute(
                            Order(
                                ticker=t,
                                side="BUY",
                                qty=stake,
                                reason="ENTRY",
                                stop_level=stop_level,
                                limit_level=limit_level,
                                currency_code=str((self.cfg.get("live", {}) or {}).get("currency_code", "GBP")),
                                expiry=str((self.cfg.get("live", {}) or {}).get("expiry", "DFB")),
                            ),
                            price=float(mid),
                            regime=regime,
                            atr_pts=atr_pts,
                            current_pos=None,
                        )
                        ps.apply_fill(fill)
                        ledger.add_fill(fill)

                        # Persist metadata keyed by deal id
                        deal_id = str(fill.fill_id)
                        live_state.setdefault("positions", {})[deal_id] = {
                            "ticker": t,
                            "epic": inst.epic,
                            "direction": "BUY",
                            "size": float(stake),
                            "entry_time": pd.Timestamp.utcnow().isoformat(),
                            "entry_price": float(fill.price),
                            "stop_level": float(stop_level) if stop_level is not None else None,
                            "limit_level": float(limit_level) if limit_level is not None else None,
                            "max_hold_days": int(self.max_hold_days) if self.max_hold_days else None,
                        }

                    order_keys_seen.add(order_key)
                    if not self.dry_run:
                        live_state.setdefault("order_keys", []).append(order_key)

            # --- Persist run artifacts ---
            ledger_df = pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp.utcnow(),
                        "cash_balance": float(ps.cash_balance),
                        "margin_used": float(ps.margin_used),
                        "equity": float(ps.equity),
                        "free_margin": float(ps.free_margin),
                    }
                ]
            ).set_index("timestamp")

            fills_df = pd.DataFrame([f.__dict__ for f in ledger.all_fills]) if ledger.all_fills else pd.DataFrame()
            trades_df = ledger.get_trades_df()

            self.run_manager.persist_run(
                run_id,
                self.cfg,
                ledger_df=ledger_df,
                fills_df=fills_df,
                trades_df=trades_df,
                metadata={
                    "mode": "LIVE_PAPER" if str((self.cfg.get("live", {}) or {}).get("acc_type", "DEMO")).upper() == "DEMO" else "LIVE",
                    "dry_run": self.dry_run,
                    "universe": self.universe,
                    "positions_count": int(len(ps.positions)),
                },
            )

            if not self.dry_run:
                save_state(state_file, live_state)
            decision_logger.close()
            return run_id

        except Exception as e:
            logger.exception(f"Live run failed: {e}")
            decision_logger.log(
                date=pd.Timestamp.utcnow(),
                ticker="SYSTEM",
                phase="SYSTEM",
                action="SKIP",
                reason_code="RUN_FAILED",
                details={"error": str(e)},
            )
            if not self.dry_run:
                save_state(state_file, live_state)
            decision_logger.close()
            raise

    # ------------------ IG helpers ------------------

    def _fetch_account_snapshot(self, logger: logging.Logger) -> Dict[str, Any]:
        """Best-effort account snapshot.

        trading-ig shapes vary by version. We extract what we can and log the rest.
        """
        out = {"cash": None, "equity": None}
        try:
            resp = self.adapter.ig.fetch_accounts()
        except Exception as e:
            logger.warning(f"fetch_accounts failed: {e}")
            return out

        try:
            # If DataFrame, pick first row
            if hasattr(resp, "iloc") and hasattr(resp, "to_dict"):
                df = resp
                row = df.iloc[0].to_dict() if len(df) else {}
            elif isinstance(resp, dict) and "accounts" in resp and isinstance(resp["accounts"], list):
                row = resp["accounts"][0] if resp["accounts"] else {}
            elif isinstance(resp, list):
                row = resp[0] if resp else {}
            else:
                row = resp if isinstance(resp, dict) else {}

            # Heuristics for keys
            for k in ["available", "availableBalance", "availableCash", "balanceAvailable"]:
                if k in row:
                    out["cash"] = float(row.get(k))
                    break

            for k in ["balance", "accountBalance", "equity", "netEquity", "balanceNet"]:
                if k in row:
                    out["equity"] = float(row.get(k))
                    break

            # Nested balance objects are common
            bal = row.get("balance") if isinstance(row, dict) else None
            if isinstance(bal, dict):
                out["equity"] = float(out["equity"] or bal.get("balance") or bal.get("equity") or out["equity"] or 0.0)
                out["cash"] = float(out["cash"] or bal.get("available") or bal.get("availableBalance") or out["cash"] or 0.0)

        except Exception as e:
            logger.warning(f"Account snapshot parse failed: {e}")

        return out

    def _fetch_open_positions(self, logger: logging.Logger) -> Any:
        try:
            return self.adapter.ig.fetch_open_positions()
        except Exception as e:
            logger.warning(f"fetch_open_positions failed: {e}")
            return None

    def _apply_reconciled_positions(self, ps: PortfolioState, open_positions: Any, live_state: Dict[str, Any], logger: logging.Logger) -> None:
        """Rebuild local PortfolioState from broker open positions.

        This keeps the bot safe if it restarts mid-flight.
        """
        ps.positions = {}
        ps.margin_used = 0.0

        if open_positions is None:
            return

        # trading-ig typically returns a DataFrame; handle list/dict too.
        rows = []
        if hasattr(open_positions, "to_dict") and hasattr(open_positions, "iterrows"):
            for _, r in open_positions.iterrows():
                rows.append(r.to_dict())
        elif isinstance(open_positions, dict) and "positions" in open_positions and isinstance(open_positions["positions"], list):
            rows = list(open_positions["positions"])
        elif isinstance(open_positions, list):
            rows = open_positions

        by_epic = {}
        for r in rows:
            try:
                m = r.get("market") if isinstance(r.get("market"), dict) else {}
                p = r.get("position") if isinstance(r.get("position"), dict) else {}

                epic = (m.get("epic") or r.get("epic") or p.get("epic"))
                if not epic:
                    continue

                deal_id = p.get("dealId") or p.get("deal_id") or r.get("dealId") or r.get("deal_id")
                direction = p.get("direction") or r.get("direction")
                size = p.get("size") or r.get("size")
                level = p.get("level") or r.get("level") or p.get("openLevel")

                by_epic[str(epic)] = {
                    "deal_id": str(deal_id) if deal_id is not None else None,
                    "direction": str(direction or "BUY").upper(),
                    "size": float(size) if size is not None else 0.0,
                    "level": float(level) if level is not None else None,
                    "epic": str(epic),
                }
            except Exception:
                continue

        # Map epics to our tickers (only universe)
        ticker_by_epic = {self.catalog.get(t).epic: t for t in self.universe}

        live_positions = live_state.get("positions", {}) or {}

        for epic, info in by_epic.items():
            if epic not in ticker_by_epic:
                continue
            t = ticker_by_epic[epic]
            inst = self.catalog.get(t)

            qty = abs(float(info.get("size") or 0.0))
            if qty <= 0:
                continue
            avg = float(info.get("level") or 0.0) if info.get("level") is not None else 0.0
            if avg <= 0:
                # No level returned -> will be updated by MTM quote.
                avg = 1.0

            gross = qty * avg * float(inst.value_per_point)
            margin = gross * float(inst.margin_factor)

            deal_id = str(info.get("deal_id") or "")

            ps.positions[t] = {
                "qty": float(qty),
                "avg_price": float(avg),
                "margin": float(margin),
                "last_p": float(avg),
                "unrealized_pnl": 0.0,
                "deal_id": deal_id if deal_id else None,
                "entry_date": None,
                "stop_price": None,
                "take_profit_price": None,
                "max_hold_days": None,
                "entry_reason": None,
            }
            ps.margin_used += float(margin)

            # Keep metadata if present
            if deal_id and deal_id not in live_positions:
                live_positions[deal_id] = {
                    "ticker": t,
                    "epic": epic,
                    "direction": str(info.get("direction") or "BUY").upper(),
                    "size": float(qty),
                    "entry_time": pd.Timestamp.utcnow().isoformat(),
                    "entry_price": float(avg),
                }

        live_state["positions"] = live_positions
        ps.free_margin = float(ps.equity) - float(ps.margin_used)

    def _close_all(self, ps: PortfolioState, decision_logger: DecisionLogger, logger: logging.Logger) -> None:
        for t, pos in list(ps.positions.items()):
            epic = self.catalog.get(t).epic
            q = self.adapter.ig.fetch_market_by_epic(epic)
            mid = extract_quote_mid(q) or pos.get("last_p") or pos.get("avg_price")
            fill = self.adapter.execute(
                Order(ticker=t, side="SELL", qty=float(pos.get("qty")), reason="KILLSWITCH"),
                price=float(mid),
                regime="UNKNOWN",
                atr_pts=0.0,
                current_pos=pos,
            )
            ps.apply_fill(fill)
            logger.info(f"Closed {t} via killswitch")
