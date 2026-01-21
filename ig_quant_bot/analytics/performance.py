# V7.2: Funding attribution, expectancy by regime, tear sheet and audit replay
#
# CONSISTENCY GUARANTEES FOR V7.2:
# - ledger_df must have index "date" and column "equity"
# - ledger_df MUST also include "cash" OR "cash_balance" (we normalize to cash_balance)
# - fills_df must include: timestamp, ticker, event_type, net_cashflow (others optional)
# - trades_df can be provided from core/ledgers.py OR rebuilt from fills
#
# This module is intentionally strict: if the platform writes inconsistent columns,
# it fails loudly instead of producing a misleading tear sheet.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TearSheet:
    final_equity: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


class PerformanceAnalyser:
    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        self.trading_days = int(trading_days)
        rf = float(risk_free_rate)
        self.rf_daily = (1.0 + rf) ** (1.0 / self.trading_days) - 1.0

    def generate_tear_sheet(
        self,
        ledger_df: pd.DataFrame,
        fills_df: pd.DataFrame,
        trades_df: Optional[pd.DataFrame] = None,
        rebuild_trades_if_missing: bool = True,
    ) -> Tuple[TearSheet, pd.DataFrame, pd.DataFrame]:
        ledger_df = self._validate_and_normalize_ledger(ledger_df)
        fills_df = self._prep_fills(fills_df)

        if trades_df is None or trades_df.empty:
            if rebuild_trades_if_missing:
                trades_df = self.rebuild_trades_from_fills(fills_df)
            else:
                trades_df = pd.DataFrame()

        trades_attrib_df = (
            self.attribute_funding_to_trades(trades_df, fills_df) if not trades_df.empty else trades_df
        )

        ts = self._compute_timeseries_metrics(ledger_df)
        trade_metrics = self._compute_trade_metrics(trades_attrib_df)

        regime_table = (
            self.expectancy_by_regime(trades_attrib_df) if not trades_attrib_df.empty else pd.DataFrame()
        )

        sheet = TearSheet(
            final_equity=ts["final_equity"],
            cagr=ts["cagr"],
            ann_vol=ts["ann_vol"],
            sharpe=ts["sharpe"],
            sortino=ts["sortino"],
            max_drawdown=ts["max_drawdown"],
            total_trades=trade_metrics["total_trades"],
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
        )

        return sheet, trades_attrib_df, regime_table

    def attribute_funding_to_trades(
        self,
        trades_df: pd.DataFrame,
        fills_df: pd.DataFrame,
        *,
        overwrite_funding: bool = False,
        overwrite_total: bool = False,
    ) -> pd.DataFrame:
        """Attach funding PnL to each trade.

        Important consistency rule:
          - If trades_df already contains a "total_net_pnl" column we preserve it by default.
            This avoids silently overwriting totals that may already include fees and liquidation adjustments.
          - We only (re)compute total_net_pnl when it is missing or when overwrite_total=True.
        """
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()

        fills_df = self._prep_fills(fills_df)
        tdf = trades_df.copy()

        # If there are no fills, we cannot attribute funding. Default to 0 funding.
        if fills_df is None or fills_df.empty or "event_type" not in fills_df.columns:
            if overwrite_funding or ("funding_net_pnl" not in tdf.columns):
                tdf["funding_net_pnl"] = 0.0
            if overwrite_total or ("total_net_pnl" not in tdf.columns):
                base_col = "trade_net_pnl" if "trade_net_pnl" in tdf.columns else None
                if base_col is None:
                    tdf["total_net_pnl"] = 0.0
                else:
                    tdf["total_net_pnl"] = pd.to_numeric(tdf[base_col], errors="coerce").fillna(0.0) + pd.to_numeric(tdf.get("funding_net_pnl", 0.0), errors="coerce").fillna(0.0)
            return tdf

        if "ticker" not in tdf.columns:
            raise ValueError("trades_df must include 'ticker'")

        for c in ["entry_time", "exit_time"]:
            if c not in tdf.columns:
                raise ValueError(f"trades_df must include '{c}'")
            tdf[c] = pd.to_datetime(tdf[c])

        # --- Funding attribution ---
        need_funding = overwrite_funding or ("funding_net_pnl" not in tdf.columns)

        if need_funding:
            # If fills are missing (common when a strict strategy generates zero trades),
            # treat funding as zero to keep analytics robust.
            if fills_df is None or fills_df.empty or "event_type" not in fills_df.columns:
                tdf["funding_net_pnl"] = 0.0
            else:
                fnd = fills_df[fills_df["event_type"].astype(str).str.upper() == "FUNDING"].copy()
                if fnd.empty:
                    tdf["funding_net_pnl"] = 0.0
                else:
                    fnd["timestamp"] = pd.to_datetime(fnd["timestamp"])
                    fnd_by_ticker: Dict[str, pd.DataFrame] = {k: v for k, v in fnd.groupby("ticker", sort=False)}

                    funding_vals = []
                    for _, row in tdf.iterrows():
                        tick = row["ticker"]
                        entry = row["entry_time"]
                        exit_ = row["exit_time"]
                        if tick not in fnd_by_ticker:
                            funding_vals.append(0.0)
                            continue
                        ff = fnd_by_ticker[tick]
                        mask = (ff["timestamp"] >= entry) & (ff["timestamp"] <= exit_)
                        funding_vals.append(float(ff.loc[mask, "net_cashflow"].sum()))
                    tdf["funding_net_pnl"] = funding_vals
        else:
            tdf["funding_net_pnl"] = pd.to_numeric(tdf["funding_net_pnl"], errors="coerce").fillna(0.0)

        # --- Total PnL ---
        if ("total_net_pnl" not in tdf.columns) or overwrite_total:
            base_trade_pnl_col = None
            for c in ["trade_net_pnl", "net_pnl", "trade_pnl", "pnl"]:
                if c in tdf.columns:
                    base_trade_pnl_col = c
                    break

            if base_trade_pnl_col is None:
                # Fall back to zero if no suitable column exists.
                tdf["trade_net_pnl"] = 0.0
                base_trade_pnl_col = "trade_net_pnl"

            base = pd.to_numeric(tdf[base_trade_pnl_col], errors="coerce").fillna(0.0).astype(float)
            funding = pd.to_numeric(tdf["funding_net_pnl"], errors="coerce").fillna(0.0).astype(float)

            total = base + funding

            # If the trade summary already contains liquidation adjustments, include them when recomputing totals.
            if "liquidation_net_pnl" in tdf.columns:
                total = total + pd.to_numeric(tdf["liquidation_net_pnl"], errors="coerce").fillna(0.0).astype(float)

            tdf["trade_net_pnl"] = base
            tdf["total_net_pnl"] = total

        # --- Fees ---
        if "fees_total" not in tdf.columns:
            tdf["fees_total"] = 0.0
        else:
            tdf["fees_total"] = pd.to_numeric(tdf["fees_total"], errors="coerce").fillna(0.0)

        # --- Return percentage ---
        if "peak_notional" in tdf.columns:
            pn = pd.to_numeric(tdf["peak_notional"], errors="coerce").replace(0.0, np.nan)
            tdf["return_pct"] = (tdf["total_net_pnl"].astype(float) / pn).fillna(0.0)
        else:
            tdf["return_pct"] = 0.0

        return tdf

    def expectancy_by_regime(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()

        df = trades_df.copy()
        if "entry_regime" not in df.columns:
            df["entry_regime"] = "UNKNOWN"

        pnl_col = "total_net_pnl" if "total_net_pnl" in df.columns else "trade_net_pnl"
        if pnl_col not in df.columns:
            return pd.DataFrame()

        agg = df.groupby("entry_regime", dropna=False).agg(
            trade_count=("ticker", "count"),
            total_pnl=(pnl_col, "sum"),
            avg_pnl=(pnl_col, "mean"),
            avg_return=("return_pct", "mean") if "return_pct" in df.columns else (pnl_col, "mean"),
            win_rate=(pnl_col, lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
        )
        return agg.sort_values("avg_pnl", ascending=False)

    def audit_replay_cash_recursion(self, ledger_df: pd.DataFrame, fills_df: pd.DataFrame, tol: float = 1e-3) -> None:
        ledger_df = self._validate_and_normalize_ledger(ledger_df)
        # Fills can be empty when no trades/funding were produced (common for strict strategies).
        # Audits should never make a run fail in that case.
        fills_df = self._prep_fills(fills_df)
        if fills_df is None or fills_df.empty:
            return

        # Defensive: accept alternative timestamp column names if upstream changes.
        ts_col = "timestamp"
        if ts_col not in fills_df.columns:
            for alt in ["ts_utc", "time", "datetime", "date_time"]:
                if alt in fills_df.columns:
                    ts_col = alt
                    break
            else:
                return

        cf = fills_df.copy()
        cf["date"] = pd.to_datetime(cf[ts_col]).dt.normalize()
        cf_by_day = cf.groupby("date")["net_cashflow"].sum()

        ledger_cash = ledger_df["cash_balance"].copy()
        ledger_cash.index = pd.to_datetime(ledger_cash.index).normalize()

        common = ledger_cash.index.intersection(cf_by_day.index)
        if len(common) < 2:
            return

        actual = ledger_cash.loc[common].astype(float)
        expected = ledger_cash.shift(1).loc[common].astype(float) + cf_by_day.loc[common].astype(float)

        diff = (actual - expected).iloc[1:].abs().max()
        if not (diff < float(tol)):
            raise AssertionError(f"Cash recursion failed: max abs diff {diff}")

    def rebuild_trades_from_fills(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        fills_df = self._prep_fills(fills_df)

        # If there are no fills (or no normalized columns), there are no trades to rebuild.
        if fills_df is None or fills_df.empty or "event_type" not in fills_df.columns:
            return pd.DataFrame()

        tf = fills_df[fills_df["event_type"].astype(str).str.upper() == "TRADE"].copy()
        if tf.empty:
            return pd.DataFrame()

        tf["timestamp"] = pd.to_datetime(tf["timestamp"])
        tf["side"] = tf["side"].astype(str).str.upper()

        active: Dict[str, list] = {}
        completed = []

        for _, r in tf.sort_values("timestamp").iterrows():
            t = str(r.get("ticker", ""))
            active.setdefault(t, []).append(r)

            net_qty = 0.0
            for x in active[t]:
                if str(x.get("side", "")).upper() == "BUY":
                    net_qty += float(x.get("qty", 0.0))
                else:
                    net_qty -= float(x.get("qty", 0.0))

            if abs(net_qty) < 1e-8:
                fills = active.pop(t)
                buys = [x for x in fills if str(x.get("side", "")).upper() == "BUY"]
                sells = [x for x in fills if str(x.get("side", "")).upper() == "SELL"]
                if not buys or not sells:
                    continue

                entry_time = pd.to_datetime(buys[0]["timestamp"])
                exit_time = pd.to_datetime(sells[-1]["timestamp"])

                trade_net_pnl = float(pd.Series([float(x.get("net_cashflow", 0.0)) for x in fills]).sum())
                fees_total = float(pd.Series([float(x.get("fee_cashflow", 0.0)) for x in fills]).sum())
                peak_notional = float(pd.Series([float(x.get("gross_notional", 0.0)) for x in buys]).max())

                entry_regime = str(buys[0].get("regime", "UNKNOWN"))
                exit_reason = str(sells[-1].get("reason", ""))

                completed.append(
                    {
                        "ticker": t,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_regime": entry_regime,
                        "exit_reason": exit_reason,
                        "trade_net_pnl": trade_net_pnl,
                        "fees_total": fees_total,
                        "peak_notional": peak_notional,
                    }
                )

        return pd.DataFrame(completed)

    def _compute_timeseries_metrics(self, ledger_df: pd.DataFrame) -> Dict[str, float]:
        eq = ledger_df["equity"].astype(float).copy()
        eq.index = pd.to_datetime(eq.index)
        eq = eq.sort_index()

        returns = eq.pct_change().dropna()

        # Handle too-few observations safely (std is NaN with 1 datapoint)
        if returns.empty:
            return dict(
                final_equity=float(eq.iloc[-1]),
                cagr=0.0,
                ann_vol=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_drawdown=0.0,
            )

        days = max((eq.index[-1] - eq.index[0]).days, 1)
        years = days / 365.25
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        # Volatility
        if len(returns) < 2:
            ret_std = 0.0
        else:
            ret_std = float(np.nan_to_num(returns.std(ddof=0)))

        ann_vol = float(ret_std * np.sqrt(self.trading_days))

        excess = returns - self.rf_daily
        # Guard against near-zero volatility leading to meaningless extreme Sharpe values
        if ret_std is not None and ret_std < 1e-8:
            ret_std = 0.0
        sharpe = float((excess.mean() / ret_std) * np.sqrt(self.trading_days)) if ret_std > 0 else 0.0

        downside = returns[returns < 0]
        if len(downside) < 2:
            downside_std = 0.0
        else:
            downside_std = float(np.nan_to_num(downside.std(ddof=0)))
        sortino = float((excess.mean() / downside_std) * np.sqrt(self.trading_days)) if downside_std > 0 else 0.0

        cum_max = eq.cummax()
        dd = (eq / cum_max) - 1.0
        max_dd = float(dd.min())

        return dict(
            final_equity=float(eq.iloc[-1]),
            cagr=float(cagr),
            ann_vol=float(ann_vol),
            sharpe=float(sharpe),
            sortino=float(sortino),
            max_drawdown=float(max_dd),
        )

    def _compute_trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        if trades_df is None or trades_df.empty:
            return dict(total_trades=0, win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0)

        pnl_col = "total_net_pnl" if "total_net_pnl" in trades_df.columns else (
            "trade_net_pnl" if "trade_net_pnl" in trades_df.columns else None
        )
        if pnl_col is None or pnl_col not in trades_df.columns:
            return dict(total_trades=int(len(trades_df)), win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0)

        pnl = pd.to_numeric(trades_df[pnl_col], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.abs().sum()) if not losses.empty else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0

        return dict(total_trades=int(len(trades_df)), win_rate=win_rate, profit_factor=profit_factor, avg_win=avg_win, avg_loss=avg_loss)

    @staticmethod
    def _prep_fills(fills_df: pd.DataFrame) -> pd.DataFrame:
        if fills_df is None or fills_df.empty:
            return pd.DataFrame()

        df = fills_df.copy()

        # Accept common timestamp column variants
        if "timestamp" not in df.columns:
            for alt in ["ts_utc", "time", "datetime", "DateTime", "date_time"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "timestamp"})
                    break
        if "timestamp" not in df.columns:
            raise ValueError("fills_df must include a timestamp column (timestamp or ts_utc)")

        if "event_type" not in df.columns:
            df["event_type"] = "TRADE"

        if "ticker" not in df.columns:
            df["ticker"] = ""

        if "side" not in df.columns:
            df["side"] = ""

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        for col in ["net_cashflow", "fee_cashflow", "margin_change", "gross_notional", "qty", "price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        df["event_type"] = df["event_type"].astype(str).str.upper()
        df["side"] = df["side"].astype(str).str.upper()

        return df

    @staticmethod
    def _validate_and_normalize_ledger(ledger_df: pd.DataFrame) -> pd.DataFrame:
        if ledger_df is None or ledger_df.empty:
            raise ValueError("ledger_df is empty")

        df = ledger_df.copy()
        df.index = pd.to_datetime(df.index)

        if "equity" not in df.columns:
            raise ValueError("ledger_df must include 'equity'")

        if "cash_balance" not in df.columns:
            if "cash" in df.columns:
                df["cash_balance"] = df["cash"]
            else:
                raise ValueError("ledger_df must include either 'cash_balance' or 'cash'")

        df["equity"] = pd.to_numeric(df["equity"], errors="coerce").astype(float)
        df["cash_balance"] = pd.to_numeric(df["cash_balance"], errors="coerce").astype(float)

        return df