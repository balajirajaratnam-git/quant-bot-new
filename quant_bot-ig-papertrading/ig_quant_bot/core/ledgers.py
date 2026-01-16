from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ig_quant_bot.core.contracts import FillEvent


class TradeLedger:
    """
    V7.2 Institutional Audit Ledger.

    Responsibilities:
      1) Persist all FillEvents (TRADE + FUNDING + FEE + LIQUIDATION) as an auditable stream.
      2) Aggregate TRADE fills into completed round-trip trades (per ticker).
      3) Attribute out-of-band cashflows (FUNDING/FEE/LIQUIDATION adjustments) to each trade
         using ticker + entry_time/exit_time window matching.

    Notes:
      - For IG Spread Betting, qty is stake (£/pt).
      - For this framework, net_cashflow is the single cash delta applied to cash_balance.
      - Funding events are event_type="FUNDING", side="" and usually net_cashflow < 0.
    """

    def __init__(self):
        self._active_trade_fills: Dict[str, List[FillEvent]] = {}
        self._all_non_trade_events: List[FillEvent] = []
        self.completed_trades: List[dict] = []
        self.all_fills: List[FillEvent] = []

    def add_fill(self, fill: FillEvent) -> None:
        self.all_fills.append(fill)

        if str(fill.event_type).upper() != "TRADE":
            self._all_non_trade_events.append(fill)
            return

        t = fill.ticker
        self._active_trade_fills.setdefault(t, []).append(fill)

        net_qty = 0.0
        for f in self._active_trade_fills[t]:
            if str(f.side).upper() == "BUY":
                net_qty += float(f.qty)
            elif str(f.side).upper() == "SELL":
                net_qty -= float(f.qty)

        if abs(net_qty) < 1e-8:
            fills = self._active_trade_fills.pop(t)
            self._summarize_trade_with_attribution(fills)

    def _summarize_trade_with_attribution(self, fills: List[FillEvent]) -> None:
        if not fills:
            return

        buys = [f for f in fills if str(f.side).upper() == "BUY"]
        sells = [f for f in fills if str(f.side).upper() == "SELL"]
        if not buys or not sells:
            return

        ticker = fills[0].ticker
        entry_time = pd.to_datetime(buys[0].timestamp)
        exit_time = pd.to_datetime(sells[-1].timestamp)

        trade_net_pnl = float(sum(float(f.net_cashflow) for f in fills))
        trade_fees = float(sum(float(f.fee_cashflow) for f in fills))

        funding_cost = 0.0
        extra_fees = 0.0
        liquidation_adj = 0.0

        for e in self._all_non_trade_events:
            if e.ticker != ticker:
                continue
            ts = pd.to_datetime(e.timestamp)
            if ts < entry_time or ts > exit_time:
                continue

            et = str(e.event_type).upper()
            if et == "FUNDING":
                funding_cost += float(e.net_cashflow)
            elif et == "FEE":
                extra_fees += float(e.net_cashflow)
            elif et == "LIQUIDATION":
                liquidation_adj += float(e.net_cashflow)

        peak_notional = float(max(float(f.gross_notional) for f in buys)) if buys else 0.0

        total_net_pnl = trade_net_pnl + funding_cost + extra_fees + liquidation_adj

        entry_regime = str(buys[0].regime)
        exit_reason = str(sells[-1].reason)

        self.completed_trades.append(
            {
                "ticker": ticker,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_regime": entry_regime,
                "exit_reason": exit_reason,
                "trade_net_pnl": float(trade_net_pnl),
                "funding_net_pnl": float(funding_cost),
                "fees_total": float(trade_fees + extra_fees),
                "liquidation_net_pnl": float(liquidation_adj),
                "total_net_pnl": float(total_net_pnl),
                "peak_notional": float(peak_notional),
                "return_pct": (float(total_net_pnl) / float(peak_notional)) if peak_notional > 0 else 0.0,
            }
        )

    def get_trades_df(self) -> pd.DataFrame:
        if not self.completed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.completed_trades)
