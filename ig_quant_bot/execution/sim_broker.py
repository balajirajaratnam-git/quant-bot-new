from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ig_quant_bot.core.contracts import FillEvent, Order

from .broker_interface import IBrokerAdapter


class IGSyntheticBroker(IBrokerAdapter):
    """
    High-Fidelity IG Spread Betting Simulator.

    Models:
      - adaptive spreads (base spread widened by regime + ATR)
      - margin locks/releases
      - overnight financing (DFB style) including weekend multiplier

    Conventions:
      - qty is stake (£/pt)
      - BUY net_cashflow is 0 (margin is a lock, not a cash movement)
      - SELL net_cashflow equals realized pnl (funding/fees are separate events)
    """

    def __init__(
        self,
        catalog,
        annual_funding_rate: float = 0.05,
        weekend_multiplier: int = 3,
        adaptive_spread_k: float = 0.10,
        commission_fixed: float = 0.0,
    ):
        self.catalog = catalog
        self.annual_funding_rate = float(annual_funding_rate)
        self.weekend_multiplier = int(weekend_multiplier)
        self.adaptive_spread_k = float(adaptive_spread_k)
        self.commission_fixed = float(commission_fixed)

    def execute(
        self,
        order: Order,
        price: float,
        regime: str,
        atr_pts: float,
        current_pos: Optional[Dict] = None,
    ) -> FillEvent:
        inst = self.catalog.get(order.ticker)

        # Adaptive spread: widen in BEAR_TREND and with volatility
        regime_mult = 1.5 if str(regime).upper() == "BEAR_TREND" else 1.0
        effective_spread = (float(inst.spread_points) * regime_mult) + (self.adaptive_spread_k * float(atr_pts))

        fill_price = float(price) + (effective_spread / 2.0) if order.side == "BUY" else float(price) - (effective_spread / 2.0)

        if order.side == "BUY":
            qty = float(order.qty) if order.qty is not None else float(order.notional) / (fill_price * float(inst.value_per_point))
            gross_notional = abs(qty) * fill_price * float(inst.value_per_point)
            margin_lock = gross_notional * float(inst.margin_factor)

            fee = -abs(self.commission_fixed) if float(self.commission_fixed) != 0.0 else 0.0

            return FillEvent(
                ticker=order.ticker,
                order_id=order.order_id,
                timestamp=pd.to_datetime(order.timestamp),
                event_type="TRADE",
                side="BUY",
                qty=float(qty),
                price=float(fill_price),
                margin_change=float(margin_lock),
                fee_cashflow=float(fee),
                net_cashflow=float(fee),
                gross_notional=float(gross_notional),
                regime=str(regime),
                reason=str(order.reason),
            )

        # SELL / Exit
        if not current_pos:
            raise ValueError(f"Simulator Error: current_pos required for SELL fill on {order.ticker}")

        current_qty = float(current_pos.get("qty", 0.0))
        if current_qty <= 0:
            raise ValueError(f"Simulator Error: invalid current_pos qty for SELL on {order.ticker}")

        qty_to_sell = float(order.qty) if order.qty is not None else current_qty
        qty_to_sell = min(qty_to_sell, current_qty)

        realized_pnl = (float(fill_price) - float(current_pos["avg_price"])) * qty_to_sell * float(inst.value_per_point)

        fee = -abs(self.commission_fixed) if float(self.commission_fixed) != 0.0 else 0.0

        # Pro-rata margin release
        release_ratio = qty_to_sell / current_qty if current_qty != 0 else 1.0
        margin_release = -(float(current_pos.get("margin", 0.0)) * release_ratio)

        gross_notional = abs(qty_to_sell) * float(fill_price) * float(inst.value_per_point)

        return FillEvent(
            ticker=order.ticker,
            order_id=order.order_id,
            timestamp=pd.to_datetime(order.timestamp),
            event_type="TRADE",
            side="SELL",
            qty=float(qty_to_sell),
            price=float(fill_price),
            margin_change=float(margin_release),
            realized_pnl_cashflow=float(realized_pnl),
            fee_cashflow=float(fee),
            net_cashflow=float(float(realized_pnl) + float(fee)),
            gross_notional=float(gross_notional),
            regime=str(regime),
            reason=str(order.reason),
        )

    def calculate_funding(self, ticker: str, pos: Dict, price: float, date: pd.Timestamp) -> FillEvent:
        """Overnight financing (DFB). Charges 3 days on Friday by default."""
        inst = self.catalog.get(ticker)

        notional = abs(float(pos.get("qty", 0.0))) * float(price) * float(inst.value_per_point)
        if notional <= 0:
            return FillEvent(ticker=ticker, timestamp=pd.to_datetime(date), event_type="FUNDING", net_cashflow=0.0, gross_notional=0.0, reason="OVERNIGHT_FINANCING")

        daily_rate = float(self.annual_funding_rate) / 365.0
        days = 1

        # Friday -> charge weekend
        # Pandas weekday: Monday=0 ... Sunday=6
        wd = pd.to_datetime(date).weekday()
        if wd == 4:
            days = int(self.weekend_multiplier)

        funding_charge = -(notional * daily_rate * float(days))

        return FillEvent(
            ticker=ticker,
            timestamp=pd.to_datetime(date),
            event_type="FUNDING",
            side="",
            net_cashflow=float(funding_charge),
            gross_notional=float(notional),
            reason="OVERNIGHT_FINANCING",
        )
