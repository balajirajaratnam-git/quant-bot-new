import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Order:
    """
    Standardized intent to trade.

    For IG Spread Betting:
      - qty represents the Stake in currency per point (e.g., £/pt)
      - notional is optional sizing intent in account currency (e.g., £ exposure)
        and is translated into stake via: stake = notional / (price * value_per_point)

    Strict Rule:
      You must specify exactly ONE of notional or qty.
    """

    ticker: str
    side: str  # BUY or SELL
    notional: Optional[float] = None
    qty: Optional[float] = None

    # --- Optional live-execution hints (ignored by the synthetic broker) ---
    # For IG, these map naturally to attached stop/limit levels on an open position.
    stop_level: Optional[float] = None
    limit_level: Optional[float] = None
    # If provided, can override the default behaviour of the live adapter.
    force_open: Optional[bool] = None
    currency_code: Optional[str] = None
    expiry: Optional[str] = None
    reason: str = ""
    timestamp: Optional[pd.Timestamp] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        # Strict XOR: you must define either Notional or Qty but not both.
        if (self.notional is None) == (self.qty is None):
            raise ValueError("Order must specify exactly one of 'notional' or 'qty'.")

        # Default timestamp if not supplied
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", pd.Timestamp.now())

        side = str(self.side).upper().strip()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order.side must be 'BUY' or 'SELL'.")

        if self.notional is not None and float(self.notional) <= 0:
            raise ValueError("Order.notional must be > 0 when provided.")

        if self.qty is not None and float(self.qty) <= 0:
            raise ValueError("Order.qty must be > 0 when provided.")

        if self.stop_level is not None and float(self.stop_level) <= 0:
            raise ValueError("Order.stop_level must be > 0 when provided.")

        if self.limit_level is not None and float(self.limit_level) <= 0:
            raise ValueError("Order.limit_level must be > 0 when provided.")

        object.__setattr__(self, "side", side)


@dataclass(frozen=True)
class FillEvent:
    """
    Audit-grade record of a financial transaction or account adjustment.

    Key economic conventions:
      - event_type separates trading fills from financial adjustments
      - side is only meaningful for TRADE events
      - qty is the Stake (£/pt) for Spread Betting
      - gross_notional is the absolute exposure magnitude:
            abs(qty) * price * value_per_point

    Cash accounting:
      - net_cashflow is the single signed delta that updates cash_balance
        (realized PnL + fees + funding)

    Margin accounting:
      - margin_change is the signed delta that updates margin_used
        (positive locks margin, negative releases)
    """

    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = "SIM"
    ticker: str = ""
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    event_type: str = "TRADE"  # TRADE, FUNDING, FEE, LIQUIDATION
    side: str = ""  # BUY, SELL, or "" for non-trade events

    qty: float = 0.0
    price: float = 0.0

    fee_cashflow: float = 0.0
    margin_change: float = 0.0
    realized_pnl_cashflow: float = 0.0
    net_cashflow: float = 0.0

    gross_notional: float = 0.0
    regime: str = "UNKNOWN"
    reason: str = ""

    def __post_init__(self):
        et = str(self.event_type).upper().strip()
        sd = str(self.side).upper().strip()

        if et not in {"TRADE", "FUNDING", "FEE", "LIQUIDATION"}:
            raise ValueError("FillEvent.event_type must be TRADE, FUNDING, FEE, or LIQUIDATION.")

        if et == "TRADE":
            if sd not in {"BUY", "SELL"}:
                raise ValueError("FillEvent.side must be BUY or SELL for TRADE events.")
        else:
            sd = ""

        object.__setattr__(self, "event_type", et)
        object.__setattr__(self, "side", sd)

        # Hard numeric coercions for audit safety
        for k in [
            "qty",
            "price",
            "fee_cashflow",
            "margin_change",
            "realized_pnl_cashflow",
            "net_cashflow",
            "gross_notional",
        ]:
            object.__setattr__(self, k, float(getattr(self, k)))

        if not pd.notna(self.timestamp):
            raise ValueError("FillEvent.timestamp must be a valid timestamp.")

        if et == "TRADE" and self.qty <= 0:
            raise ValueError("FillEvent.qty must be > 0 for TRADE events.")
