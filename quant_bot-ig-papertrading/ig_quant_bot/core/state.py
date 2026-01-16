from __future__ import annotations

from typing import Dict

from ig_quant_bot.core.contracts import FillEvent


class PortfolioState:
    """
    V7.2 Institutional State Engine.

    Maintains the 'Three Truths':
      - cash_balance
      - margin_used
      - equity (= cash_balance + unrealized_pnl)

    Notes:
      - In this framework, FillEvent.net_cashflow is the single signed cash delta.
      - We also increment equity by net_cashflow immediately to keep the state coherent
        between MTM refreshes. update_mtm remains the source of truth for unrealized PnL.
    """

    def __init__(self, initial_cash: float, allow_shorting: bool = False, *, strict_reconciliation: bool = True):
        self.cash_balance = float(initial_cash)
        self.margin_used = 0.0

        # {ticker: {'qty','avg_price','margin','last_p','unrealized_pnl'}}
        # qty is Stake (£/pt) for IG Spread Betting
        self.positions: Dict[str, dict] = {}

        self.last_prices: Dict[str, float] = {}
        self.equity = float(initial_cash)
        self.free_margin = float(initial_cash)
        self.peak_equity = float(initial_cash)

        self.allow_shorting = bool(allow_shorting)
        self.strict_reconciliation = bool(strict_reconciliation)

    def update_mtm(self, prices: Dict[str, float], catalog) -> None:
        """Recompute equity using latest prices (or cached prices as fallbacks)."""
        if prices:
            self.last_prices.update({k: float(v) for k, v in prices.items() if v is not None})

        unrealized_pnl = 0.0

        for t, pos in self.positions.items():
            curr_p = prices.get(t)
            if curr_p is None:
                curr_p = pos.get("last_p")
            if curr_p is None:
                curr_p = self.last_prices.get(t)
            if curr_p is None:
                curr_p = pos["avg_price"]

            curr_p = float(curr_p)
            pos["last_p"] = curr_p
            self.last_prices[t] = curr_p

            inst = catalog.get(t)
            pnl = (curr_p - float(pos["avg_price"])) * float(pos["qty"]) * float(inst.value_per_point)
            pos["unrealized_pnl"] = pnl
            unrealized_pnl += pnl

        self.equity = float(self.cash_balance) + float(unrealized_pnl)
        self.free_margin = float(self.equity) - float(self.margin_used)
        self.peak_equity = max(float(self.peak_equity), float(self.equity))

    def apply_fill(self, fill: FillEvent) -> None:
        """Apply a FillEvent to cash, margin and positions with a reconciliation gate."""
        # 1) Global financials
        self.cash_balance += float(fill.net_cashflow)
        self.margin_used += float(fill.margin_change)

        # Keep equity coherent between MTM refreshes
        self.equity += float(fill.net_cashflow)

        # 2) Position state
        if fill.event_type == "TRADE":
            if fill.side == "BUY":
                if fill.ticker not in self.positions:
                    self.positions[fill.ticker] = {
                        "qty": 0.0,
                        "avg_price": 0.0,
                        "margin": 0.0,
                        "last_p": float(fill.price),
                        "unrealized_pnl": 0.0,
                        "deal_id": str(fill.fill_id) if getattr(fill, 'fill_id', None) else None,
                        "entry_date": None,
                        "stop_price": None,
                        "take_profit_price": None,
                        "max_hold_days": None,
                        "entry_reason": None,
                    }

                pos = self.positions[fill.ticker]

                new_qty = float(pos["qty"]) + float(fill.qty)
                if (not self.allow_shorting) and new_qty < -1e-10:
                    raise AssertionError("Shorting detected in Long-Only mode!")

                # Weighted average price by stake
                if abs(new_qty) > 1e-12:
                    pos["avg_price"] = (
                        (float(pos["qty"]) * float(pos["avg_price"])) + (float(fill.qty) * float(fill.price))
                    ) / new_qty
                pos["qty"] = new_qty
                pos["margin"] += float(fill.margin_change)
                pos["last_p"] = float(fill.price)
                # Keep the latest deal_id for live trading reconciliation
                if getattr(fill, 'fill_id', None):
                    pos["deal_id"] = str(fill.fill_id)
                self.last_prices[fill.ticker] = float(fill.price)

            elif fill.side == "SELL":
                if fill.ticker not in self.positions:
                    raise KeyError(f"State Critical: Attempted to sell {fill.ticker} not in portfolio.")

                pos = self.positions[fill.ticker]

                if (not self.allow_shorting) and float(fill.qty) > float(pos["qty"]) + 1e-8:
                    raise AssertionError("Over-sell detected in Long-Only mode!")

                pos["margin"] += float(fill.margin_change)  # negative on release
                pos["qty"] = float(pos["qty"]) - float(fill.qty)
                pos["last_p"] = float(fill.price)
                # Keep the latest deal_id for live trading reconciliation
                if getattr(fill, 'fill_id', None):
                    pos["deal_id"] = str(fill.fill_id)
                self.last_prices[fill.ticker] = float(fill.price)

                if abs(float(pos["qty"])) < 1e-8:
                    del self.positions[fill.ticker]

        # 3) Reconciliation gate for margin
        if self.strict_reconciliation:
            calc_margin = sum(float(p.get("margin", 0.0)) for p in self.positions.values())
            if abs(float(self.margin_used) - float(calc_margin)) > 1e-4:
                raise AssertionError(f"Margin Mismatch: Global {self.margin_used} vs Sum {calc_margin}")

        self.free_margin = float(self.equity) - float(self.margin_used)
        self.peak_equity = max(float(self.peak_equity), float(self.equity))

    def get_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (float(self.equity) / float(self.peak_equity)) - 1.0
