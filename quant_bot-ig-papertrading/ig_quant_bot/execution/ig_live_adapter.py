from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import pandas as pd

from ig_quant_bot.core.contracts import FillEvent, Order
from .broker_interface import IBrokerAdapter


class IGLiveAdapter(IBrokerAdapter):
    """
    Production adapter for IG using the 'trading-ig' library.

    Notes:
      - This module is not used by the backtest path.
      - For SELL fills, margin release is estimated from current_pos when IG does not
        return an explicit margin field for close confirmations.

    Expected config keys:
      username, password, api_key, acc_type (DEMO|LIVE), acc_number
    """

    def __init__(self, config: Dict, catalog):
        self.catalog = catalog
        self.logger = logging.getLogger(__name__)

        try:
            from trading_ig import IGService
        except Exception as e:
            raise ImportError("trading-ig is required for IGLiveAdapter. Install via requirements.txt") from e

        self.ig = IGService(
            config["username"],
            config["password"],
            config["api_key"],
            config.get("acc_type", "DEMO"),
        )
        self.session = self.ig.create_session()
        self.account_id = config.get("acc_number")

        # If the user provided an account id and the library supports it, switch explicitly.
        # This avoids accidental order placement on the wrong account when multiple accounts exist.
        if self.account_id and hasattr(self.ig, "switch_account"):
            try:
                # Common signature: switch_account(account_id, default_account=True)
                self.ig.switch_account(self.account_id, default_account=True)
            except TypeError:
                # Some versions only accept the account id
                self.ig.switch_account(self.account_id)
            except Exception as e:
                self.logger.warning(f"Could not switch IG account to {self.account_id}: {e}")

    def execute(
        self,
        order: Order,
        price: float,
        regime: str,
        atr_pts: float,
        current_pos: Optional[Dict] = None,
    ) -> FillEvent:
        inst = self.catalog.get(order.ticker)
        direction = "BUY" if order.side == "BUY" else "SELL"

        # size = stake (£/pt)
        if order.notional is not None:
            size = round(float(order.notional) / (float(price) * float(inst.value_per_point)), 2)
        else:
            size = float(order.qty)

        self.logger.info(f"Submitting {direction} {inst.epic} | size={size}")

        # If this is a SELL intended to close, prefer IG close endpoint when possible.
        if order.side == "SELL" and current_pos is not None and current_pos.get("deal_id"):
            deal_id = str(current_pos.get("deal_id"))
            self.logger.info(f"Closing position {deal_id} | epic={inst.epic} | size={size}")
            if hasattr(self.ig, 'close_open_position'):
                resp = self.ig.close_open_position(
                    deal_id=deal_id,
                    direction=direction,
                    size=size,
                    order_type='MARKET',
                )
            elif hasattr(self.ig, 'close_position'):
                resp = self.ig.close_position(
                    deal_id=deal_id,
                    direction=direction,
                    size=size,
                    order_type='MARKET',
                )
            else:
                raise RuntimeError('IGService does not expose a close method in this environment.')
        else:
            currency_code = str(order.currency_code or 'GBP')
            expiry = str(order.expiry or 'DFB')
            force_open = bool(order.force_open) if order.force_open is not None else True

            kwargs = {
                'currency_code': currency_code,
                'direction': direction,
                'epic': inst.epic,
                'order_type': 'MARKET',
                'expiry': expiry,
                'size': size,
                'force_open': force_open,
                'guaranteed_stop': False,
            }

            if order.stop_level is not None:
                kwargs['stop_level'] = float(order.stop_level)
            if order.limit_level is not None:
                kwargs['limit_level'] = float(order.limit_level)

            resp = self._create_open_position_safe(kwargs, price=float(price), require_stop=(order.stop_level is not None))

        deal_ref = resp.get('dealReference')
        if not deal_ref:
            raise RuntimeError(f"IG submission failed: {resp}")

        conf = self._poll_confirmation(deal_ref)
        if conf.get("dealStatus") != "ACCEPTED":
            raise RuntimeError(f"IG deal rejected: {conf.get('reason', 'Unknown')}")

        fill_price = float(conf.get("level", price))
        gross_notional = abs(float(size)) * fill_price * float(inst.value_per_point)

        if order.side == "BUY":
            margin_val = float(conf.get("marginDeposit", gross_notional * float(inst.margin_factor)))
            margin_change = margin_val
            net_cf = 0.0
            realized = 0.0
        else:
            if current_pos is None:
                raise ValueError("current_pos is required for SELL in live adapter")

            # Realized profit is sometimes returned, sometimes not.
            realized = float(conf.get("profit", 0.0))
            net_cf = realized

            # Release margin based on current held margin (best effort).
            current_qty = float(current_pos.get("qty", size))
            ratio = (float(size) / current_qty) if current_qty else 1.0
            margin_change = -(float(current_pos.get("margin", 0.0)) * ratio)

        return FillEvent(
            fill_id=str(conf.get('dealId') or conf.get('dealReference') or deal_ref or order.order_id),
            order_id=order.order_id,
            ticker=order.ticker,
            timestamp=pd.Timestamp.now(),
            event_type="TRADE",
            side=order.side,
            qty=float(size),
            price=float(fill_price),
            margin_change=float(margin_change),
            realized_pnl_cashflow=float(realized),
            net_cashflow=float(net_cf),
            gross_notional=float(gross_notional),
            regime=str(regime),
            reason=str(order.reason),
        )

    def _create_open_position_safe(self, kwargs: Dict, *, price: float, require_stop: bool) -> Dict:
        """Create an IG open position robustly across trading-ig versions.

        Strategy:
          - Try stop_level/limit_level (preferred)
          - Fallback to stop_distance/limit_distance when stop_level/limit_level are unsupported
          - Fallback to string force_open/guaranteed_stop ("true"/"false") if required

        If require_stop is True and we cannot attach a stop, we raise RuntimeError.
        """
        last_err = None

        # A sequence of increasingly compatible kwargs shapes.
        attempts = []
        attempts.append(dict(kwargs))

        # Some trading-ig versions expect "true"/"false" strings.
        kw_str = dict(kwargs)
        if "force_open" in kw_str:
            kw_str["force_open"] = "true" if bool(kw_str["force_open"]) else "false"
        if "guaranteed_stop" in kw_str:
            kw_str["guaranteed_stop"] = "true" if bool(kw_str["guaranteed_stop"]) else "false"
        attempts.append(kw_str)

        has_stop_level = "stop_level" in kwargs
        has_limit_level = "limit_level" in kwargs

        # Distance fallbacks. Distances are in points (price units).
        if has_stop_level:
            stop_level = float(kwargs["stop_level"])
            stop_distance = abs(float(price) - stop_level)
            for base in (dict(kwargs), dict(kw_str)):
                base.pop("stop_level", None)
                base["stop_distance"] = stop_distance
                attempts.append(base)

        if has_limit_level:
            limit_level = float(kwargs["limit_level"])
            limit_distance = abs(limit_level - float(price))
            for base in (dict(kwargs), dict(kw_str)):
                base.pop("limit_level", None)
                base["limit_distance"] = limit_distance
                attempts.append(base)

        if has_stop_level and has_limit_level:
            stop_level = float(kwargs["stop_level"])
            limit_level = float(kwargs["limit_level"])
            stop_distance = abs(float(price) - stop_level)
            limit_distance = abs(limit_level - float(price))
            for base in (dict(kwargs), dict(kw_str)):
                base.pop("stop_level", None)
                base.pop("limit_level", None)
                base["stop_distance"] = stop_distance
                base["limit_distance"] = limit_distance
                attempts.append(base)

        # Deduplicate attempts to avoid spam.
        seen = set()
        uniq = []
        for a in attempts:
            key = tuple(sorted(a.items()))
            if key not in seen:
                seen.add(key)
                uniq.append(a)

        for a in uniq:
            try:
                return self.ig.create_open_position(**a)
            except TypeError as e:
                last_err = e
                continue

        if require_stop:
            raise RuntimeError(f"Could not attach protective stop on IG open position; last_error={last_err}")
        raise RuntimeError(f"IG create_open_position failed; last_error={last_err}")

    def _poll_confirmation(self, deal_ref: str, timeout: int = 15) -> Dict:
        start = time.time()
        while time.time() - start < timeout:
            conf = self.ig.fetch_deal_confirmation(deal_ref)
            if conf and conf.get("dealStatus"):
                return conf
            time.sleep(0.5)
        raise TimeoutError(f"IG confirmation timeout for dealRef: {deal_ref}")

    def calculate_funding(self, ticker: str, pos: Dict, price: float, date: pd.Timestamp) -> FillEvent:
        # In live, funding is applied by IG. This is a placeholder for reconciliation.
        return FillEvent(
            ticker=ticker,
            timestamp=pd.to_datetime(date),
            event_type="FUNDING",
            net_cashflow=0.0,
            gross_notional=0.0,
            reason="LIVE_FUNDING_RECONCILE_TODO",
        )
