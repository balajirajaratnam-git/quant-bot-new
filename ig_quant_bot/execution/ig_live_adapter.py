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

        acc_type = str(config.get("acc_type", "DEMO")).upper()
        if acc_type == "LIVE" and not bool(config.get("allow_live", False)):
            raise ValueError(
                "Refusing to connect to IG LIVE because live.allow_live is false. "
                "Set live.allow_live: true explicitly when you are ready."
            )

        try:
            from trading_ig import IGService
        except Exception as e:
            raise ImportError("trading-ig is required for IGLiveAdapter. Install via requirements.txt") from e

        self.ig = IGService(
            config["username"],
            config["password"],
            config["api_key"],
            acc_type,
        )
        self.session = self.ig.create_session()
        self.account_id = config.get("acc_number")

        # --- Rate limiting (protects IG API quotas) ---
        rate_cfg = (config.get("rate_limit", {}) or {})
        # Minimum time between calls of the same type (history, market etc.)
        self._min_interval_s = float(rate_cfg.get("min_interval_seconds", 2.0))
        # If IG signals a quota issue, pause before retrying.
        self._cooldown_on_exceeded_s = float(rate_cfg.get("cooldown_seconds_on_exceeded", 70.0))
        self._max_retries_on_exceeded = int(rate_cfg.get("max_retries", 1))
        self._last_call_ts: Dict[str, float] = {}

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
                msg = str(e)
                # trading-ig throws this when switching to the already-selected account
                if "accountId-must-be-different" in msg or "must-be-different" in msg:
                    self.logger.info(f"IG account already selected ({self.account_id}); no switch needed")
                else:
                    self.logger.warning(f"Could not switch IG account to {self.account_id}: {e}")

    def _throttle(self, key: str) -> None:
        """Enforce a simple minimum interval per call type."""
        now = time.time()
        last = float(self._last_call_ts.get(key, 0.0))
        wait = float(self._min_interval_s) - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts[key] = time.time()


    def _round_level(self, level: Optional[float], ref_price: Optional[float]) -> Optional[float]:
        """Round stop/limit levels to a sensible precision to avoid IG precision rejections.
        Uses a heuristic based on ref_price magnitude and falls back to 2 decimals.
        """
        if level is None:
            return None
        try:
            price = float(ref_price) if ref_price is not None else None
            if price is None:
                decimals = 2
            elif abs(price) >= 1000:
                decimals = 1
            elif abs(price) >= 100:
                decimals = 2
            elif abs(price) >= 1:
                decimals = 4
            else:
                decimals = 5
            return round(float(level), decimals)
        except Exception:
            return level

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
                sl = self._round_level(float(order.stop_level), float(price))
                kwargs['stop_level'] = float(sl if sl is not None else order.stop_level)
            if order.limit_level is not None:
                tp = self._round_level(float(order.limit_level), float(price))
                kwargs['limit_level'] = float(tp if tp is not None else order.limit_level)

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

    def _normalize_resolution(self, resolution: str) -> str:
        """Normalize resolution strings for trading-ig compatibility.

        trading-ig uses pandas to_offset() under the hood, so resolution must be a
        pandas-offset-like string (e.g., "D", "1H", "5Min").
        """
        if resolution is None:
            return "D"
        r = str(resolution).strip()
        if not r:
            return "D"
        ru = r.upper()
        mapping = {
            "DAY": "D",
            "DAILY": "D",
            "D": "D",
            "1D": "D",
            "WEEK": "W",
            "W": "W",
            "1W": "W",
            "MONTH": "M",
            "M": "M",
            "1M": "M",
            "HOUR": "1H",
            "H": "1H",
            "1H": "1H",
            "HOUR_2": "2H",
            "2H": "2H",
            "HOUR_3": "3H",
            "3H": "3H",
            "HOUR_4": "4H",
            "4H": "4H",
            "MIN": "1Min",
            "MINUTE": "1Min",
            "1MIN": "1Min",
            "1MINUTE": "1Min",
            "1T": "1Min",
            "MIN_5": "5Min",
            "5MIN": "5Min",
            "MINUTE_5": "5Min",
            "5T": "5Min",
            "MIN_15": "15Min",
            "15MIN": "15Min",
            "MINUTE_15": "15Min",
            "MIN_30": "30Min",
            "30MIN": "30Min",
            "MINUTE_30": "30Min",
        }
        return mapping.get(ru, r)

    def fetch_historical_prices(self, epic: str, *, resolution: str, numpoints: int):
        """Fetch historical prices robustly across trading-ig versions.

        Different trading-ig releases use different parameter names:
          - numpoints
          - num_points
        Some also accept numpoints positionally.
        """
        resolution_n = self._normalize_resolution(resolution)

        def _call_history():
            # Preferred: keyword 'numpoints'
            try:
                return self.ig.fetch_historical_prices_by_epic(epic, resolution=resolution_n, numpoints=numpoints)
            except TypeError:
                pass

            # Fallback: keyword 'num_points'
            try:
                return self.ig.fetch_historical_prices_by_epic(epic, resolution=resolution_n, num_points=numpoints)
            except TypeError:
                pass

            # Fallback: positional
            return self.ig.fetch_historical_prices_by_epic(epic, resolution_n, numpoints)

        # Rate limit and retry on ApiExceededException.
        retries = max(0, int(self._max_retries_on_exceeded))
        for attempt in range(retries + 1):
            self._throttle("history")
            try:
                return _call_history()
            except Exception as e:
                if e.__class__.__name__ == "ApiExceededException" and attempt < retries:
                    self.logger.warning(
                        f"IG quota exceeded on history fetch; cooling down for {self._cooldown_on_exceeded_s:.0f}s then retrying."
                    )
                    time.sleep(float(self._cooldown_on_exceeded_s))
                    continue
                raise

    def fetch_market(self, epic: str):
        """Fetch market details for an epic across trading-ig versions."""
        self._throttle('market')
        if hasattr(self.ig, "fetch_market_by_epic"):
            return self.ig.fetch_market_by_epic(epic)
        if hasattr(self.ig, "fetch_market"):
            # Some versions use fetch_market(epic)
            return self.ig.fetch_market(epic)
        raise AttributeError("IG client has no fetch_market_by_epic/fetch_market method")

    def search_markets(self, search_term: str):
        """Search for markets across trading-ig versions."""
        self._throttle('search')
        if hasattr(self.ig, "search_markets"):
            return self.ig.search_markets(search_term)
        if hasattr(self.ig, "searchMarkets"):
            return self.ig.searchMarkets(search_term)
        raise AttributeError("IG client has no search_markets method")
