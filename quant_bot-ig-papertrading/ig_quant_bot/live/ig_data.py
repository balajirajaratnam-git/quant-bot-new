from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


def _mid_from_price_obj(price_obj: Dict) -> Optional[float]:
    """Extract a mid price from IG's price object dict.

    IG responses often use nested dicts like:
      {"bid": 123.4, "ask": 123.6, "lastTraded": 123.5}
    """
    if not isinstance(price_obj, dict):
        return None
    bid = price_obj.get("bid")
    ask = price_obj.get("ask")
    last = price_obj.get("lastTraded")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except Exception:
            pass
    if last is not None:
        try:
            return float(last)
        except Exception:
            return None
    return None


def prices_to_ohlc_df(resp: Dict, *, tz: str = "UTC") -> pd.DataFrame:
    """Convert trading-ig fetch_historical_prices response into an OHLC DataFrame.

    The exact shape of trading-ig responses can vary by version.
    This converter is intentionally defensive.
    """
    if resp is None:
        return pd.DataFrame()

    prices = None
    if isinstance(resp, dict):
        prices = resp.get("prices")
    if prices is None and hasattr(resp, "get"):
        try:
            prices = resp.get("prices")
        except Exception:
            prices = None

    # Some versions return a DataFrame directly
    if isinstance(resp, pd.DataFrame):
        df = resp.copy()
        # Try to normalise column names
        for c in ["open", "high", "low", "close"]:
            if c in df.columns and c.capitalize() not in df.columns:
                df[c.capitalize()] = df[c]
        if "Open" in df.columns and "Close" in df.columns:
            df.index = pd.to_datetime(df.index)
            df["is_real_bar"] = True
            return df[["Open", "High", "Low", "Close", "is_real_bar"]].dropna(how="any")

    if not isinstance(prices, list):
        return pd.DataFrame()

    rows = []
    for p in prices:
        if not isinstance(p, dict):
            continue
        ts = p.get("snapshotTimeUTC") or p.get("snapshotTime") or p.get("time")
        if ts is None:
            continue
        try:
            dt = pd.to_datetime(ts, utc=True)
        except Exception:
            continue

        open_px = _mid_from_price_obj(p.get("openPrice") or {})
        high_px = _mid_from_price_obj(p.get("highPrice") or {})
        low_px = _mid_from_price_obj(p.get("lowPrice") or {})
        close_px = _mid_from_price_obj(p.get("closePrice") or {})

        if any(v is None for v in [open_px, high_px, low_px, close_px]):
            continue

        rows.append(
            {
                "dt": dt,
                "Open": float(open_px),
                "High": float(high_px),
                "Low": float(low_px),
                "Close": float(close_px),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("dt").sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df["is_real_bar"] = True
    return df


def extract_quote_mid(resp: Dict) -> Optional[float]:
    """Extract a mid quote from trading-ig fetch_market_by_epic response."""
    if not isinstance(resp, dict):
        return None

    # Common shapes: resp["snapshot"]["bid"]/"offer" or resp["snapshot"]["bid"]/"ask"
    snapshot = resp.get("snapshot") or resp.get("Snapshot") or {}
    if isinstance(snapshot, dict):
        bid = snapshot.get("bid")
        offer = snapshot.get("offer")
        ask = snapshot.get("ask")
        if bid is not None and (offer is not None or ask is not None):
            try:
                a = float(offer) if offer is not None else float(ask)
                return (float(bid) + a) / 2.0
            except Exception:
                pass
        last = snapshot.get("marketStatus")  # not a price

    # Fallback: look for bid/offer at root
    bid = resp.get("bid")
    offer = resp.get("offer") or resp.get("ask")
    if bid is not None and offer is not None:
        try:
            return (float(bid) + float(offer)) / 2.0
        except Exception:
            return None

    return None
