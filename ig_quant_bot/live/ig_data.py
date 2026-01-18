from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

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

    def _normalize_dt_index(index_like: Any) -> pd.DatetimeIndex:
        """Convert an index-like object to a timezone-aware DatetimeIndex.

        trading-ig may return:
          - DatetimeIndex
          - MultiIndex (timestamps embedded in one level)
          - RangeIndex / integer index (rare)

        If conversion fails, we synthesize a daily index ending "now".
        """
        # Direct DatetimeIndex
        if isinstance(index_like, pd.DatetimeIndex):
            idx = index_like
        elif isinstance(index_like, pd.MultiIndex):
            # Try levels and pick the one that parses best.
            best_idx = None
            best_non_nat = -1
            for lvl in range(index_like.nlevels):
                cand = pd.to_datetime(index_like.get_level_values(lvl), errors="coerce", utc=True)
                non_nat = int(cand.notna().sum())
                if non_nat > best_non_nat:
                    best_non_nat = non_nat
                    best_idx = cand
            idx = best_idx if best_idx is not None else pd.to_datetime([], utc=True)
        else:
            idx = pd.to_datetime(index_like, errors="coerce", utc=True)

        # Ensure DatetimeIndex
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)

        # If all NaT, synthesize an index (keeps ordering, enables indicators).
        if len(idx) > 0 and idx.isna().all():
            end = pd.Timestamp.utcnow().floor("min")
            idx = pd.date_range(end=end, periods=len(index_like), freq="D", tz="UTC")

        # Normalize timezone
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return idx

    # Some versions return prices as a DataFrame under resp["prices"]
    if isinstance(prices, pd.DataFrame):
        dfp = prices.copy()

        # trading-ig commonly returns a DataFrame with tuple-like column labels
        # representing a 2-level structure: ("bid"|"ask"|"last", "Open"|"High"|"Low"|"Close").
        # In some environments this is a proper MultiIndex, in others it's an Index of tuples.
        def _is_tuple_cols() -> bool:
            try:
                if isinstance(dfp.columns, pd.MultiIndex):
                    return True
                return len(dfp.columns) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in dfp.columns)
            except Exception:
                return False

        def _col_tuple(level0: str, level1: str) -> Optional[Tuple[str, str]]:
            """Return an existing 2-level column key (tuple) case-insensitively."""
            target0 = str(level0).lower()
            target1 = str(level1).lower()
            for c in dfp.columns:
                if not (isinstance(c, tuple) and len(c) == 2):
                    continue
                if str(c[0]).lower() == target0 and str(c[1]).lower() == target1:
                    return (c[0], c[1])
            return None

        # Determine timestamp source. In some trading-ig versions the index is a RangeIndex
        # and timestamps are stored in a column (snapshotTimeUTC/snapshotTime).
        ts_col = None
        for cand in ("snapshotTimeUTC", "snapshotTime", "time", "timestamp"):
            if cand in dfp.columns:
                ts_col = cand
                break

        if ts_col is not None:
            idx = pd.to_datetime(dfp[ts_col], utc=True, errors="coerce")
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.DatetimeIndex(idx)
        else:
            idx = _normalize_dt_index(dfp.index)

        # IMPORTANT: trading-ig often returns a naive DatetimeIndex while we normalize
        # to a timezone-aware index. If we keep the original dfp index, pandas will
        # align on index labels during assignment and silently produce all-NaN outputs.
        # To avoid that, we overwrite dfp.index with the normalized index so downstream
        # Series line up exactly with `out`.
        try:
            if len(idx) == len(dfp.index):
                dfp.index = idx
        except Exception:
            pass

        # Case 0: 2-level bid/ask/last columns like ('bid','Open') etc.
        # Your debug output shows exactly this shape.
        if _is_tuple_cols():
            # Try common variants for ask-level naming.
            ask_level = "ask" if any(str(c[0]).lower() == "ask" for c in dfp.columns if isinstance(c, tuple)) else "offer"
            needed = ["Open", "High", "Low", "Close"]
            out = pd.DataFrame(index=idx)

            ok = True
            for f in needed:
                bid_key = _col_tuple("bid", f)
                ask_key = _col_tuple(ask_level, f)
                last_key = _col_tuple("last", f)

                # Some IG markets provide historical "last" but not historical bid/ask.
                # We prefer mid(bid, ask) when available and fall back to last when bid/ask are missing.
                series = None

                if bid_key is not None and ask_key is not None:
                    b = pd.to_numeric(dfp[bid_key], errors="coerce")
                    a = pd.to_numeric(dfp[ask_key], errors="coerce")
                    mid = (b + a) / 2.0
                    # If bid/ask are entirely missing, fall back to last.
                    if mid.notna().any():
                        series = mid

                if series is None and last_key is not None:
                    last = pd.to_numeric(dfp[last_key], errors="coerce")
                    if last.notna().any():
                        series = last

                if series is None:
                    ok = False
                    break

                out[f] = series

            if ok:
                out = out.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"})
                out = out.dropna(subset=["Open", "High", "Low", "Close"])
                # If index parsing failed earlier, do not drop the entire dataset.
                if hasattr(out.index, "isna"):
                    out = out.loc[~out.index.isna()]
                if not out.empty:
                    out.index = out.index.tz_convert(tz)
                    out["is_real_bar"] = True
                    return out[["Open", "High", "Low", "Close", "is_real_bar"]]

        def _col(base: str, side: str) -> Optional[str]:
            """Find a bid/ask/offer column across common naming variations."""
            b = base
            s = side
            candidates = [
                f"{b}.{s}",
                f"{b}_{s}",
                f"{b}{s}",
                f"{b}{s.capitalize()}",
                f"{b.capitalize()}.{s}",
                f"{b.capitalize()}_{s}",
                f"{b.capitalize()}{s}",
                f"{b.capitalize()}{s.capitalize()}",
            ]
            # IG sometimes uses "offer" instead of ask
            if s == "ask":
                candidates += [c.replace("ask", "offer") for c in candidates]

            colset = set(dfp.columns)
            for c in candidates:
                if c in colset:
                    return c

            # Final fuzzy fallback: match suffix .bid/.ask/.offer
            sfx = s.lower() if s != "ask" else "ask"
            for c in dfp.columns:
                cl = str(c).lower().replace("_", "").replace(".", "")
                if b.lower() in cl and sfx in cl:
                    return c
                if s == "ask" and b.lower() in cl and "offer" in cl:
                    return c
            return None
        # If already in OHLC form
        if all(c in dfp.columns for c in ["Open", "High", "Low", "Close"]):
            dfp.index = idx
            dfp = dfp.dropna(subset=["Open", "High", "Low", "Close"])
            if dfp.empty:
                return pd.DataFrame()
            dfp.index = dfp.index.tz_convert(tz)
            dfp["is_real_bar"] = True
            return dfp[["Open", "High", "Low", "Close", "is_real_bar"]]

        # Case 1: bid/ask columns that trading-ig may provide (many naming variants)
        bases = {
            "Open": "openPrice",
            "High": "highPrice",
            "Low": "lowPrice",
            "Close": "closePrice",
        }

        bid_cols = {k: _col(v, "bid") for k, v in bases.items()}
        ask_cols = {k: _col(v, "ask") for k, v in bases.items()}

        if all(bid_cols.values()) and all(ask_cols.values()):
            out = pd.DataFrame(index=idx)
            for k in ("Open", "High", "Low", "Close"):
                b = bid_cols[k]
                a = ask_cols[k]
                out[k] = (pd.to_numeric(dfp[b], errors="coerce") + pd.to_numeric(dfp[a], errors="coerce")) / 2.0
            out = out.dropna(subset=["Open", "High", "Low", "Close"])
            out = out.loc[~out.index.isna()]
            if out.empty:
                return pd.DataFrame()
            out.index = out.index.tz_convert(tz)
            out["is_real_bar"] = True
            return out[["Open", "High", "Low", "Close", "is_real_bar"]]

        # Case 2: object columns openPrice/highPrice/... that contain dict-like bid/ask
        if all(c in dfp.columns for c in ["openPrice", "highPrice", "lowPrice", "closePrice"]):
            out = pd.DataFrame(index=idx)
            out["Open"] = dfp["openPrice"].apply(lambda x: _mid_from_price_obj(x) if isinstance(x, dict) else None)
            out["High"] = dfp["highPrice"].apply(lambda x: _mid_from_price_obj(x) if isinstance(x, dict) else None)
            out["Low"] = dfp["lowPrice"].apply(lambda x: _mid_from_price_obj(x) if isinstance(x, dict) else None)
            out["Close"] = dfp["closePrice"].apply(lambda x: _mid_from_price_obj(x) if isinstance(x, dict) else None)
            out = out.dropna(subset=["Open", "High", "Low", "Close"])
            out = out.loc[~out.index.isna()]
            if out.empty:
                return pd.DataFrame()
            out.index = out.index.tz_convert(tz)
            out["is_real_bar"] = True
            return out[["Open", "High", "Low", "Close", "is_real_bar"]]

        return pd.DataFrame()

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


def extract_quote_bid_ask(resp: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract (bid, ask_or_offer) from trading-ig fetch_market_by_epic response."""
    if not isinstance(resp, dict):
        return (None, None)

    snapshot = resp.get("snapshot") or resp.get("Snapshot") or {}
    if isinstance(snapshot, dict):
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        offer = snapshot.get("offer")
        try:
            b = float(bid) if bid is not None else None
        except Exception:
            b = None
        try:
            a = float(offer) if offer is not None else (float(ask) if ask is not None else None)
        except Exception:
            a = None
        if b is not None or a is not None:
            return (b, a)

    # Fallback: root keys
    bid = resp.get("bid")
    offer = resp.get("offer") or resp.get("ask")
    try:
        b = float(bid) if bid is not None else None
    except Exception:
        b = None
    try:
        a = float(offer) if offer is not None else None
    except Exception:
        a = None
    return (b, a)


def extract_market_status(resp: Dict) -> Optional[str]:
    """Extract marketStatus from trading-ig fetch_market_by_epic response."""
    if not isinstance(resp, dict):
        return None
    snapshot = resp.get("snapshot") or resp.get("Snapshot") or {}
    if isinstance(snapshot, dict):
        ms = snapshot.get("marketStatus")
        if isinstance(ms, str) and ms:
            return ms
    ms = resp.get("marketStatus")
    if isinstance(ms, str) and ms:
        return ms
    return None
