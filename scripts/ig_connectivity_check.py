from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ig_quant_bot.utils.config_loader import load_config
import ig_quant_bot
from ig_quant_bot.execution.instrument_db import InstrumentCatalog
from ig_quant_bot.execution.ig_live_adapter import IGLiveAdapter
from ig_quant_bot.live.ig_data import prices_to_ohlc_df
from ig_quant_bot.live.history_cache import load_cache, save_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Connectivity check for IG (DEMO recommended)")
    ap.add_argument("--config", default="config.yaml", help="Config path")
    ap.add_argument("--ticker", default="QQQ", help="Ticker key from instrument_db (default: QQQ)")
    ap.add_argument("--epic", default=None, help="Override epic to test without editing instrument_db")
    ap.add_argument("--resolution", default=None, help="Override resolution (default from config)")
    ap.add_argument("--numpoints", type=int, default=None, help="Override history points (default from config)")
    ap.add_argument("--debug", action="store_true", help="Print response diagnostics")
    ap.add_argument("--use-cache", action="store_true", help="Use cached OHLC if present to avoid IG quota usage")
    ap.add_argument("--force-fetch", action="store_true", help="Fetch from IG even if cache exists")
    ap.add_argument("--write-cache", action="store_true", help="Write fetched OHLC to cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    live = cfg.get("live", {}) or {}

    catalog = InstrumentCatalog()
    inst = catalog.get(str(args.ticker))

    resolution = str(args.resolution or live.get("resolution", "DAY")).upper()
    numpoints = int(args.numpoints or live.get("lookback_points", 50))

    hc = (live.get("history_cache", {}) or {})
    cache_dir = str(hc.get("dir", "vault/cache/ig_history"))
    cache_max_age = float(hc.get("max_age_hours", 24.0))

    print(f"ig_quant_bot version: {getattr(ig_quant_bot, '__version__', 'unknown')}")
    epic = str(args.epic or inst.epic)
    print(f"Connecting to IG acc_type={live.get('acc_type','DEMO')} ticker={inst.ticker} epic={epic}")

    adapter = IGLiveAdapter(live, catalog)

    # Prefer cached OHLC when requested, to preserve IG API quota during testing.
    if args.use_cache and not args.force_fetch:
        try:
            hit = load_cache(cache_dir, epic=str(epic), resolution=str(resolution), max_age_hours=cache_max_age)
            if hit is not None and not hit.df.empty:
                print(f"Using cached OHLC ({hit.source}) from: {hit.path}")
                print(f"Fetched rows: {len(hit.df)}")
                print("Last 3 rows:")
                print(hit.df.tail(3).to_string())
                print("OK")
                return 0
        except Exception:
            pass

    # Basic account call (best effort)
    try:
        if hasattr(adapter.ig, "fetch_accounts"):
            accounts = adapter.ig.fetch_accounts()
            print("Accounts fetched")
            if isinstance(accounts, dict) and accounts.get("accounts"):
                print(f"Accounts count: {len(accounts.get('accounts'))}")
    except Exception as e:
        print(f"Warning: could not fetch accounts: {e}")

    # Price history fetch
    resp = None
    ohlc = None
    fetch_err = None
    try:
        resp = adapter.fetch_historical_prices(epic, resolution=resolution, numpoints=numpoints)
        ohlc = prices_to_ohlc_df(resp, tz="UTC")
        if args.write_cache and isinstance(ohlc, pd.DataFrame) and not ohlc.empty:
            try:
                p = save_cache(cache_dir, epic=str(epic), resolution=str(resolution), df=ohlc, source="ig")
                print(f"Cached OHLC to: {p}")
            except Exception:
                pass
    except Exception as e:
        fetch_err = e
        ohlc = None

    if fetch_err is not None:
        print("Historical price fetch failed")
        print(f"Error type: {fetch_err.__class__.__name__}")
        print(f"Error: {fetch_err!r}")
        if args.debug:
            import traceback
            tb = traceback.format_exc()
            print("--- Traceback (tail) ---")
            print("\n".join(tb.splitlines()[-25:]))
        # Continue into market snapshot and market search to help the user find a working epic.
        ohlc = pd.DataFrame()

    print(f"Fetched rows: {len(ohlc) if ohlc is not None else 0}")
    if not ohlc.empty:
        print("Last 3 rows:")
        print(ohlc.tail(3).to_string())
        print("OK")
        return 0

    # Diagnostics for empty data
    if args.debug:
        print("--- Debug diagnostics ---")
        print(f"Response type: {type(resp)}")
        if isinstance(resp, dict):
            print(f"Top-level keys: {list(resp.keys())}")
            prices = resp.get("prices")
            print(f"prices type: {type(prices)}")
            if isinstance(prices, list):
                print(f"prices len: {len(prices)}")
                if prices:
                    print(f"first price keys: {list(prices[0].keys())}")
            elif hasattr(prices, "shape"):
                print(f"prices shape: {getattr(prices, 'shape', None)}")
                try:
                    cols = list(prices.columns)
                    print(f"prices columns (first 30): {cols[:30]}")
                    # Show index characteristics too (this often explains parsing issues)
                    try:
                        idx = prices.index
                        print(f"prices index type: {type(idx)}")
                        # show first/last values without dumping everything
                        if len(idx) > 0:
                            print(f"prices index first: {idx[0]}")
                            print(f"prices index last: {idx[-1]}")
                    except Exception:
                        pass

                    # Show a quick sample value for bid/ask/last if present
                    try:
                        if len(prices) > 0:
                            row0 = prices.iloc[0]
                            for k in [("bid","Open"),("ask","Open"),("last","Open")]:
                                if k in prices.columns:
                                    print(f"sample {k}: {row0[k]}")
                    except Exception:
                        pass
                except Exception:
                    pass

    # Market snapshot can reveal permissions or status
    try:
        market = adapter.fetch_market(epic)
        snapshot = market.get("snapshot") if isinstance(market, dict) else None
        status = None
        if isinstance(snapshot, dict):
            status = snapshot.get("marketStatus")
        if status is None and isinstance(market, dict):
            status = market.get("marketStatus")
        print(f"Market status: {status}")
    except Exception as e:
        print(f"Warning: could not fetch market snapshot: {e}")

    # Market search can help identify the correct epic
    try:
        term = inst.ticker
        # If the ticker is not search-friendly, try a token from the epic, for example NASDAQ or GOLD
        epic_parts = str(epic).split(".")
        if len(epic_parts) >= 3:
            term2 = epic_parts[2]
        else:
            term2 = None

        results = adapter.search_markets(term)
        if isinstance(results, dict) and not results.get("markets") and term2:
            results = adapter.search_markets(term2)

        markets = results.get("markets") if isinstance(results, dict) else None
        if isinstance(markets, list) and markets:
            print("Top market search results:")
            for m in markets[:5]:
                if not isinstance(m, dict):
                    continue
                epic = m.get("epic")
                name = m.get("instrumentName") or m.get("instrumentType")
                print(f"  - {epic} :: {name}")
    except Exception as e:
        print(f"Warning: could not search markets: {e}")

    # Provide actionable guidance
    print("No usable OHLC rows were parsed from IG history.")
    print("Common reasons: the epic is not valid on this account, this account has no access to the market, or the response shape differs from what the converter expects.")
    print("Try: --debug, try a different resolution (for example --resolution 1H), confirm the epic in instrument_db, or pick a working epic from the search results above.")

    # Non-zero exit to indicate failure
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
