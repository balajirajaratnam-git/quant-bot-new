"""Entry-gate diagnostics for strategy configs.

This script answers:
- Why did my backtest produce 0 trades?
- Which specific gate (regime, RSI, trend filter, RSI turn, signal strength) is blocking entries?
- Roughly how many entry opportunities exist *before* considering position/margin constraints?

It mirrors the lookahead-safe approach used in QuantDesk.run():
- A decision on date D is based on the PREVIOUS bar (D-1).
- We only consider dates where the current bar is a "real" bar (is_real_bar=True).
- We never enter on the last date in the backtest window.

Usage examples:
  python scripts/diagnose_entry_gates.py --config config_A2_edge_E1.yaml --start 2010-01-01
  python scripts/diagnose_entry_gates.py --config config_A2_edge_E1.yaml --start 2010-01-01 --ticker QQQ
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

from ig_quant_bot.main import QuantDesk


def _pct(x: int, denom: int) -> float:
    return 0.0 if denom == 0 else (100.0 * float(x) / float(denom))


def _regime_match(regime: str, filter_name: str) -> bool:
    rf = (filter_name or "ANY").upper()
    r = (regime or "").upper()
    return rf == "ANY" or r == rf


def diagnose_one(
    f: pd.DataFrame,
    master_idx: pd.DatetimeIndex,
    regime_filter: str,
    rsi_entry: float,
    require_trend_up: bool,
    require_rsi_turn: bool,
    min_signal_strength: float,
) -> dict:
    """Return counts for each gate and some helpful distributions."""

    # Align to master calendar and ensure tz-naive comparison safety.
    f = f.reindex(master_idx)

    # Dates considered for entry checks:
    # - skip the first date (no prev bar)
    # - skip the last date (QuantDesk avoids entry on last_date)
    considered = master_idx[1:-1]

    cur_is_real = f.loc[considered, "is_real_bar"].astype(bool).fillna(False)

    prev = f.shift(1).loc[considered]

    prev_valid = prev.get("valid_signal", pd.Series(False, index=considered)).astype(bool).fillna(False)

    # Gate booleans (vectorized)
    gate_regime = prev["Regime"].astype(str).str.upper().apply(lambda r: _regime_match(r, regime_filter))
    gate_rsi = prev["RSI"].astype(float) < float(rsi_entry)

    if require_trend_up:
        gate_trend = (prev["Close"].astype(float) > prev["SMA"].astype(float)) & (prev["SMA_Slope"].astype(float) > 0.0)
    else:
        gate_trend = pd.Series(True, index=considered)

    if require_rsi_turn:
        gate_rsi_turn = prev["RSI_Slope"].astype(float) > 0.0
    else:
        gate_rsi_turn = pd.Series(True, index=considered)

    gate_strength = prev["Signal_Strength"].astype(float) >= float(min_signal_strength)

    # Apply is_real_bar on current bar, then apply prev gates.
    base = cur_is_real

    # Funnel counts
    n_total = int(base.sum())
    n_valid = int((base & prev_valid).sum())
    n_regime = int((base & prev_valid & gate_regime).sum())
    n_rsi = int((base & prev_valid & gate_regime & gate_rsi).sum())
    n_trend = int((base & prev_valid & gate_regime & gate_rsi & gate_trend).sum())
    n_turn = int((base & prev_valid & gate_regime & gate_rsi & gate_trend & gate_rsi_turn).sum())
    n_strength = int((base & prev_valid & gate_regime & gate_rsi & gate_trend & gate_rsi_turn & gate_strength).sum())

    # Helpful conditional distributions to choose sane thresholds
    # RSI distribution under trend_up (if enabled)
    if require_trend_up:
        rsi_under_trend = prev.loc[base & prev_valid & gate_trend, "RSI"].astype(float)
    else:
        rsi_under_trend = prev.loc[base & prev_valid, "RSI"].astype(float)

    rsi_q = {}
    if len(rsi_under_trend.dropna()) > 0:
        qs = [0.01, 0.05, 0.10, 0.25, 0.50]
        rsi_q = {f"q{int(q*100):02d}": float(rsi_under_trend.quantile(q)) for q in qs}

    # Show first few candidate dates (entry opportunities)
    candidate_mask = base & prev_valid & gate_regime & gate_rsi & gate_trend & gate_rsi_turn & gate_strength
    sample_dates = [d.strftime("%Y-%m-%d") for d in considered[candidate_mask][:10]]

    return {
        "total_considered_real_bars": n_total,
        "pass_prev_valid_signal": n_valid,
        "pass_regime": n_regime,
        "pass_rsi": n_rsi,
        "pass_trend_filter": n_trend,
        "pass_rsi_turn": n_turn,
        "pass_signal_strength": n_strength,
        "pct_prev_valid": _pct(n_valid, n_total),
        "pct_regime": _pct(n_regime, n_total),
        "pct_rsi": _pct(n_rsi, n_total),
        "pct_trend": _pct(n_trend, n_total),
        "pct_turn": _pct(n_turn, n_total),
        "pct_strength": _pct(n_strength, n_total),
        "rsi_quantiles_under_trend": rsi_q,
        "sample_candidate_dates": sample_dates,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--ticker", default=None, help="Optional: analyse only one ticker")
    args = ap.parse_args()

    q = QuantDesk(config_path=args.config)

    # Pull strategy params exactly as QuantDesk will use them
    strat = q.cfg.get("strategy", {}) or {}
    regime_filter = str(strat.get("regime_filter", "ANY"))
    rsi_entry = float(strat.get("rsi_entry_threshold", 30))
    require_trend_up = bool(strat.get("require_trend_up", False))
    require_rsi_turn = bool(strat.get("require_rsi_turn", False))
    min_signal_strength = float(strat.get("min_signal_strength", 0.0))

    factors, master_idx = q.load_and_sync_data(args.start, args.end)

    tickers = list(factors.keys())
    if args.ticker:
        tickers = [t for t in tickers if t.upper() == args.ticker.upper()]
        if not tickers:
            print(f"Ticker {args.ticker} not found in config universe.")
            return 2

    print("=== Entry Gate Diagnostics ===")
    print(f"Config: {args.config}")
    print(f"Start: {args.start}  End: {args.end or '(default)'}")
    print("--- Strategy gates ---")
    print(f"regime_filter={regime_filter}  rsi_entry_threshold={rsi_entry}")
    print(f"require_trend_up={require_trend_up}  require_rsi_turn={require_rsi_turn}  min_signal_strength={min_signal_strength}")
    print()

    for t in tickers:
        res = diagnose_one(
            factors[t],
            master_idx,
            regime_filter=regime_filter,
            rsi_entry=rsi_entry,
            require_trend_up=require_trend_up,
            require_rsi_turn=require_rsi_turn,
            min_signal_strength=min_signal_strength,
        )

        print(f"--- {t} ---")
        print(f"Real bars considered: {res['total_considered_real_bars']}")
        print(
            "Funnel: "
            f"prev_valid={res['pass_prev_valid_signal']} "
            f"regime={res['pass_regime']} "
            f"rsi={res['pass_rsi']} "
            f"trend={res['pass_trend_filter']} "
            f"rsi_turn={res['pass_rsi_turn']} "
            f"strength={res['pass_signal_strength']}"
        )
        print(
            "Rates (% of real bars): "
            f"prev_valid={res['pct_prev_valid']:.2f}% "
            f"regime={res['pct_regime']:.2f}% "
            f"rsi={res['pct_rsi']:.2f}% "
            f"trend={res['pct_trend']:.2f}% "
            f"rsi_turn={res['pct_turn']:.2f}% "
            f"strength={res['pct_strength']:.2f}%"
        )

        qd = res.get("rsi_quantiles_under_trend") or {}
        if qd:
            print("RSI quantiles under trend filter:")
            print("  " + "  ".join([f"{k}={v:.2f}" for k, v in qd.items()]))

        sd = res.get("sample_candidate_dates") or []
        if sd:
            print("Sample candidate dates (first 10):", ", ".join(sd))
        else:
            print("No candidate dates found with current gates (before position constraints).")

        print()

    print("Tip: If 'pass_rsi' is ~0, your RSI threshold is too low for the chosen regime/trend filter.")
    print("     Use the RSI quantiles under trend filter to pick a threshold that yields enough candidates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
