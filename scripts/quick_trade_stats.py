"""Quick trade statistics for the latest run.

Works even when there are 0 trades. It looks for CSV artifacts first then parquet.

Usage:
  python scripts/quick_trade_stats.py
  python scripts/quick_trade_stats.py --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def _pick_run_dir(base: str, run_id: str | None) -> str:
    if run_id:
        path = os.path.join(base, run_id)
        if not os.path.isdir(path):
            raise SystemExit(f"Run dir not found: {path}")
        return path

    run_dirs = sorted(glob.glob(os.path.join(base, "*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        raise SystemExit(f"No runs found under: {base}")
    return run_dirs[-1]


def _read_trades(run_dir: str) -> pd.DataFrame:
    p_csv = os.path.join(run_dir, "trades.csv")
    p_parq = os.path.join(run_dir, "trades.parquet")

    if os.path.isfile(p_csv):
        return pd.read_csv(p_csv)
    if os.path.isfile(p_parq):
        return pd.read_parquet(p_parq)

    # Nothing present. Return empty.
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--base", default=os.path.join("vault", "runs"))
    args = ap.parse_args()

    run_dir = _pick_run_dir(args.base, args.run_id)
    df = _read_trades(run_dir)

    print(f"Latest run: {run_dir}")

    if df.empty:
        print("No trades found.")
        return 0

    pnl_col = "total_net_pnl" if "total_net_pnl" in df.columns else ("pnl" if "pnl" in df.columns else None)
    if pnl_col is None:
        print("Trades file present but no PnL column found. Columns:")
        print(list(df.columns))
        return 0

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    w = pnl[pnl > 0]
    l = pnl[pnl < 0]

    print(f"Trades: {len(df)}")
    print(pnl.describe())
    print(
        f"wins {len(w)} losses {len(l)} avg_win {w.mean() if len(w) else 0:.2f} avg_loss {l.mean() if len(l) else 0:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
