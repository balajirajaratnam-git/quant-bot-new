from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ig_quant_bot.analytics.decision_report import load_decision_log, summarise_decisions


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise a decision_log.csv")
    ap.add_argument("path", help="Path to decision_log.csv")
    ap.add_argument("--top", type=int, default=15, help="Top N skip reasons")
    args = ap.parse_args()

    df = load_decision_log(args.path)
    summary = summarise_decisions(df, top_n=args.top)

    print(f"Rows: {summary.total_rows} | Orders: {summary.total_orders} | Skips: {summary.total_skips}")

    print("\nTop skip reasons:")
    if summary.top_skip_reasons.empty:
        print("  (none)")
    else:
        print(summary.top_skip_reasons.to_string(index=False))

    print("\nOrders by reason:")
    if summary.orders_by_reason.empty:
        print("  (none)")
    else:
        print(summary.orders_by_reason.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
