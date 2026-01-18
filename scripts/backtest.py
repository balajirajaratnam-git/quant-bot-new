from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ig_quant_bot.main import QuantDesk


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the IG-style research backtest")
    ap.add_argument("--config", default="config.yaml", help="Config path")
    ap.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD) or omit for latest")
    args = ap.parse_args()

    QuantDesk(config_path=args.config).run(start=args.start, end=args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
