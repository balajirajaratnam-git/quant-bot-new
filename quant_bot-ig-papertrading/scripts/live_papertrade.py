from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script:
#   python scripts/live_papertrade.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ig_quant_bot.live.live_runner import LiveQuantDesk


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a single live paper-trading cycle against IG")
    ap.add_argument("--config", default="config.yaml", help="Config path")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Offline smoke test: validates config loading and logging without connecting to IG",
    )
    args = ap.parse_args()

    if args.smoke:
        run_id = LiveQuantDesk(config_path=args.config, connect=False).run_smoke()
    else:
        run_id = LiveQuantDesk(config_path=args.config).run_once()
    print(f"Run complete: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
