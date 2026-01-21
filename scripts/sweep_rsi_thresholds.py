from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import copy
import itertools
from typing import Dict, Any, List

import yaml

from ig_quant_bot.main import QuantDesk


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_one(cfg_path: str, start: str, end: str | None) -> Dict[str, Any]:
    q = QuantDesk(config_path=cfg_path)
    q.run(start=start, end=end)
    # QuantDesk prints and persists artifacts. We parse summary from analyser if available.
    sheet = getattr(q, "last_tear_sheet", None)
    if sheet is None:
        return {}
    return {
        "final_equity": float(getattr(sheet, "final_equity", 0.0)),
        "cagr": float(getattr(sheet, "cagr", 0.0)),
        "sharpe": float(getattr(sheet, "sharpe", 0.0)),
        "maxdd": float(getattr(sheet, "max_drawdown", 0.0)),
        "trades": int(getattr(sheet, "total_trades", 0)),
        "win_rate": float(getattr(sheet, "win_rate", 0.0)),
        "pf": float(getattr(sheet, "profit_factor", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Base config yaml (e.g., config_A1.yaml)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--rsi-entries", default="30,35,40", help="Comma list")
    ap.add_argument("--regimes", default="BULL_STABLE,ANY", help="Comma list; ANY disables regime filter")
    args = ap.parse_args()

    base_cfg_path = Path(args.base_config)
    base_cfg = load_yaml(str(base_cfg_path))

    rsi_entries = [int(x.strip()) for x in args.rsi_entries.split(",") if x.strip()]
    regimes = [x.strip().upper() for x in args.regimes.split(",") if x.strip()]

    tmp_dir = base_cfg_path.parent / "vault" / "tmp_sweeps"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for rsi_e, reg in itertools.product(rsi_entries, regimes):
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("strategy", {})
        cfg["strategy"]["rsi_entry_threshold"] = rsi_e
        if reg == "ANY":
            # Do not remove the key: QuantDesk defaults regime_filter to BULL_STABLE.
            # We want an explicit opt-out.
            cfg["strategy"]["regime_filter"] = "ANY"
        else:
            cfg["strategy"]["regime_filter"] = reg

        tmp_cfg_path = tmp_dir / f"sweep_rsi{rsi_e}_reg{reg}.yaml"
        write_yaml(cfg, str(tmp_cfg_path))

        print(f"\n=== Running rsi_entry={rsi_e} regime={reg} ===")
        out = run_one(str(tmp_cfg_path), start=args.start, end=args.end)
        out.update({"rsi_entry": rsi_e, "regime": reg})
        results.append(out)

    print("\n\n=== Sweep summary (sorted by PF then CAGR) ===")
    def _score(d: Dict[str, Any]) -> tuple[float, float]:
        pf = d.get("pf")
        cagr = d.get("cagr")
        try:
            pf_v = float(pf)
        except Exception:
            pf_v = 0.0
        try:
            cagr_v = float(cagr)
        except Exception:
            cagr_v = 0.0
        if pf_v == float("inf"):
            pf_v = 999.0
        return (pf_v, cagr_v)

    results_sorted = sorted(results, key=_score, reverse=True)
    for r in results_sorted:
        pf = r.get("pf", 0)
        try:
            pf_f = float(pf)
        except Exception:
            pf_f = 0.0
        pf_s = "inf" if pf_f == float("inf") else f"{pf_f:.2f}"
        win = r.get("win_rate")
        win_s = f"{float(win):.2f}" if win is not None else "n/a"
        print(
            "rsi={rsi} reg={reg} trades={tr} win={win} PF={pf} CAGR={cagr:.4f} MaxDD={mdd:.4f} Sharpe={sh:.2f} FinalEq={eq:.2f}".format(
                rsi=r.get("rsi_entry"),
                reg=r.get("regime"),
                tr=int(r.get("trades") or 0),
                win=win_s,
                pf=pf_s,
                cagr=float(r.get("cagr") or 0.0),
                mdd=float(r.get("maxdd") or 0.0),
                sh=float(r.get("sharpe") or 0.0),
                eq=float(r.get("final_equity") or 0.0),
            )
        )

    print("\nTip: Pick a setting with >= 30 trades, PF > 1.1 and MaxDD you can stomach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())