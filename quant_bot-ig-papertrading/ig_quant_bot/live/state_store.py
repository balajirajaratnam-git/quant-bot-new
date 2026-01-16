from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


DEFAULT_STATE = {
    "version": 1,
    "peak_equity": 0.0,
    "last_run_utc": None,
    "order_keys": [],
    "positions": {},
}


def load_state(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return dict(DEFAULT_STATE)

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_STATE)
        out = dict(DEFAULT_STATE)
        out.update(obj)
        # Normalise types
        if "order_keys" not in out or not isinstance(out["order_keys"], list):
            out["order_keys"] = []
        if "positions" not in out or not isinstance(out["positions"], dict):
            out["positions"] = {}
        out["peak_equity"] = float(out.get("peak_equity", 0.0) or 0.0)
        return out
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = dict(state)
    state["last_run_utc"] = pd.Timestamp.utcnow().isoformat()

    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
