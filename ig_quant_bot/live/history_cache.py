from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class HistoryCacheHit:
    """Represents a cache hit for historical OHLC data."""

    df: pd.DataFrame
    source: str
    path: str


def _safe_key(text: str) -> str:
    s = (text or "").strip()
    s = s.replace(" ", "_")
    s = s.replace(".", "_")
    s = s.replace(":", "_")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = _SAFE_RE.sub("_", s)
    return s.strip("_") or "key"


def cache_path(cache_dir: str, *, epic: str, resolution: str) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = f"{_safe_key(epic)}__{_safe_key(resolution)}"
    return root / f"{key}.pkl"


def load_cache(
    cache_dir: str,
    *,
    epic: str,
    resolution: str,
    max_age_hours: float,
) -> Optional[HistoryCacheHit]:
    """Load cached OHLC.

    The cache payload is a dict:
      {"source": "ig"|"yf", "df": <DataFrame>}
    """
    path = cache_path(cache_dir, epic=epic, resolution=resolution)
    if not path.exists():
        return None

    try:
        age_hours = (pd.Timestamp.utcnow() - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")).total_seconds() / 3600.0
        if age_hours > float(max_age_hours):
            return None
    except Exception:
        # If age check fails, still attempt to load.
        pass

    try:
        payload = pd.read_pickle(path)
        if isinstance(payload, dict) and isinstance(payload.get("df"), pd.DataFrame):
            src = str(payload.get("source") or "unknown")
            df = payload["df"].copy()
            return HistoryCacheHit(df=df, source=src, path=str(path))
        if isinstance(payload, pd.DataFrame):
            return HistoryCacheHit(df=payload.copy(), source="unknown", path=str(path))
    except Exception:
        return None
    return None


def save_cache(cache_dir: str, *, epic: str, resolution: str, df: pd.DataFrame, source: str) -> str:
    path = cache_path(cache_dir, epic=epic, resolution=resolution)
    payload: Dict[str, Any] = {"source": str(source), "df": df.copy()}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, path)
    return str(path)
