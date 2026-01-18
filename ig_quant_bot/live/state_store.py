from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


DEFAULT_STATE = {
    "version": 2,  # Bumped for trailing stop support
    "peak_equity": 0.0,
    "last_run_utc": None,
    "order_keys": [],
    "positions": {},
}


def load_state(path: Path) -> Dict[str, Any]:
    """Load persisted bot state from JSON file.
    
    The state includes:
      - peak_equity: Highest equity value seen (for drawdown calculation)
      - order_keys: Idempotency keys to prevent duplicate orders
      - positions: Per-deal metadata including trailing stop state
    
    Position metadata schema (stored under deal_id key):
      - ticker: str
      - epic: str
      - direction: str (BUY/SELL)
      - size: float
      - entry_time: ISO datetime string
      - entry_price: float
      - stop_level: float or None (current stop level, may trail)
      - limit_level: float or None
      - max_hold_days: int or None
      - highest_price_since_entry: float (for trailing stop calculation)
      - trailing_stop_activated: bool
    """
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
        
        # Migrate v1 positions to v2 schema (add trailing stop fields)
        for deal_id, pos_meta in out["positions"].items():
            if isinstance(pos_meta, dict):
                if "highest_price_since_entry" not in pos_meta:
                    pos_meta["highest_price_since_entry"] = pos_meta.get("entry_price", 0.0)
                if "trailing_stop_activated" not in pos_meta:
                    pos_meta["trailing_stop_activated"] = False
        
        return out
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Persist bot state to JSON file.
    
    This is called after every run (even dry runs save order keys for idempotency).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = dict(state)
    state["last_run_utc"] = pd.Timestamp.utcnow().isoformat()

    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def update_trailing_stop(
    pos_meta: Dict[str, Any],
    current_price: float,
    atr_pts: float,
    activation_atr_mult: float,
    trail_atr_mult: float,
) -> Dict[str, Any]:
    """Update trailing stop state for a position.
    
    This implements a trailing stop that:
      1. Activates when price moves activation_atr_mult ATRs above entry
      2. Once activated, trails trail_atr_mult ATRs below the highest price
      3. Never moves down (only ratchets up)
    
    Args:
        pos_meta: Position metadata dict from state
        current_price: Current mid price
        atr_pts: Current ATR in price points
        activation_atr_mult: ATR multiple to activate trailing (e.g., 1.0)
        trail_atr_mult: ATR multiple to trail behind high (e.g., 1.2)
    
    Returns:
        Updated pos_meta dict (mutates in place and returns)
    """
    entry_price = float(pos_meta.get("entry_price", 0.0))
    if entry_price <= 0 or atr_pts <= 0:
        return pos_meta
    
    # Track highest price since entry
    highest = float(pos_meta.get("highest_price_since_entry", entry_price))
    if current_price > highest:
        highest = current_price
        pos_meta["highest_price_since_entry"] = highest
    
    # Check if trailing should activate
    activation_level = entry_price + (activation_atr_mult * atr_pts)
    was_activated = bool(pos_meta.get("trailing_stop_activated", False))
    
    if highest >= activation_level:
        pos_meta["trailing_stop_activated"] = True
        
        # Calculate trailing stop level
        trail_stop = highest - (trail_atr_mult * atr_pts)
        
        # Only update if it's higher than current stop (never move stop down)
        current_stop = pos_meta.get("stop_level")
        if current_stop is None or trail_stop > float(current_stop):
            pos_meta["stop_level"] = float(trail_stop)
            pos_meta["stop_updated_reason"] = "TRAILING"
    
    return pos_meta
