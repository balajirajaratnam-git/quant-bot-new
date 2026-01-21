"""
Professional Exit Logic Module (V2)

This module provides improved exit decision functions that can be called from
the live runner. It separates exit logic from the main orchestration for
clarity and testability.

Exit Types Supported:
  1. RSI Mean-Reversion Exit: RSI recovers to threshold
  2. Time Stop: Position held too long
  3. Trailing Stop Hit: Price falls below trailing stop
  4. Adverse Exit: Failed reversal detection (underwater after RSI recovers)
  5. Take Profit: Price hits limit level (handled by broker)
  6. Stop Loss: Price hits stop level (handled by broker)

The broker handles #5 and #6 via attached orders. This module handles #1-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class ExitSignal:
    """Result of exit evaluation for a position."""
    should_exit: bool
    reason_code: str
    details: Dict[str, Any]


def evaluate_exit(
    ticker: str,
    pos: Dict[str, Any],
    pos_meta: Dict[str, Any],
    row: pd.Series,
    current_price: float,
    *,
    rsi_exit_threshold: float = 45.0,
    max_hold_days: int = 10,
    adverse_exit_enabled: bool = True,
    adverse_exit_min_days: int = 3,
    adverse_exit_rsi_threshold: float = 40.0,
) -> ExitSignal:
    """Evaluate whether a position should be exited.
    
    This function checks multiple exit conditions in priority order:
      1. Trailing stop hit (highest priority - protective)
      2. RSI mean-reversion complete
      3. Time stop exceeded
      4. Adverse exit (failed reversal)
    
    Args:
        ticker: Instrument ticker
        pos: Position dict from PortfolioState (qty, avg_price, etc.)
        pos_meta: Position metadata from live_state (entry_time, stops, etc.)
        row: Feature row (RSI, Regime, ATR_pts, etc.)
        current_price: Current mid price from broker
        rsi_exit_threshold: RSI level to exit on mean-reversion
        max_hold_days: Maximum days to hold before time stop
        adverse_exit_enabled: Whether to check for failed reversals
        adverse_exit_min_days: Minimum days before adverse exit can trigger
        adverse_exit_rsi_threshold: RSI level that indicates recovery (for adverse check)
    
    Returns:
        ExitSignal with should_exit flag, reason code, and details
    """
    details: Dict[str, Any] = {
        "ticker": ticker,
        "current_price": current_price,
    }
    
    rsi = float(row.get("RSI", 50.0))
    details["rsi"] = rsi
    
    entry_price = float(pos_meta.get("entry_price", 0.0) or pos.get("avg_price", 0.0))
    details["entry_price"] = entry_price
    
    # Calculate position P&L
    qty = float(pos.get("qty", 0.0))
    pnl_points = (current_price - entry_price) if qty > 0 else 0.0
    pnl_pct = (pnl_points / entry_price * 100) if entry_price > 0 else 0.0
    details["pnl_pct"] = pnl_pct
    details["pnl_points"] = pnl_points
    
    # 1. Check trailing stop (if activated)
    trailing_activated = bool(pos_meta.get("trailing_stop_activated", False))
    stop_level = pos_meta.get("stop_level") or pos.get("stop_price")
    
    if trailing_activated and stop_level is not None:
        if current_price <= float(stop_level):
            details["stop_level"] = stop_level
            details["trailing_activated"] = True
            return ExitSignal(
                should_exit=True,
                reason_code="TRAILING_STOP_HIT",
                details=details,
            )
    
    # 2. Check RSI mean-reversion exit
    if rsi >= rsi_exit_threshold:
        details["rsi_threshold"] = rsi_exit_threshold
        return ExitSignal(
            should_exit=True,
            reason_code="RSI_MEAN_REVERSION",
            details=details,
        )
    
    # 3. Check time stop
    entry_time = pd.to_datetime(pos_meta.get("entry_time"), errors="coerce")
    if max_hold_days > 0 and pd.notna(entry_time):
        held_days = (pd.Timestamp.utcnow().normalize() - entry_time.normalize()).days
        details["held_days"] = held_days
        details["max_hold_days"] = max_hold_days
        
        if held_days >= max_hold_days:
            return ExitSignal(
                should_exit=True,
                reason_code="TIME_STOP",
                details=details,
            )
    else:
        held_days = 0
        details["held_days"] = held_days
    
    # 4. Check adverse exit (failed reversal)
    # This triggers when: we've held for min_days, RSI has recovered somewhat,
    # but price is still underwater. This indicates the reversal failed.
    if adverse_exit_enabled and held_days >= adverse_exit_min_days:
        is_underwater = pnl_pct < 0
        rsi_recovered = rsi >= adverse_exit_rsi_threshold
        
        if is_underwater and rsi_recovered:
            details["adverse_rsi_threshold"] = adverse_exit_rsi_threshold
            details["adverse_min_days"] = adverse_exit_min_days
            return ExitSignal(
                should_exit=True,
                reason_code="ADVERSE_EXIT_FAILED_REVERSAL",
                details=details,
            )
    
    # No exit signal
    return ExitSignal(
        should_exit=False,
        reason_code="NO_EXIT_SIGNAL",
        details=details,
    )


def evaluate_entry_quality(
    row: pd.Series,
    *,
    rsi_entry_threshold: float = 28.0,
    min_signal_strength: float = 50.0,
    regime_filter: str = "BULL_STABLE",
    require_trend_up: bool = False,
    require_rsi_turn: bool = False,
) -> Tuple[bool, float, str, Dict[str, Any]]:
    """Evaluate whether an entry is permitted on the given bar.

    Returns: (allowed, quality_score, reason_code, details)
    """
    details: Dict[str, Any] = {}

    rsi = float(row.get("RSI", np.nan))
    regime = str(row.get("Regime", ""))
    strength = float(row.get("Signal_Strength", 0.0))

    details["rsi"] = rsi
    details["regime"] = regime
    details["signal_strength"] = strength

    # Regime filter: keep behavior aligned with backtests
    rf = str(regime_filter or "").upper()
    if rf not in ("", "ANY"):
        if rf == "NOT_BEAR":
            if regime.upper() == "BEAR_TREND":
                return (False, 0.0, "REGIME_BEAR_BLOCKED", details)
        elif rf in ("BULL", "TREND_UP"):
            try:
                if not (float(row.get("Close")) > float(row.get("SMA")) and float(row.get("SMA_Slope")) > 0):
                    return (False, 0.0, "TREND_FILTER_FAIL", details)
            except Exception:
                return (False, 0.0, "TREND_FILTER_FAIL", details)
        else:
            if regime != regime_filter:
                return (False, 0.0, "REGIME_MISMATCH", details)

    # Optional: require trend up even if regime filter is loose
    if require_trend_up:
        try:
            if not (float(row.get("Close")) > float(row.get("SMA")) and float(row.get("SMA_Slope")) > 0):
                return (False, 0.0, "TREND_FILTER_FAIL", details)
        except Exception:
            return (False, 0.0, "TREND_FILTER_FAIL", details)

    # Optional: require RSI to be turning up
    if require_rsi_turn:
        try:
            if float(row.get("RSI_Slope", 0.0)) <= 0:
                return (False, 0.0, "RSI_NOT_TURNING_UP", details)
        except Exception:
            return (False, 0.0, "RSI_NOT_TURNING_UP", details)

    # Check RSI threshold
    if not np.isfinite(rsi):
        return (False, 0.0, "RSI_NAN", details)
    if rsi >= float(rsi_entry_threshold):
        return (False, 0.0, "RSI_NOT_LOW_ENOUGH", details)

    # Check signal strength
    if strength < float(min_signal_strength):
        return (False, 0.0, "SIGNAL_STRENGTH_TOO_LOW", details)

    # Quality score: lower RSI and higher strength is better
    quality = float((rsi_entry_threshold - rsi)) + 0.01 * strength
    details["quality"] = quality
    return (True, quality, "OK", details)
