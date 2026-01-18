from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    """
    V8.0 Institutional Factor Engine (Professional Upgrade)

    Changes from V7.2:
      - is_real_bar is now auto-generated if not provided (fixes live runner bug)
      - Added RSI_Slope for momentum confirmation
      - Added Trend_Strength for graduated regime scoring (not just binary)
      - Added Distance_From_SMA for mean-reversion magnitude
      - Added signal_strength composite score for entry ranking

    Guarantees:
      - Preserves the master calendar index (no dropna that chops the calendar)
      - Exposes calendar-safety seam via:
          - is_real_bar (provided upstream or auto-generated)
          - valid_signal (only True when indicators are warmed up and bar is real)
      - Lookahead safe: strategy should always read prev = f.iloc[loc-1]

    Inputs required in df:
      - OHLC columns: Open, High, Low, Close
      - is_real_bar: bool (optional - will be set to True for all rows if not provided)
    """

    @staticmethod
    def compute(df: pd.DataFrame, rsi_p: int = 14, sma_p: int = 200) -> pd.DataFrame:
        f = df.copy()

        # ---------
        # Guards (relaxed: is_real_bar is now optional)
        # ---------
        required = {"Open", "High", "Low", "Close"}
        missing = required.difference(set(f.columns))
        if missing:
            raise ValueError(f"FeatureEngine.compute missing required columns: {sorted(missing)}")

        # Auto-generate is_real_bar if not present (fixes live runner bug)
        if "is_real_bar" not in f.columns:
            f["is_real_bar"] = True

        # Ensure index is datetime-like and sorted
        f.index = pd.to_datetime(f.index)
        f = f.sort_index()

        # Coerce is_real_bar to boolean (important after reindex/ffill)
        f["is_real_bar"] = f["is_real_bar"].astype(bool)

        # ----------------
        # 1) RSI (Wilder-style via EMA approximation)
        # ----------------
        delta = f["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1 / float(rsi_p), min_periods=int(rsi_p), adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / float(rsi_p), min_periods=int(rsi_p), adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        f["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        # NEW: RSI momentum (slope over 3 bars) - positive means RSI is recovering
        f["RSI_Slope"] = f["RSI"].diff(3)

        # ----------------
        # 2) SMA + slope (trend conviction)
        # ----------------
        f["SMA"] = f["Close"].rolling(int(sma_p), min_periods=int(sma_p)).mean()
        f["SMA_Slope"] = f["SMA"].diff(5)

        # NEW: Distance from SMA as percentage (for mean-reversion magnitude)
        f["Distance_From_SMA_Pct"] = (f["Close"] - f["SMA"]) / f["SMA"].replace(0.0, np.nan) * 100

        # ----------------
        # 3) ATR in points and pct
        # ----------------
        tr1 = f["High"] - f["Low"]
        tr2 = (f["High"] - f["Close"].shift(1)).abs()
        tr3 = (f["Low"] - f["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        f["ATR_pts"] = tr.rolling(14, min_periods=14).mean()
        f["ATR_pct"] = f["ATR_pts"] / f["Close"].replace(0.0, np.nan)

        # ----------------
        # 4) Vol threshold + regimes (calendar-safe rolling median)
        # ----------------
        f["Vol_Median"] = f["ATR_pct"].rolling(252, min_periods=252).median()

        # NEW: Trend strength as continuous score (0-100) instead of just binary regime
        # Combines: price vs SMA, SMA slope direction, volatility environment
        trend_score = pd.Series(50.0, index=f.index)  # Neutral baseline

        # Price position relative to SMA: +/- 25 points
        above_sma = f["Close"] > f["SMA"]
        trend_score = trend_score + np.where(above_sma, 25, -25)

        # SMA slope direction: +/- 15 points
        slope_positive = f["SMA_Slope"] > 0
        trend_score = trend_score + np.where(slope_positive, 15, -15)

        # Volatility environment: +/- 10 points (low vol is bullish for mean-reversion)
        low_vol = f["ATR_pct"] <= f["Vol_Median"]
        trend_score = trend_score + np.where(low_vol, 10, -10)

        f["Trend_Strength"] = trend_score.clip(0, 100)

        # Original binary regime (kept for backward compatibility)
        f["Regime"] = "NEUTRAL"
        bull_mask = (
            (f["Close"] > f["SMA"]) & (f["SMA_Slope"] > 0) & (f["ATR_pct"] <= f["Vol_Median"])
        )
        bear_mask = (f["Close"] < f["SMA"]) & (f["ATR_pct"] > f["Vol_Median"])
        f.loc[bull_mask, "Regime"] = "BULL_STABLE"
        f.loc[bear_mask, "Regime"] = "BEAR_TREND"

        # ----------------
        # 5) NEW: Signal strength composite for entry ranking
        # ----------------
        # Combines: how oversold (RSI), trend quality, RSI recovering
        # Higher is better for long entries
        f["Signal_Strength"] = 0.0

        # RSI oversold contribution: lower RSI = higher score (max 40 points at RSI=10)
        rsi_component = ((50 - f["RSI"].clip(10, 50)) / 40) * 40
        f["Signal_Strength"] = f["Signal_Strength"] + rsi_component.fillna(0)

        # Trend quality contribution: scaled from Trend_Strength (max 30 points)
        trend_component = (f["Trend_Strength"] / 100) * 30
        f["Signal_Strength"] = f["Signal_Strength"] + trend_component.fillna(0)

        # RSI momentum contribution: recovering RSI is better (max 20 points)
        # Positive RSI_Slope means RSI is turning up from oversold
        rsi_mom_component = f["RSI_Slope"].clip(-10, 10) + 10  # Range 0-20
        f["Signal_Strength"] = f["Signal_Strength"] + rsi_mom_component.fillna(0)

        # Near support (close to SMA) contribution: max 10 points
        # Being within 3% of SMA is good for mean-reversion
        dist_component = (3 - f["Distance_From_SMA_Pct"].abs().clip(0, 3)) / 3 * 10
        f["Signal_Strength"] = f["Signal_Strength"] + dist_component.fillna(0)

        # ----------------
        # 6) valid_signal seam
        # ----------------
        f["valid_signal"] = (
            f["is_real_bar"]
            & f["RSI"].notna()
            & f["SMA"].notna()
            & f["ATR_pts"].notna()
            & f["ATR_pct"].notna()
            & f["Vol_Median"].notna()
        )

        return f
