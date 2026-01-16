from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    """
    V7.2 Institutional Factor Engine

    Guarantees:
      - Preserves the master calendar index (no dropna that chops the calendar)
      - Exposes calendar-safety seam via:
          - is_real_bar (provided upstream by main.py after master reindex + ffill)
          - valid_signal (only True when indicators are warmed up and bar is real)
      - Lookahead safe: strategy should always read prev = f.iloc[loc-1]

    Inputs required in df:
      - OHLC columns: Open, High, Low, Close
      - is_real_bar: bool (True for actual downloaded bars, False for forward-filled calendar holes)
    """

    @staticmethod
    def compute(df: pd.DataFrame, rsi_p: int = 14, sma_p: int = 200) -> pd.DataFrame:
        f = df.copy()

        # ---------
        # Guards
        # ---------
        required = {"Open", "High", "Low", "Close", "is_real_bar"}
        missing = required.difference(set(f.columns))
        if missing:
            raise ValueError(f"FeatureEngine.compute missing required columns: {sorted(missing)}")

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

        # ----------------
        # 2) SMA + slope (trend conviction)
        # ----------------
        f["SMA"] = f["Close"].rolling(int(sma_p), min_periods=int(sma_p)).mean()
        f["SMA_Slope"] = f["SMA"].diff(5)

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

        f["Regime"] = "NEUTRAL"
        bull_mask = (
            (f["Close"] > f["SMA"]) & (f["SMA_Slope"] > 0) & (f["ATR_pct"] <= f["Vol_Median"])
        )
        bear_mask = (f["Close"] < f["SMA"]) & (f["ATR_pct"] > f["Vol_Median"])
        f.loc[bull_mask, "Regime"] = "BULL_STABLE"
        f.loc[bear_mask, "Regime"] = "BEAR_TREND"

        # ----------------
        # 5) valid_signal seam
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
