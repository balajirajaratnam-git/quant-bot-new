from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class DecisionSummary:
    total_rows: int
    total_orders: int
    total_skips: int
    top_skip_reasons: pd.DataFrame
    orders_by_reason: pd.DataFrame


def load_decision_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def summarise_decisions(df: pd.DataFrame, *, top_n: int = 15) -> DecisionSummary:
    if df is None or df.empty:
        empty = pd.DataFrame()
        return DecisionSummary(0, 0, 0, empty, empty)

    total_rows = int(len(df))
    orders = df[df["action"].astype(str).str.upper() == "ORDER"].copy()
    skips = df[df["action"].astype(str).str.upper() == "SKIP"].copy()

    total_orders = int(len(orders))
    total_skips = int(len(skips))

    top_skip = (
        skips.groupby(["phase", "reason_code"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    orders_by_reason = (
        orders.groupby(["phase", "side", "reason_code"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return DecisionSummary(
        total_rows=total_rows,
        total_orders=total_orders,
        total_skips=total_skips,
        top_skip_reasons=top_skip,
        orders_by_reason=orders_by_reason,
    )
