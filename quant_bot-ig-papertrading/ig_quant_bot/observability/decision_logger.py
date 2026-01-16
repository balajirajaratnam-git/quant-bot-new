from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DecisionRecord:
    ts_utc: str
    run_id: str
    date: str
    ticker: str
    phase: str  # ENTRY | EXIT | RISK | SYSTEM
    action: str  # SKIP | ORDER | INFO
    side: str = ""
    qty: float = 0.0
    price_mid: float = 0.0
    stop_level: float = 0.0
    limit_level: float = 0.0
    regime: str = ""
    rsi: float = 0.0
    atr_pts: float = 0.0
    equity: float = 0.0
    free_margin: float = 0.0
    margin_used: float = 0.0
    reason_code: str = ""
    idempotency_key: str = ""
    details_json: str = "{}"


class DecisionLogger:
    """Append-only CSV decision log.

    This is designed to answer:
      - Why did we not trade?
      - What constraint bound sizing?
      - Which exit fired?

    The logger is safe to leave enabled during live paper trading.
    """

    def __init__(self, file_path: Path, *, run_id: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)

        self._fh = open(self.file_path, mode="a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(DecisionRecord.__annotations__.keys()))

        # Header only if file is new/empty
        if self.file_path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def log(
        self,
        *,
        date: pd.Timestamp,
        ticker: str,
        phase: str,
        action: str,
        reason_code: str = "",
        side: str = "",
        qty: float = 0.0,
        price_mid: float = 0.0,
        stop_level: Optional[float] = None,
        limit_level: Optional[float] = None,
        regime: str = "",
        rsi: Optional[float] = None,
        atr_pts: Optional[float] = None,
        equity: Optional[float] = None,
        free_margin: Optional[float] = None,
        margin_used: Optional[float] = None,
        idempotency_key: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts_utc = pd.Timestamp.utcnow().isoformat()
        rec = DecisionRecord(
            ts_utc=ts_utc,
            run_id=self.run_id,
            date=pd.to_datetime(date).date().isoformat(),
            ticker=str(ticker),
            phase=str(phase),
            action=str(action),
            side=str(side),
            qty=float(qty or 0.0),
            price_mid=float(price_mid or 0.0),
            stop_level=float(stop_level) if stop_level is not None else 0.0,
            limit_level=float(limit_level) if limit_level is not None else 0.0,
            regime=str(regime or ""),
            rsi=float(rsi) if rsi is not None else 0.0,
            atr_pts=float(atr_pts) if atr_pts is not None else 0.0,
            equity=float(equity) if equity is not None else 0.0,
            free_margin=float(free_margin) if free_margin is not None else 0.0,
            margin_used=float(margin_used) if margin_used is not None else 0.0,
            reason_code=str(reason_code or ""),
            idempotency_key=str(idempotency_key or ""),
            details_json=json.dumps(details or {}, ensure_ascii=False, default=str),
        )
        self._writer.writerow(asdict(rec))
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
