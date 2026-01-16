from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from ig_quant_bot.core.contracts import FillEvent, Order


class IBrokerAdapter(ABC):
    """Broker contract shared by backtest and live adapters."""

    @abstractmethod
    def execute(
        self,
        order: Order,
        price: float,
        regime: str,
        atr_pts: float,
        current_pos: Optional[Dict] = None,
    ) -> FillEvent:
        pass

    @abstractmethod
    def calculate_funding(self, ticker: str, pos: Dict, price: float, date: pd.Timestamp) -> FillEvent:
        pass
