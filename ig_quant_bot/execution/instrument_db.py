from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Instrument:
    """Metadata for simulated markets.

    Notes:
      - By default this project uses yfinance daily OHLC for tickers like SPY/QQQ/GLD.
      - Those prices are ETFs, not IG index epics. Treat this as an ETF-proxy research setup unless
        you replace the price source with true index/DFB data."""

    ticker: str
    epic: str
    value_per_point: float
    margin_factor: float
    spread_points: float
    group: str = "GENERIC"


class InstrumentCatalog:
    """Static instrument catalog (tickers -> IG contract math)."""

    def __init__(self):
        self._db: Dict[str, Instrument] = {
            "QQQ": Instrument(
                ticker="QQQ",
                epic="IX.D.NASDAQ.IFD.IP",
                value_per_point=1.0,
                margin_factor=0.20,
                spread_points=1.0,
                group="US_EQ",
            ),
            "SPY": Instrument(
                ticker="SPY",
                epic="IX.D.SPTRD.IFD.IP",
                value_per_point=1.0,
                margin_factor=0.20,
                spread_points=0.8,
                group="US_EQ",
            ),
            "GLD": Instrument(
                ticker="GLD",
                epic="IX.D.GOLD.IFD.IP",
                value_per_point=10.0,
                margin_factor=0.05,
                spread_points=0.5,
                group="COMMOD",
            ),
        }

    def get(self, ticker: str) -> Instrument:
        if ticker not in self._db:
            raise KeyError(f"Instrument '{ticker}' not found in the catalog.")
        return self._db[ticker]

    def list_all(self) -> Dict[str, Instrument]:
        return dict(self._db)
