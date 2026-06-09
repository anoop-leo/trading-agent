"""Domain models shared across Phase 1 modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SignalScores:
    trend_score: int
    momentum_score: int
    volume_score: int
    bottom_score: int = 0
    sr_score: int = 0
    rr_score: int = 0
    regime_score: int = 0

    @property
    def total(self) -> int:
        return (
            self.trend_score
            + self.momentum_score
            + self.volume_score
            + self.bottom_score
            + self.sr_score
            + self.rr_score
            + self.regime_score
        )


class MarketDataProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Return normalized OHLCV candles."""


class OutputWriter(Protocol):
    def __call__(self, payload: dict[str, Any], output_dir: Path) -> Path:
        """Persist a JSON-compatible output payload."""
