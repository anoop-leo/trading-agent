"""Market regime scoring skill."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MarketRegime(str, Enum):
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


class MarketRegimeError(ValueError):
    """Raised when market regime cannot be calculated."""


@dataclass(frozen=True)
class MarketRegimeResult:
    market_regime: MarketRegime
    regime_score: int


def calculate_market_regime(current_price: float, ema50: float, ema200: float) -> MarketRegimeResult:
    """Classify BTC market regime from price versus EMA50/EMA200."""

    if current_price <= 0 or ema50 <= 0 or ema200 <= 0:
        raise MarketRegimeError("current_price, ema50, and ema200 must be greater than zero.")

    if current_price > ema200:
        return MarketRegimeResult(market_regime=MarketRegime.BULL, regime_score=10)
    if current_price >= ema50:
        return MarketRegimeResult(market_regime=MarketRegime.NEUTRAL, regime_score=5)
    return MarketRegimeResult(market_regime=MarketRegime.BEAR, regime_score=0)
