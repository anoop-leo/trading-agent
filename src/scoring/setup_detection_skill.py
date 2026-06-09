"""Setup detection skill for Phase 1.2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Setup(str, Enum):
    BOTTOMING = "BOTTOMING"
    BREAKOUT = "BREAKOUT"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    PULLBACK = "PULLBACK"
    RANGE_BOUND = "RANGE_BOUND"
    BEAR_TREND = "BEAR_TREND"


class SetupDetectionError(ValueError):
    """Raised when setup detection inputs are invalid."""


@dataclass(frozen=True)
class SetupInput:
    price: float
    ema20: float
    ema50: float
    ema200: float
    support: float
    resistance: float
    volume_ratio: float
    trend_score: int
    momentum_score: int
    bottom_score: int
    market_regime: str
    rsi: float | None = None


@dataclass(frozen=True)
class SetupResult:
    setup: Setup
    setup_score: int
    setup_confidence: int
    setup_reason: list[str]


def _validate_score(score: int, name: str) -> None:
    if not 0 <= score <= 10:
        raise SetupDetectionError(f"{name} must be between 0 and 10.")


def _validate_input(setup_input: SetupInput) -> None:
    for name in ("trend_score", "momentum_score", "bottom_score"):
        _validate_score(getattr(setup_input, name), name)
    for name in ("price", "ema20", "ema50", "ema200", "support", "resistance"):
        if getattr(setup_input, name) <= 0:
            raise SetupDetectionError(f"{name} must be greater than zero.")
    if setup_input.volume_ratio < 0:
        raise SetupDetectionError("volume_ratio must not be negative.")
    if setup_input.market_regime not in {"BULL", "NEUTRAL", "BEAR"}:
        raise SetupDetectionError("market_regime must be BULL, NEUTRAL, or BEAR.")


def _confidence(setup_score: int) -> int:
    return min(100, round(setup_score * 9))


def _result(setup: Setup, setup_score: int, reasons: list[str]) -> SetupResult:
    bounded_score = max(0, min(10, setup_score))
    return SetupResult(
        setup=setup,
        setup_score=bounded_score,
        setup_confidence=_confidence(bounded_score),
        setup_reason=reasons,
    )


def detect_setup(setup_input: SetupInput) -> SetupResult:
    """Classify the current market structure into one deterministic setup."""

    _validate_input(setup_input)

    if (
        setup_input.price > setup_input.resistance
        and setup_input.volume_ratio > 1.5
        and setup_input.momentum_score >= 6
    ):
        volume_score = min(10, round(setup_input.volume_ratio * 4))
        score = round((setup_input.momentum_score + volume_score) / 2)
        return _result(
            Setup.BREAKOUT,
            score,
            ["Price broke above resistance", "Volume expansion confirms breakout", "Momentum supports continuation"],
        )

    if (
        setup_input.price > setup_input.ema20
        and setup_input.price > setup_input.ema50
        and setup_input.price > setup_input.ema200
        and setup_input.trend_score >= 8
    ):
        return _result(
            Setup.TREND_FOLLOWING,
            setup_input.trend_score,
            ["Price is above EMA20, EMA50, and EMA200", "Trend score confirms strong uptrend"],
        )

    if (
        setup_input.bottom_score >= 7
        and setup_input.trend_score <= 5
        and setup_input.market_regime == "BEAR"
    ):
        reasons = ["Bottom score is elevated", "Trend remains weak", "Market regime is BEAR"]
        if setup_input.price <= setup_input.support * 1.03:
            reasons.append("Price near support")
        if setup_input.rsi is not None and setup_input.rsi >= 30:
            reasons.append("RSI recovering")
        return _result(Setup.BOTTOMING, setup_input.bottom_score, reasons)

    ema_low = min(setup_input.ema20, setup_input.ema50)
    ema_high = max(setup_input.ema20, setup_input.ema50)
    if setup_input.market_regime == "BULL" and ema_low <= setup_input.price <= ema_high and setup_input.momentum_score >= 5:
        score = round((setup_input.momentum_score + 10) / 2)
        return _result(
            Setup.PULLBACK,
            score,
            ["Market regime is BULL", "Price is between EMA20 and EMA50", "Momentum remains constructive"],
        )

    if 4 <= setup_input.trend_score <= 6 and 4 <= setup_input.momentum_score <= 6:
        score = round((setup_input.trend_score + setup_input.momentum_score) / 2)
        return _result(
            Setup.RANGE_BOUND,
            score,
            ["Trend score is neutral", "Momentum score is neutral", "No directional setup has priority"],
        )

    if (
        setup_input.price < setup_input.ema20
        and setup_input.price < setup_input.ema50
        and setup_input.price < setup_input.ema200
        and setup_input.trend_score <= 3
    ):
        return _result(
            Setup.BEAR_TREND,
            10 - setup_input.trend_score,
            ["Price is below EMA20, EMA50, and EMA200", "Trend score confirms strong downtrend"],
        )

    return _result(
        Setup.RANGE_BOUND,
        4,
        ["No higher-priority setup matched", "Market structure is inconclusive"],
    )
