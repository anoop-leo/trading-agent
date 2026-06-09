"""Deterministic Decision Engine v2 for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    BUY = "BUY"
    BUY_WATCH = "BUY WATCH"
    WAIT = "WAIT"
    AVOID_LONG = "AVOID LONG"
    WATCH_FOR_REVERSAL = "WATCH FOR REVERSAL"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


@dataclass(frozen=True)
class PriceZone:
    low: float
    high: float


@dataclass(frozen=True)
class DecisionInput:
    trend_score: int
    momentum_score: int
    volume_score: int
    bottom_score: int
    sr_score: int
    rr_score: int
    regime_score: int
    current_price: float
    ema20: float
    ema50: float
    ema200: float
    recent_swing_high: float
    recent_swing_low: float
    rr_ratio: float
    market_regime: str
    support: float | None = None
    setup: str | None = None
    setup_score: int | None = None
    setup_confidence: int | None = None
    volume_ratio: float | None = None
    position_mode: str = "NO_POSITION"
    symbol: str = "Asset"
    rsi: float | None = None
    macd: str | None = None


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    decision_meaning: str
    confidence: int
    entry_zone: PriceZone | None
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    rationale: list[str]


class DecisionError(ValueError):
    """Raised when a deterministic decision cannot be made."""


def _validate_score(score: int, name: str) -> None:
    if not 0 <= score <= 10:
        raise DecisionError(f"{name} must be between 0 and 10.")


def _validate_input(decision_input: DecisionInput) -> None:
    _validate_score(decision_input.trend_score, "trend_score")
    _validate_score(decision_input.momentum_score, "momentum_score")
    _validate_score(decision_input.volume_score, "volume_score")
    _validate_score(decision_input.bottom_score, "bottom_score")
    _validate_score(decision_input.sr_score, "sr_score")
    _validate_score(decision_input.rr_score, "rr_score")
    _validate_score(decision_input.regime_score, "regime_score")

    prices = {
        "current_price": decision_input.current_price,
        "ema20": decision_input.ema20,
        "ema50": decision_input.ema50,
        "ema200": decision_input.ema200,
        "recent_swing_high": decision_input.recent_swing_high,
        "recent_swing_low": decision_input.recent_swing_low,
    }
    invalid = [name for name, value in prices.items() if value <= 0]
    if invalid:
        raise DecisionError(f"Price inputs must be positive: {', '.join(invalid)}.")
    if decision_input.rr_ratio < 0:
        raise DecisionError("rr_ratio must not be negative.")
    if decision_input.market_regime not in {"BULL", "NEUTRAL", "BEAR"}:
        raise DecisionError("market_regime must be BULL, NEUTRAL, or BEAR.")
    if decision_input.support is not None and decision_input.support <= 0:
        raise DecisionError("support must be greater than zero when provided.")
    if decision_input.setup is not None and decision_input.setup not in {
        "BOTTOMING",
        "BREAKOUT",
        "TREND_FOLLOWING",
        "PULLBACK",
        "RANGE_BOUND",
        "BEAR_TREND",
    }:
        raise DecisionError("setup must be a supported Phase 1.2 setup.")
    if decision_input.setup_score is not None:
        _validate_score(decision_input.setup_score, "setup_score")
    if decision_input.setup_confidence is not None and not 0 <= decision_input.setup_confidence <= 100:
        raise DecisionError("setup_confidence must be between 0 and 100.")
    if decision_input.volume_ratio is not None and decision_input.volume_ratio < 0:
        raise DecisionError("volume_ratio must not be negative.")
    if decision_input.position_mode not in {"NO_POSITION", "HOLDING"}:
        raise DecisionError("position_mode must be NO_POSITION or HOLDING.")


def calculate_confidence(decision_input: DecisionInput) -> int:
    """Return seven-skill overall score as a 0-100 confidence value."""

    _validate_input(decision_input)
    total = (
        decision_input.trend_score
        + decision_input.momentum_score
        + decision_input.volume_score
        + decision_input.bottom_score
        + decision_input.sr_score
        + decision_input.rr_score
        + decision_input.regime_score
    )
    return round(total * 100 / 70)


def _has_strong_exit_conditions(decision_input: DecisionInput) -> bool:
    price_breaks_support = (
        decision_input.support is not None and decision_input.current_price < decision_input.support
    )
    return (
        decision_input.market_regime == "BEAR"
        and decision_input.trend_score <= 3
        and decision_input.momentum_score <= 3
        and decision_input.bottom_score <= 4
        and (decision_input.volume_score >= 5 or price_breaks_support)
    )


def _has_bearish_bottom_conflict(decision_input: DecisionInput) -> bool:
    return (
        decision_input.bottom_score >= 8
        and decision_input.trend_score <= 3
        and decision_input.momentum_score <= 3
        and decision_input.market_regime == "BEAR"
    )


def _classify_no_position(decision_input: DecisionInput, confidence: int) -> Decision:
    if decision_input.setup == "BREAKOUT":
        return Decision.BUY
    if decision_input.setup == "TREND_FOLLOWING":
        return Decision.BUY
    if decision_input.setup == "BOTTOMING":
        return Decision.WATCH_FOR_REVERSAL
    if decision_input.setup == "PULLBACK":
        return Decision.BUY_WATCH
    if decision_input.setup == "RANGE_BOUND":
        return Decision.WAIT
    if decision_input.setup == "BEAR_TREND":
        return Decision.AVOID_LONG

    if (
        confidence >= 85
        and decision_input.rr_ratio >= 2.0
        and decision_input.market_regime == "BULL"
    ):
        return Decision.BUY

    if confidence >= 70:
        return Decision.BUY

    if decision_input.bottom_score >= 8:
        return Decision.WATCH_FOR_REVERSAL

    if _has_strong_exit_conditions(decision_input):
        return Decision.AVOID_LONG

    if decision_input.market_regime == "BEAR" and decision_input.bottom_score < 8:
        return Decision.AVOID_LONG

    if confidence < 40:
        return Decision.AVOID_LONG

    if 40 <= confidence < 70:
        return Decision.WAIT

    return Decision.WAIT


def _classify_holding(decision_input: DecisionInput, confidence: int) -> Decision:
    if decision_input.setup == "BREAKOUT":
        return Decision.ADD
    if decision_input.setup == "TREND_FOLLOWING":
        return Decision.ADD if confidence >= 70 else Decision.HOLD
    if decision_input.setup == "PULLBACK":
        return Decision.HOLD
    if decision_input.setup == "BOTTOMING":
        return Decision.HOLD
    if decision_input.setup == "RANGE_BOUND":
        return Decision.HOLD
    if decision_input.setup == "BEAR_TREND":
        return Decision.EXIT if _has_strong_exit_conditions(decision_input) else Decision.REDUCE

    if _has_strong_exit_conditions(decision_input):
        return Decision.EXIT

    if decision_input.market_regime == "BEAR" and decision_input.bottom_score < 8:
        return Decision.REDUCE

    if (
        confidence >= 85
        and decision_input.rr_ratio >= 2.0
        and decision_input.market_regime == "BULL"
    ):
        return Decision.ADD

    if confidence >= 70 and decision_input.market_regime == "BULL":
        return Decision.ADD

    if confidence < 40:
        return Decision.REDUCE

    return Decision.HOLD


def _classify_decision(decision_input: DecisionInput, confidence: int) -> Decision:
    if decision_input.position_mode == "HOLDING":
        return _classify_holding(decision_input, confidence)
    return _classify_no_position(decision_input, confidence)


def _adjust_confidence(decision: Decision, decision_input: DecisionInput, confidence: int) -> int:
    if decision == Decision.WATCH_FOR_REVERSAL and _has_bearish_bottom_conflict(decision_input):
        return max(55, min(confidence, 65))
    return confidence


def _build_entry_zone(decision: Decision, decision_input: DecisionInput) -> PriceZone | None:
    if decision in {Decision.BUY, Decision.BUY_WATCH, Decision.ADD}:
        return PriceZone(low=decision_input.ema20, high=decision_input.current_price)
    if decision == Decision.WATCH_FOR_REVERSAL:
        return PriceZone(low=decision_input.recent_swing_low, high=decision_input.ema20)
    return None


def _build_stop_loss(decision: Decision, decision_input: DecisionInput) -> float | None:
    if decision in {Decision.BUY, Decision.BUY_WATCH, Decision.ADD}:
        return decision_input.recent_swing_low
    if decision == Decision.WATCH_FOR_REVERSAL:
        return decision_input.recent_swing_low * 0.985
    return None


def _build_targets(decision: Decision, decision_input: DecisionInput) -> tuple[float | None, float | None]:
    if decision in {Decision.BUY, Decision.BUY_WATCH, Decision.ADD}:
        target_1 = decision_input.ema200 if decision_input.current_price < decision_input.ema200 else None
        return target_1, decision_input.recent_swing_high
    if decision == Decision.WATCH_FOR_REVERSAL:
        return decision_input.ema50, decision_input.ema200
    return None, None


def _price_rationale(decision_input: DecisionInput) -> str:
    symbol = decision_input.symbol
    price = decision_input.current_price
    above_ema20 = price > decision_input.ema20
    above_ema50 = price > decision_input.ema50
    above_ema200 = price > decision_input.ema200

    if above_ema20 and above_ema50 and above_ema200:
        return f"{symbol} is above EMA20, EMA50, and EMA200"
    if above_ema20 and above_ema50 and not above_ema200:
        return f"{symbol} is above EMA20 and EMA50 but still below EMA200"
    if above_ema20 and not above_ema50:
        return f"{symbol} is above EMA20 but below longer trend averages"
    if not above_ema20 and not above_ema50 and not above_ema200:
        return f"{symbol} is below EMA20, EMA50, and EMA200"
    return f"{symbol} has mixed EMA alignment"


def _momentum_rationale(decision_input: DecisionInput) -> str:
    rsi_text = "RSI is unavailable"
    if decision_input.rsi is not None:
        if 50 <= decision_input.rsi <= 70:
            rsi_text = "RSI is healthy"
        elif decision_input.rsi < 30:
            rsi_text = "RSI is oversold"
        elif decision_input.rsi > 70:
            rsi_text = "RSI is extended"
        else:
            rsi_text = "RSI is neutral"

    macd = decision_input.macd or "unknown"
    return f"{rsi_text} and MACD is {macd}"


def _bottom_rationale(decision_input: DecisionInput) -> str:
    if _has_bearish_bottom_conflict(decision_input):
        return "Market is bearish, but bottom detection is strong. Wait for confirmation before buying."
    if decision_input.bottom_score >= 7:
        return "Bottom detection suggests early reversal formation"
    if decision_input.bottom_score < 5:
        return "Bottom detection does not confirm a reversal"
    return "Bottom detection is inconclusive"


def _bottom_volume_rationale(decision_input: DecisionInput) -> str | None:
    if decision_input.volume_ratio is not None and decision_input.volume_ratio < 0.8:
        return "Bottom signal lacks volume confirmation."
    return None


def _trade_quality_rationale(decision_input: DecisionInput) -> str:
    return (
        f"Trade quality scores are SR {decision_input.sr_score}/10, "
        f"RR {decision_input.rr_score}/10 at {decision_input.rr_ratio:.2f}R, "
        f"and market regime {decision_input.market_regime}"
    )


def _setup_rationale(decision_input: DecisionInput) -> str | None:
    if decision_input.setup is None:
        return None
    confidence = decision_input.setup_confidence
    if confidence is None:
        return f"Setup detected: {decision_input.setup}"
    return f"Setup detected: {decision_input.setup} with {confidence}% setup confidence"


def _decision_rationale(decision: Decision, confidence: int) -> str:
    return f"Decision is {decision.value} with {confidence}% confidence from deterministic Phase 1 rules"


def decision_meaning(decision: Decision, position_mode: str) -> str:
    """Explain a decision label in the context of the user's position mode."""

    no_position_meanings = {
        Decision.BUY: "Potential long entry setup detected.",
        Decision.BUY_WATCH: "Potential long setup, but wait for better confirmation before entering.",
        Decision.WATCH_FOR_REVERSAL: "Potential reversal forming. Wait for confirmation before buying.",
        Decision.WAIT: "No clear new long setup. Wait.",
        Decision.AVOID_LONG: "Do not open a new long position here.",
    }
    holding_meanings = {
        Decision.ADD: "Existing holding is in a favorable continuation setup; adding can be considered within risk limits.",
        Decision.HOLD: "Continue holding the existing position.",
        Decision.REDUCE: "Bearish risk is rising; consider reducing exposure.",
        Decision.EXIT: "Bearish conditions are strong; consider exiting the current position.",
    }
    if position_mode == "HOLDING":
        return holding_meanings.get(decision, "Decision applies to an existing holding.")
    return no_position_meanings.get(decision, "Decision applies to no-position mode.")


def build_rationale(decision: Decision, decision_input: DecisionInput, confidence: int) -> list[str]:
    rationale = [
        _price_rationale(decision_input),
        _momentum_rationale(decision_input),
        _bottom_rationale(decision_input),
    ]
    bottom_volume_rationale = _bottom_volume_rationale(decision_input)
    if bottom_volume_rationale is not None:
        rationale.append(bottom_volume_rationale)
    rationale.append(_trade_quality_rationale(decision_input))
    setup_rationale = _setup_rationale(decision_input)
    if setup_rationale is not None:
        rationale.append(setup_rationale)
    rationale.append(_decision_rationale(decision, confidence))
    return rationale


def make_decision(decision_input: DecisionInput) -> DecisionResult:
    """Make a deterministic Phase 1 trading decision."""

    raw_confidence = calculate_confidence(decision_input)
    decision = _classify_decision(decision_input, raw_confidence)
    confidence = _adjust_confidence(decision, decision_input, raw_confidence)
    target_1, target_2 = _build_targets(decision, decision_input)
    return DecisionResult(
        decision=decision,
        decision_meaning=decision_meaning(decision, decision_input.position_mode),
        confidence=confidence,
        entry_zone=_build_entry_zone(decision, decision_input),
        stop_loss=_build_stop_loss(decision, decision_input),
        target_1=target_1,
        target_2=target_2,
        rationale=build_rationale(decision, decision_input, confidence),
    )
