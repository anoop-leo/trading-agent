"""Multi-timeframe alignment skill for Phase 1.3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


DEFAULT_TIMEFRAMES = ("1h", "4h", "1d")
BULLISH_SETUPS = {"BREAKOUT", "TREND_FOLLOWING"}
BEARISH_SETUPS = {"BEAR_TREND"}
RANGE_SETUPS = {"RANGE_BOUND"}
SUPPORTED_SETUPS = {
    "BOTTOMING",
    "BREAKOUT",
    "TREND_FOLLOWING",
    "PULLBACK",
    "RANGE_BOUND",
    "BEAR_TREND",
}
SUPPORTED_REGIMES = {"BULL", "NEUTRAL", "BEAR"}


class Alignment(str, Enum):
    BULLISH_ALIGNMENT = "BULLISH_ALIGNMENT"
    BEARISH_ALIGNMENT = "BEARISH_ALIGNMENT"
    REVERSAL_FORMING = "REVERSAL_FORMING"
    PULLBACK_IN_UPTREND = "PULLBACK_IN_UPTREND"
    PULLBACK_IN_DOWNTREND = "PULLBACK_IN_DOWNTREND"
    MIXED_ALIGNMENT = "MIXED_ALIGNMENT"
    RANGE_ALIGNMENT = "RANGE_ALIGNMENT"


class MultiTimeframeError(ValueError):
    """Raised when multi-timeframe alignment cannot be calculated."""


@dataclass(frozen=True)
class TimeframeSignal:
    timeframe: str
    trend_score: int
    momentum_score: int
    volume_score: int
    bottom_score: int
    sr_score: int
    rr_score: int
    regime_score: int
    setup: str
    setup_confidence: int
    decision: str
    price: float
    rsi: float
    macd: str
    ema20: float
    ema50: float
    ema200: float
    market_regime: str


@dataclass(frozen=True)
class MultiTimeframeResult:
    timeframes: dict[str, TimeframeSignal]
    alignment: Alignment
    alignment_score: int
    summary: str


def _validate_score(score: int, name: str) -> None:
    if not 0 <= score <= 10:
        raise MultiTimeframeError(f"{name} must be between 0 and 10.")


def _validate_signal(signal: TimeframeSignal) -> None:
    if not signal.timeframe:
        raise MultiTimeframeError("timeframe must not be empty.")
    for name in (
        "trend_score",
        "momentum_score",
        "volume_score",
        "bottom_score",
        "sr_score",
        "rr_score",
        "regime_score",
    ):
        _validate_score(getattr(signal, name), name)
    if signal.setup not in SUPPORTED_SETUPS:
        raise MultiTimeframeError("setup must be a supported Phase 1 setup.")
    if not 0 <= signal.setup_confidence <= 100:
        raise MultiTimeframeError("setup_confidence must be between 0 and 100.")
    if signal.market_regime not in SUPPORTED_REGIMES:
        raise MultiTimeframeError("market_regime must be BULL, NEUTRAL, or BEAR.")
    for name in ("price", "ema20", "ema50", "ema200"):
        if getattr(signal, name) <= 0:
            raise MultiTimeframeError(f"{name} must be greater than zero.")
    if signal.macd not in {"bullish", "bearish", "neutral"}:
        raise MultiTimeframeError("macd must be bullish, bearish, or neutral.")


def _ordered_signals(signals: Mapping[str, TimeframeSignal]) -> dict[str, TimeframeSignal]:
    if not signals:
        raise MultiTimeframeError("At least one timeframe signal is required.")
    ordered: dict[str, TimeframeSignal] = {}
    for timeframe, signal in signals.items():
        if timeframe != signal.timeframe:
            raise MultiTimeframeError("Timeframe map keys must match signal timeframes.")
        if timeframe in ordered:
            raise MultiTimeframeError(f"Duplicate timeframe {timeframe!r}.")
        _validate_signal(signal)
        ordered[timeframe] = signal
    return ordered


def _daily_signal(signals: Mapping[str, TimeframeSignal]) -> TimeframeSignal | None:
    return signals.get("1d")


def _is_bullish_context(signal: TimeframeSignal | None) -> bool:
    return (
        signal is not None
        and signal.setup in BULLISH_SETUPS
        and signal.market_regime != "BEAR"
    ) or (
        signal is not None
        and signal.market_regime == "BULL"
        and signal.trend_score >= 7
    )


def _is_daily_bullish_trend(signal: TimeframeSignal | None) -> bool:
    return (
        signal is not None
        and signal.market_regime == "BULL"
        and signal.price > signal.ema200
        and (signal.setup in BULLISH_SETUPS or signal.trend_score >= 7)
    )


def _is_bearish_context(signal: TimeframeSignal | None) -> bool:
    return signal is not None and (signal.setup == "BEAR_TREND" or signal.market_regime == "BEAR")


def _is_medium_term_pullback(signal: TimeframeSignal | None) -> bool:
    return signal is not None and signal.setup == "PULLBACK" and signal.market_regime != "BEAR"


def _is_short_term_weakness(signal: TimeframeSignal | None) -> bool:
    return signal is not None and (
        signal.setup in {"PULLBACK", "RANGE_BOUND", "BEAR_TREND", "BOTTOMING"}
        or signal.price < signal.ema20
        or signal.momentum_score <= 5
        or signal.macd == "bearish"
    )


def _is_pullback_in_uptrend(
    daily: TimeframeSignal | None,
    medium_term: TimeframeSignal | None,
    short_term: TimeframeSignal | None,
) -> bool:
    return (
        _is_daily_bullish_trend(daily)
        and _is_medium_term_pullback(medium_term)
        and _is_short_term_weakness(short_term)
    )


def _is_short_term_countertrend_bounce(signal: TimeframeSignal | None) -> bool:
    return signal is not None and signal.setup in {"BREAKOUT", "TREND_FOLLOWING", "PULLBACK", "RANGE_BOUND"}


def _count_setups(signals: Mapping[str, TimeframeSignal], setups: set[str]) -> int:
    return sum(1 for signal in signals.values() if signal.setup in setups)


def _volume_confirms(signals: Mapping[str, TimeframeSignal]) -> bool:
    average_volume_score = sum(signal.volume_score for signal in signals.values()) / len(signals)
    return average_volume_score >= 7


def _classify_alignment(signals: Mapping[str, TimeframeSignal]) -> Alignment:
    bullish_count = _count_setups(signals, BULLISH_SETUPS)
    bearish_count = _count_setups(signals, BEARISH_SETUPS)
    range_count = _count_setups(signals, RANGE_SETUPS)
    daily = _daily_signal(signals)
    medium_term = signals.get("4h")
    daily_is_bullish = _is_bullish_context(daily)
    daily_is_bearish = _is_bearish_context(daily)
    short_term = signals.get("1h")
    higher_timeframe_bearish = any(
        signal.setup == "BEAR_TREND" for timeframe, signal in signals.items() if timeframe in {"4h", "1d"}
    )

    if short_term is not None and short_term.setup == "BOTTOMING" and higher_timeframe_bearish:
        return Alignment.REVERSAL_FORMING
    if _is_pullback_in_uptrend(daily, medium_term, short_term):
        return Alignment.PULLBACK_IN_UPTREND
    if daily_is_bearish and _is_short_term_countertrend_bounce(short_term):
        return Alignment.PULLBACK_IN_DOWNTREND
    if bullish_count >= 2 and not daily_is_bearish:
        return Alignment.BULLISH_ALIGNMENT
    if bearish_count >= 2 and daily_is_bearish:
        return Alignment.BEARISH_ALIGNMENT
    if range_count >= 2:
        return Alignment.RANGE_ALIGNMENT
    return Alignment.MIXED_ALIGNMENT


def _alignment_score(alignment: Alignment, signals: Mapping[str, TimeframeSignal]) -> int:
    bullish_count = _count_setups(signals, BULLISH_SETUPS)
    bearish_count = _count_setups(signals, BEARISH_SETUPS)
    range_count = _count_setups(signals, RANGE_SETUPS)
    daily = _daily_signal(signals)
    medium_term = signals.get("4h")
    short_term = signals.get("1h")
    daily_bullish = _is_bullish_context(daily)
    daily_bearish = _is_bearish_context(daily)
    volume_bonus = 5 if _volume_confirms(signals) else 0

    if alignment == Alignment.BULLISH_ALIGNMENT:
        score = 80 + (bullish_count * 5) + (5 if daily_bullish else 0) + volume_bonus
        return max(80, min(100, score))
    if alignment == Alignment.BEARISH_ALIGNMENT:
        score = 30 - (max(0, bearish_count - 2) * 5) - (5 if daily_bearish else 0) - volume_bonus
        return max(0, min(30, score))
    if alignment == Alignment.REVERSAL_FORMING:
        confidence_bonus = round((short_term.setup_confidence if short_term else 50) / 10)
        score = 40 + confidence_bonus + volume_bonus
        return max(40, min(60, score))
    if alignment == Alignment.PULLBACK_IN_UPTREND:
        medium_bonus = 5 if _is_medium_term_pullback(medium_term) else 0
        score = 60 + (5 if daily_bullish else 0) + medium_bonus + volume_bonus
        return max(60, min(80, score))
    if alignment == Alignment.PULLBACK_IN_DOWNTREND:
        medium_penalty = 5 if _is_bearish_context(medium_term) else 0
        score = 45 - (5 if daily_bearish else 0) - medium_penalty - volume_bonus
        return max(25, min(45, score))
    if alignment == Alignment.RANGE_ALIGNMENT:
        daily_range = daily is not None and daily.setup == "RANGE_BOUND"
        score = 35 + (range_count * 5) + (5 if daily_range else 0)
        return max(35, min(55, score))

    score = 50 + ((bullish_count - bearish_count) * 5)
    return max(40, min(60, score))


def _summary(alignment: Alignment, signals: Mapping[str, TimeframeSignal]) -> str:
    bullish_count = _count_setups(signals, BULLISH_SETUPS)
    bearish_count = _count_setups(signals, BEARISH_SETUPS)
    range_count = _count_setups(signals, RANGE_SETUPS)

    if alignment == Alignment.BULLISH_ALIGNMENT:
        return f"{bullish_count} of {len(signals)} timeframes are bullish. Long setups are allowed."
    if alignment == Alignment.BEARISH_ALIGNMENT:
        if bearish_count == len(signals):
            return "All major timeframes are bearish. Avoid long trades."
        return f"{bearish_count} of {len(signals)} timeframes are bearish and daily confirms. Avoid long trades."
    if alignment == Alignment.REVERSAL_FORMING:
        higher_bearish = [
            timeframe
            for timeframe, signal in signals.items()
            if timeframe in {"4h", "1d"} and signal.setup == "BEAR_TREND"
        ]
        if set(higher_bearish) == {"4h", "1d"}:
            return "1h is bottoming, but 4h and 1d remain bearish."
        joined = " and ".join(higher_bearish) if higher_bearish else "higher timeframes"
        return f"1h is bottoming, but {joined} remains bearish."
    if alignment == Alignment.PULLBACK_IN_UPTREND:
        return "Daily trend is bullish, 4h is pulling back, and 1h shows weakness. Watch for continuation."
    if alignment == Alignment.PULLBACK_IN_DOWNTREND:
        return "Daily trend is bearish while the short-term timeframe is bouncing. Avoid chasing long entries."
    if alignment == Alignment.RANGE_ALIGNMENT:
        return f"{range_count} of {len(signals)} timeframes are range-bound. Avoid trend trades."
    return "Timeframes conflict. Wait for cleaner alignment."


def analyze_multi_timeframe(signals: Mapping[str, TimeframeSignal]) -> MultiTimeframeResult:
    """Classify alignment across timeframe-level signal snapshots."""

    ordered = _ordered_signals(signals)
    alignment = _classify_alignment(ordered)
    return MultiTimeframeResult(
        timeframes=ordered,
        alignment=alignment,
        alignment_score=_alignment_score(alignment, ordered),
        summary=_summary(alignment, ordered),
    )


def chart_annotation(result: MultiTimeframeResult) -> str:
    """Return a compact chart title annotation for multi-timeframe context."""

    lines = [f"{timeframe}: {signal.setup}" for timeframe, signal in result.timeframes.items()]
    lines.append(f"Alignment: {result.alignment.value}")
    return "\n".join(lines)
