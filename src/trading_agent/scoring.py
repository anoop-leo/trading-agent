"""Skill scoring rules for Phase 1 signals."""

from __future__ import annotations

import math

import pandas as pd

from trading_agent.models import SignalScores


REQUIRED_SCORE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volume",
    "volume_ma_20",
)


class ScoringError(ValueError):
    """Raised when skill scores cannot be calculated."""


def _is_present(value: object) -> bool:
    return value is not None and not pd.isna(value)


def _latest_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ScoringError("Cannot score an empty indicator frame.")
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ScoringError(f"Indicator frame missing required columns: {', '.join(missing)}.")
    return frame.iloc[-1]


def score_trend(row: pd.Series) -> int:
    """Score trend strength from latest price versus EMA levels."""

    close = row["close"]
    score = 0
    if _is_present(row["ema_20"]) and close > row["ema_20"]:
        score += 2
    if _is_present(row["ema_50"]) and close > row["ema_50"]:
        score += 3
    if _is_present(row["ema_200"]) and close > row["ema_200"]:
        score += 5
    return score


def score_momentum(row: pd.Series, previous_row: pd.Series | None = None) -> int:
    """Score momentum from RSI and MACD conditions."""

    score = 0
    rsi = row["rsi_14"]
    if _is_present(rsi) and 50 <= rsi <= 70:
        score += 4
    if _is_present(row["macd"]) and _is_present(row["macd_signal"]) and row["macd"] > row["macd_signal"]:
        score += 4
        if (
            previous_row is not None
            and _is_present(row["macd_histogram"])
            and _is_present(previous_row["macd_histogram"])
            and row["macd_histogram"] > previous_row["macd_histogram"]
        ):
            score += 2
    return score


def calculate_volume_ratio(row: pd.Series) -> float:
    """Calculate latest volume as a ratio of average volume."""

    volume_average = row["volume_ma_20"]
    if not _is_present(volume_average) or math.isclose(float(volume_average), 0.0):
        return 0.0
    return float(row["volume"] / volume_average)


def score_volume(row: pd.Series) -> int:
    """Score volume expansion versus rolling average volume."""

    ratio = calculate_volume_ratio(row)
    if ratio <= 1.0:
        return 0
    if ratio >= 2.0:
        return 10
    return max(0, min(10, round((ratio - 1.0) * 9)))


def calculate_recent_swing_high(indicator_frame: pd.DataFrame, window: int = 20) -> float:
    """Calculate recent swing high from the latest window."""

    if window <= 0:
        raise ScoringError("window must be greater than zero.")
    if "high" not in indicator_frame.columns:
        raise ScoringError("Indicator frame missing required column: high.")
    return float(indicator_frame["high"].tail(window).max())


def calculate_recent_swing_low(indicator_frame: pd.DataFrame, window: int = 20) -> float:
    """Calculate recent swing low from the latest window."""

    if window <= 0:
        raise ScoringError("window must be greater than zero.")
    if "low" not in indicator_frame.columns:
        raise ScoringError("Indicator frame missing required column: low.")
    return float(indicator_frame["low"].tail(window).min())


def score_bottom(row: pd.Series, previous_row: pd.Series | None, recent_swing_low: float) -> int:
    """Score early reversal/bottom formation from price, RSI, and MACD improvement."""

    score = 0
    close = float(row["close"])

    if close <= recent_swing_low * 1.03:
        score += 4

    rsi = row["rsi_14"]
    if _is_present(rsi):
        if 30 <= rsi <= 45:
            score += 3
        elif rsi < 30:
            score += 2
        elif 45 < rsi <= 55:
            score += 2

    if (
        previous_row is not None
        and _is_present(row["macd_histogram"])
        and _is_present(previous_row["macd_histogram"])
        and row["macd_histogram"] > previous_row["macd_histogram"]
    ):
        score += 2

    if close > row["open"]:
        score += 1

    score = min(score, 10)
    volume_ratio = calculate_volume_ratio(row)
    if volume_ratio < 0.5:
        return min(score, 4)
    if volume_ratio < 0.8:
        return min(score, 6)
    return score


def calculate_scores(indicator_frame: pd.DataFrame) -> SignalScores:
    """Calculate Phase 1 skill scores from the latest indicator row."""

    row = _latest_row(indicator_frame)
    previous_row = indicator_frame.iloc[-2] if len(indicator_frame) > 1 else None
    recent_swing_low = calculate_recent_swing_low(indicator_frame)
    return SignalScores(
        trend_score=score_trend(row),
        momentum_score=score_momentum(row, previous_row),
        volume_score=score_volume(row),
        bottom_score=score_bottom(row, previous_row, recent_swing_low),
    )
