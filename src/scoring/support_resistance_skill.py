"""Support and resistance scoring skill."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SupportResistanceError(ValueError):
    """Raised when support/resistance cannot be calculated."""


@dataclass(frozen=True)
class SupportResistanceResult:
    support: float
    resistance: float
    distance_to_support: float
    distance_to_resistance: float
    sr_score: int


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in ("high", "low") if column not in frame.columns]
    if missing:
        raise SupportResistanceError(f"OHLCV frame missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise SupportResistanceError("OHLCV frame is empty.")


def _swing_lows(frame: pd.DataFrame, window: int) -> list[tuple[int, float]]:
    lows = frame["low"].reset_index(drop=True)
    swings: list[tuple[int, float]] = []
    for index in range(window, len(lows) - window):
        value = float(lows.iloc[index])
        neighbors = lows.iloc[index - window : index + window + 1]
        if value == float(neighbors.min()) and value < float(neighbors.drop(index).min()):
            swings.append((index, value))
    return swings


def _swing_highs(frame: pd.DataFrame, window: int) -> list[tuple[int, float]]:
    highs = frame["high"].reset_index(drop=True)
    swings: list[tuple[int, float]] = []
    for index in range(window, len(highs) - window):
        value = float(highs.iloc[index])
        neighbors = highs.iloc[index - window : index + window + 1]
        if value == float(neighbors.max()) and value > float(neighbors.drop(index).max()):
            swings.append((index, value))
    return swings


def _most_recent_level(swings: list[tuple[int, float]], current_price: float, side: str) -> float | None:
    if side == "support":
        candidates = [value for _index, value in swings if value <= current_price]
    else:
        candidates = [value for _index, value in swings if value >= current_price]
    return candidates[-1] if candidates else None


def _score_support_resistance(current_price: float, support: float, resistance: float) -> int:
    if resistance <= support:
        return 5

    distance_to_support = max(0.0, current_price - support)
    distance_to_resistance = max(0.0, resistance - current_price)
    price_range = resistance - support

    near_support = distance_to_support <= price_range * 0.25
    far_from_resistance = distance_to_resistance >= price_range * 0.50
    near_resistance = distance_to_resistance <= price_range * 0.25

    if near_resistance:
        return 0
    if near_support and far_from_resistance:
        return 10
    return 5


def calculate_support_resistance(
    frame: pd.DataFrame,
    current_price: float,
    swing_window: int = 2,
    fallback_window: int = 20,
) -> SupportResistanceResult:
    """Identify nearest meaningful support/resistance and score trade location."""

    _validate_frame(frame)
    if current_price <= 0:
        raise SupportResistanceError("current_price must be greater than zero.")
    if swing_window <= 0 or fallback_window <= 0:
        raise SupportResistanceError("swing_window and fallback_window must be greater than zero.")

    recent_frame = frame.tail(fallback_window)
    support = _most_recent_level(_swing_lows(frame, swing_window), current_price, "support")
    resistance = _most_recent_level(_swing_highs(frame, swing_window), current_price, "resistance")

    if support is None:
        support = float(recent_frame["low"].min())
    if resistance is None:
        resistance = float(recent_frame["high"].max())

    return SupportResistanceResult(
        support=support,
        resistance=resistance,
        distance_to_support=max(0.0, current_price - support),
        distance_to_resistance=max(0.0, resistance - current_price),
        sr_score=_score_support_resistance(current_price, support, resistance),
    )
