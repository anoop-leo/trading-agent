"""Market-structure stop placement for Phase 1.12 backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


STOP_TYPES = ("fixed", "atr", "swing_low", "support_zone")


@dataclass(frozen=True)
class SupportZone:
    support_low: float
    support_high: float
    strength: int

    @property
    def low(self) -> float:
        return self.support_low

    @property
    def high(self) -> float:
        return self.support_high

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class StructureStopResult:
    requested_stop_type: str
    stop_type: str
    stop_price: float | None
    atr: float | None
    swing_low: float | None
    support_zone: SupportZone | None
    stop_distance_pct: float | None
    stop_distance_atr: float | None
    atr_floor_applied: bool
    candidates: dict[str, dict[str, Any]]

    def to_signal_fields(self) -> dict[str, Any]:
        return {
            "stop_loss": self.stop_price,
            "stop_type": self.stop_type,
            "requested_stop_type": self.requested_stop_type,
            "atr": self.atr,
            "swing_low": self.swing_low,
            "support_zone": self.support_zone.to_dict() if self.support_zone is not None else None,
            "stop_distance_pct": self.stop_distance_pct,
            "stop_distance_atr": self.stop_distance_atr,
            "atr_floor_applied": self.atr_floor_applied,
            "stop_candidates": self.candidates,
        }


class StructureStopEngine:
    """Select deterministic stops from ATR, swing lows, and support zones."""

    def __init__(
        self,
        stop_type: str = "fixed",
        swing_lookback: int = 20,
        support_lookback: int = 50,
        atr_period: int = 14,
        atr_buffer: float = 0.5,
        minimum_atr_distance: float = 1.5,
        collect_candidates: bool = False,
    ) -> None:
        normalized = stop_type.lower()
        if normalized not in STOP_TYPES:
            supported = ", ".join(STOP_TYPES)
            raise ValueError(f"Unsupported stop type {stop_type!r}. Supported stop types: {supported}.")
        if swing_lookback <= 0 or support_lookback <= 0 or atr_period <= 0:
            raise ValueError("lookbacks and ATR period must be positive.")
        if atr_buffer < 0 or minimum_atr_distance <= 0:
            raise ValueError("ATR stop settings must be non-negative and minimum distance must be positive.")
        self.stop_type = normalized
        self.swing_lookback = int(swing_lookback)
        self.support_lookback = int(support_lookback)
        self.atr_period = int(atr_period)
        self.atr_buffer = float(atr_buffer)
        self.minimum_atr_distance = float(minimum_atr_distance)
        self.collect_candidates = bool(collect_candidates)

    def evaluate(
        self,
        frame: pd.DataFrame,
        entry_price: float,
        fixed_stop: float | None = None,
        support: float | None = None,
    ) -> StructureStopResult:
        del support
        price = float(entry_price)
        atr = calculate_atr(frame, self.atr_period)
        needs_swing_low = self.collect_candidates or self.stop_type in {"swing_low", "support_zone"}
        needs_support_zone = self.collect_candidates or self.stop_type == "support_zone"
        swing_low_payload = find_recent_swing_low(frame, self.swing_lookback) if needs_swing_low else {"swing_low": None}
        swing_low = _float_or_none(swing_low_payload.get("swing_low"))
        support_zone = detect_support_zone(frame, lookback=self.support_lookback) if needs_support_zone else None
        fixed_candidate = self._candidate("FIXED", _valid_stop(fixed_stop, price), price, atr, False)
        atr_candidate = self._candidate(
            "ATR",
            self._atr_stop(price, atr),
            price,
            atr,
            atr is not None,
        )
        swing_candidate = self._structure_candidate("SWING_LOW", swing_low, price, atr)
        support_candidate = self._structure_candidate(
            "SUPPORT_ZONE",
            support_zone.support_low if support_zone is not None else None,
            price,
            atr,
        )
        candidates = {
            "fixed": fixed_candidate,
            "atr": atr_candidate,
            "swing_low": swing_candidate,
            "support_zone": support_candidate,
        }
        selected = self._select_candidate(candidates)
        return StructureStopResult(
            requested_stop_type=self.stop_type.upper(),
            stop_type=str(selected["stop_type"]),
            stop_price=_round_or_none(selected["stop_price"]),
            atr=_round_or_none(atr),
            swing_low=_round_or_none(swing_low),
            support_zone=support_zone,
            stop_distance_pct=selected["stop_distance_pct"],
            stop_distance_atr=selected["stop_distance_atr"],
            atr_floor_applied=bool(selected["atr_floor_applied"]),
            candidates=candidates,
        )

    def _select_candidate(self, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if self.stop_type == "fixed":
            return candidates["fixed"]
        if self.stop_type == "atr":
            return candidates["atr"]
        if self.stop_type == "swing_low":
            return _first_available(candidates["swing_low"], candidates["atr"])
        return _first_available(candidates["support_zone"], candidates["swing_low"], candidates["atr"])

    def _structure_candidate(
        self,
        stop_type: str,
        structure_low: float | None,
        entry_price: float,
        atr: float | None,
    ) -> dict[str, Any]:
        if structure_low is None or atr is None or atr <= 0:
            return self._candidate(stop_type, None, entry_price, atr, False)
        raw_stop = float(structure_low) - (self.atr_buffer * atr)
        stop_price, floor_applied = self._apply_atr_floor(raw_stop, entry_price, atr)
        return self._candidate(stop_type, stop_price, entry_price, atr, floor_applied)

    def _atr_stop(self, entry_price: float, atr: float | None) -> float | None:
        if atr is None or atr <= 0:
            return None
        return entry_price - (self.minimum_atr_distance * atr)

    def _apply_atr_floor(self, stop_price: float, entry_price: float, atr: float) -> tuple[float, bool]:
        minimum_stop = self._atr_stop(entry_price, atr)
        if minimum_stop is None:
            return stop_price, False
        if entry_price - stop_price < self.minimum_atr_distance * atr:
            return minimum_stop, True
        return stop_price, False

    def _candidate(
        self,
        stop_type: str,
        stop_price: float | None,
        entry_price: float,
        atr: float | None,
        atr_floor_applied: bool,
    ) -> dict[str, Any]:
        stop = _valid_stop(stop_price, entry_price)
        distance = entry_price - stop if stop is not None else None
        return {
            "stop_type": stop_type,
            "stop_price": _round_or_none(stop),
            "stop_distance_pct": _round_or_none((distance / entry_price) * 100 if distance is not None else None),
            "stop_distance_atr": _round_or_none(distance / atr if distance is not None and atr is not None and atr > 0 else None),
            "atr_floor_applied": atr_floor_applied,
        }


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> float | None:
    """Calculate ATR(14) by default from OHLC candles."""

    if frame.empty or period <= 0:
        return None
    required = {"high", "low", "close"}
    if not required.issubset(frame.columns):
        return None
    data = frame[list(required)].astype(float).reset_index(drop=True)
    true_ranges: list[float] = []
    previous_close: float | None = None
    for row in data.itertuples(index=False):
        high = float(row.high)
        low = float(row.low)
        if previous_close is None:
            true_range = high - low
        else:
            true_range = max(high - low, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(true_range)
        previous_close = float(row.close)
    if not true_ranges:
        return None
    return sum(true_ranges[-period:]) / min(len(true_ranges), period)


def find_recent_swing_low(frame: pd.DataFrame, lookback: int = 20, swing_window: int = 2) -> dict[str, float | None]:
    """Find the most recent low surrounded by higher lows."""

    points = _swing_low_points(frame, lookback=lookback, swing_window=swing_window)
    return {"swing_low": _round_or_none(points[-1][1]) if points else None}


def detect_support_zone(
    candles: pd.DataFrame,
    lookback: int = 50,
    swing_window: int = 2,
    cluster_tolerance_pct: float = 0.005,
    minimum_touches: int = 2,
) -> SupportZone | None:
    """Detect the strongest clustered swing-low support zone."""

    if lookback <= 0 or cluster_tolerance_pct < 0 or minimum_touches <= 0:
        return None
    swing_lows = _swing_low_points(candles, lookback=lookback, swing_window=swing_window)
    if not swing_lows:
        return None
    clusters: list[list[tuple[int, float]]] = []
    for point in sorted(swing_lows, key=lambda item: item[1]):
        _, low = point
        matched: list[tuple[int, float]] | None = None
        for cluster in clusters:
            center = sum(value for _, value in cluster) / len(cluster)
            tolerance = max(abs(center) * cluster_tolerance_pct, 1e-9)
            if abs(low - center) <= tolerance:
                matched = cluster
                break
        if matched is None:
            clusters.append([point])
        else:
            matched.append(point)

    valid_clusters = [cluster for cluster in clusters if len(cluster) >= minimum_touches]
    if not valid_clusters:
        return None
    strongest = max(
        valid_clusters,
        key=lambda cluster: (
            min(10, len(cluster) * 2),
            len(cluster),
            max(index for index, _ in cluster),
        ),
    )
    lows = [value for _, value in strongest]
    return SupportZone(
        support_low=_round_or_none(min(lows)) or min(lows),
        support_high=_round_or_none(max(lows)) or max(lows),
        strength=min(10, len(strongest) * 2),
    )


def detect_recent_swing_low(frame: pd.DataFrame, lookback: int = 20, swing_window: int = 2) -> float | None:
    """Backward-compatible float return for older backtesting code/tests."""

    return _float_or_none(find_recent_swing_low(frame, lookback, swing_window).get("swing_low"))


def _swing_low_points(frame: pd.DataFrame, lookback: int, swing_window: int) -> list[tuple[int, float]]:
    if frame.empty or "low" not in frame.columns or lookback <= 0 or swing_window <= 0:
        return []
    lows = [float(value) for value in frame["low"].tail(lookback).reset_index(drop=True)]
    if len(lows) < (swing_window * 2) + 1:
        return []
    swing_lows: list[tuple[int, float]] = []
    for index in range(swing_window, len(lows) - swing_window):
        value = lows[index]
        left = lows[index - swing_window : index]
        right = lows[index + 1 : index + swing_window + 1]
        if left and right and value < min(left) and value < min(right):
            swing_lows.append((index, value))
    return swing_lows


def _first_available(*candidates: dict[str, Any]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("stop_price") is not None:
            return candidate
    return candidates[-1]


def _valid_stop(stop_price: float | None, entry_price: float) -> float | None:
    if stop_price is None or pd.isna(stop_price):
        return None
    stop = float(stop_price)
    if stop <= 0 or stop >= entry_price:
        return None
    return stop


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
