"""Deterministic BTC accumulation scoring for Investor Agent V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WEIGHTS = {
    "distance_from_200d_ma": 20,
    "mvrv": 20,
    "weekly_rsi": 15,
    "fear_and_greed": 15,
    "monthly_trend": 10,
    "drawdown_from_cycle_high": 10,
    "liquidity_condition": 5,
    "thesis_risk_check": 5,
}


@dataclass(frozen=True)
class AccumulationBand:
    accumulation_band: str
    dca_intensity: str
    suggested_dca_multiplier: str


def score_distance_from_200d_ma(distance_pct: float | None) -> int:
    if distance_pct is None:
        return 0
    if distance_pct <= -25:
        return 20
    if distance_pct <= -10:
        return 16
    if -10 < distance_pct <= 10:
        return 12
    if distance_pct <= 25:
        return 6
    if distance_pct <= 50:
        return 2
    return 0


def score_mvrv(value: float | None) -> int:
    if value is None:
        return 10
    if value < 1.0:
        return 20
    if value < 1.5:
        return 16
    if value < 2.0:
        return 12
    if value <= 3.0:
        return 6
    return 0


def score_weekly_rsi(value: float | None) -> int:
    if value is None:
        return 0
    if value < 30:
        return 15
    if value <= 40:
        return 12
    if value <= 50:
        return 9
    if value <= 60:
        return 6
    if value <= 70:
        return 3
    return 0


def score_fear_and_greed(value: float | None) -> int:
    if value is None:
        return 0
    if value <= 20:
        return 15
    if value <= 40:
        return 12
    if value <= 60:
        return 8
    if value <= 80:
        return 3
    return 0


def score_monthly_trend(value: str) -> int:
    normalized = value.upper()
    if normalized == "BELOW_EMA20":
        return 10
    if normalized == "NEAR_EMA20":
        return 8
    if normalized == "ABOVE_EMA20_EMA10_BELOW_EMA20":
        return 5
    if normalized == "BULLISH_ABOVE_EMA20":
        return 3
    if normalized == "EXTREMELY_EXTENDED":
        return 0
    return 0


def score_drawdown_from_cycle_high(drawdown_pct: float | None) -> int:
    if drawdown_pct is None:
        return 0
    if drawdown_pct <= -70:
        return 10
    if drawdown_pct <= -50:
        return 8
    if drawdown_pct <= -30:
        return 6
    if drawdown_pct <= -15:
        return 3
    if drawdown_pct < 0:
        return 1
    return 0


def score_liquidity_condition(value: str | None) -> int:
    normalized = (value or "NEUTRAL").upper()
    if normalized == "EXPANDING":
        return 5
    if normalized == "NEUTRAL":
        return 3
    if normalized == "CONTRACTING":
        return 0
    return 3


def score_thesis_risk(level: str | None) -> int:
    normalized = (level or "LOW").upper()
    if normalized == "LOW":
        return 5
    if normalized == "MODERATE":
        return 2
    if normalized == "HIGH":
        return 0
    return 2


def band_for_accumulation_score(score: float, thesis_risk_level: str = "LOW") -> AccumulationBand:
    if score < 30:
        band = AccumulationBand("EXPENSIVE", "MINIMAL_DCA", "0.0x to 0.25x normal DCA")
    elif score < 60:
        band = AccumulationBand("FAIR", "NORMAL_DCA", "0.5x to 1.0x normal DCA")
    elif score < 80:
        band = AccumulationBand("GOOD_ACCUMULATION", "INCREASED_DCA", "1.0x to 1.5x normal DCA")
    else:
        band = AccumulationBand("AGGRESSIVE_ACCUMULATION", "AGGRESSIVE_DCA", "1.5x to 2.5x normal DCA")

    if thesis_risk_level.upper() == "HIGH" and band.accumulation_band not in {"EXPENSIVE", "FAIR"}:
        return AccumulationBand("FAIR", "NORMAL_DCA", "0.5x to 1.0x normal DCA")
    if thesis_risk_level.upper() == "HIGH" and band.suggested_dca_multiplier == "1.5x to 2.5x normal DCA":
        return AccumulationBand(band.accumulation_band, band.dca_intensity, "0.5x to 1.0x normal DCA")
    return band


def calculate_accumulation_score(factor_scores: dict[str, dict[str, Any]]) -> int:
    return int(round(sum(float(item.get("score", 0)) for item in factor_scores.values())))


def factor_payload(value: Any, score: int, weight: int, value_key: str = "value") -> dict[str, Any]:
    return {
        value_key: value,
        "score": score,
        "weight": weight,
    }
