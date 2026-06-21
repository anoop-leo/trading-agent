"""Deterministic equity investor scoring (growth-bucket individual stocks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EQUITY_INVESTOR_WEIGHTS = {
    "valuation_pe": 20,
    "valuation_peg": 15,
    "valuation_pb": 10,
    "fcf_yield": 20,
    "quality_roe": 10,
    "growth_consistency": 10,
    "distance_from_200d_ma": 15,
}


@dataclass(frozen=True)
class EquityInvestorBand:
    investor_band: str
    accumulation_bias: str
    final_investor_action: str
    suggested_dca_multiplier: str


def score_valuation_pe(pe: float | None) -> int:
    if pe is None or pe <= 0:
        return 0
    if pe < 12:
        return 20
    if pe < 18:
        return 16
    if pe < 25:
        return 10
    if pe < 35:
        return 5
    return 0


def score_valuation_peg(peg: float | None) -> int:
    if peg is None or peg <= 0:
        return 0
    if peg < 1:
        return 15
    if peg < 1.5:
        return 11
    if peg < 2:
        return 6
    return 2


def score_valuation_pb(price_to_book: float | None) -> int:
    if price_to_book is None or price_to_book <= 0:
        return 0
    if price_to_book < 1:
        return 10
    if price_to_book < 3:
        return 7
    if price_to_book < 6:
        return 4
    return 1


def score_fcf_yield(fcf_yield_pct: float | None) -> int:
    if fcf_yield_pct is None:
        return 0
    if fcf_yield_pct < 0:
        return 0
    if fcf_yield_pct >= 8:
        return 20
    if fcf_yield_pct >= 5:
        return 16
    if fcf_yield_pct >= 3:
        return 10
    if fcf_yield_pct >= 1:
        return 5
    return 2


def score_quality_roe(return_on_equity: float | None) -> int:
    if return_on_equity is None:
        return 0
    if return_on_equity < 0:
        return 0
    if return_on_equity >= 0.20:
        return 10
    if return_on_equity >= 0.12:
        return 7
    if return_on_equity >= 0.05:
        return 4
    return 2


def score_growth_consistency(revenue_growth_yoy: float | None, earnings_growth_yoy: float | None) -> int:
    values = [value for value in (revenue_growth_yoy, earnings_growth_yoy) if value is not None]
    if not values:
        return 0
    average = sum(values) / len(values)
    if average >= 0.15:
        return 10
    if average >= 0.05:
        return 7
    if average >= 0:
        return 4
    return 0


def score_distance_from_200d_ma(distance_pct: float | None) -> int:
    if distance_pct is None:
        return 0
    if distance_pct <= -30:
        return 15
    if distance_pct <= -15:
        return 12
    if -15 < distance_pct <= 10:
        return 8
    if distance_pct <= 30:
        return 4
    return 0


def band_for_equity_investor_score(score: int) -> EquityInvestorBand:
    if score >= 85:
        return EquityInvestorBand(
            "STRONG_ACCUMULATION_ZONE", "HIGH", "ACCUMULATE_OPPORTUNISTICALLY", "1.25x to 2.0x normal DCA"
        )
    if score >= 70:
        return EquityInvestorBand(
            "ACCUMULATION_ZONE", "MEDIUM_HIGH", "ACCUMULATE_SLOWLY", "1.0x to 1.25x normal DCA"
        )
    if score >= 55:
        return EquityInvestorBand(
            "NEUTRAL_WATCH_ZONE", "MEDIUM", "NORMAL_DCA_ONLY", "0.75x to 1.0x normal DCA"
        )
    if score >= 40:
        return EquityInvestorBand(
            "WEAK_ACCUMULATION_ZONE", "LOW", "WAIT_FOR_BETTER_PRICE_OR_CONFIRMATION", "0.25x to 0.75x normal DCA"
        )
    return EquityInvestorBand("AVOID_ZONE", "VERY_LOW", "DO_NOT_ACCUMULATE", "0x normal DCA")


def calculate_equity_investor_score(factor_scores: dict[str, dict[str, Any]]) -> int:
    return int(round(sum(float(item.get("score", 0)) for item in factor_scores.values())))


def factor_payload(value: Any, score: int, weight: int, value_key: str = "value") -> dict[str, Any]:
    return {
        value_key: value,
        "score": score,
        "weight": weight,
    }
