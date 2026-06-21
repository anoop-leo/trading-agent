"""Margin-of-safety scoring for long-term BTC accumulation."""

from __future__ import annotations


MARGIN_OF_SAFETY_FACTORS = (
    "distance_from_200d_ma",
    "mvrv",
    "drawdown_from_cycle_high",
    "fear_and_greed",
)


def calculate_margin_of_safety_score(factor_scores: dict[str, dict[str, object]]) -> int:
    score = 0.0
    max_score = 0.0
    for factor in MARGIN_OF_SAFETY_FACTORS:
        payload = factor_scores[factor]
        score += float(payload["score"])
        max_score += float(payload["weight"])
    if max_score <= 0:
        return 0
    return int(round((score / max_score) * 100))


def margin_of_safety_band(score: float) -> str:
    if score < 30:
        return "NO_MARGIN_OF_SAFETY"
    if score < 60:
        return "SOME_MARGIN_OF_SAFETY"
    if score < 80:
        return "GOOD_MARGIN_OF_SAFETY"
    return "RARE_MARGIN_OF_SAFETY"
