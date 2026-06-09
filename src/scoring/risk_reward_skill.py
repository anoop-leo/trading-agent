"""Risk/reward scoring skill."""

from __future__ import annotations

from dataclasses import dataclass

class RiskRewardError(ValueError):
    """Raised when risk/reward cannot be calculated."""


@dataclass(frozen=True)
class RiskRewardResult:
    risk: float
    reward: float
    rr_ratio: float
    rr_score: int


def _score_rr_ratio(rr_ratio: float) -> int:
    if rr_ratio < 1.0:
        return 0
    if rr_ratio < 2.0:
        return 4
    if rr_ratio < 3.0:
        return 7
    return 10


def calculate_risk_reward(current_price: float, support: float, resistance: float) -> RiskRewardResult:
    """Evaluate upside versus downside using entry=current, stop=support, target=resistance."""

    if current_price <= 0 or support <= 0 or resistance <= 0:
        raise RiskRewardError("current_price, support, and resistance must be greater than zero.")

    risk = max(0.0, current_price - support)
    reward = max(0.0, resistance - current_price)
    rr_ratio = 10.0 if risk == 0 and reward > 0 else (0.0 if risk == 0 else reward / risk)

    return RiskRewardResult(
        risk=risk,
        reward=reward,
        rr_ratio=rr_ratio,
        rr_score=_score_rr_ratio(rr_ratio),
    )
