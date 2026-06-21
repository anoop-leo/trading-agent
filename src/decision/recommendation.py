"""Shared PositionRecommendation / RiskDecision contract.

Every investor agent and the signal engine only ever propose a
PositionRecommendation. The live risk engine (src/risk/live_risk_engine.py)
is the only thing allowed to turn one into an approved size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ASSET_CLASSES = {"crypto", "equity"}
BUCKETS = {"core", "growth", "speculative"}
ACTIONS = {"buy", "sell", "hold", "trim"}
RISK_DECISION_STATUSES = {"approved", "adjusted", "blocked"}


@dataclass(frozen=True)
class PositionRecommendation:
    symbol: str
    asset_class: str
    bucket: str
    action: str
    conviction_score: float
    suggested_size_usd: float
    rationale: str
    source_agent: str

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"Unsupported asset_class {self.asset_class!r}.")
        if self.bucket not in BUCKETS:
            raise ValueError(f"Unsupported bucket {self.bucket!r}.")
        if self.action not in ACTIONS:
            raise ValueError(f"Unsupported action {self.action!r}.")
        if not 0.0 <= self.conviction_score <= 1.0:
            raise ValueError("conviction_score must be between 0 and 1.")
        if self.suggested_size_usd < 0:
            raise ValueError("suggested_size_usd must not be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PositionRecommendation":
        return cls(
            symbol=str(payload["symbol"]),
            asset_class=str(payload["asset_class"]),
            bucket=str(payload["bucket"]),
            action=str(payload["action"]),
            conviction_score=float(payload["conviction_score"]),
            suggested_size_usd=float(payload["suggested_size_usd"]),
            rationale=str(payload["rationale"]),
            source_agent=str(payload["source_agent"]),
        )


@dataclass(frozen=True)
class RiskDecision:
    recommendation: PositionRecommendation
    status: str
    approved_size_usd: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in RISK_DECISION_STATUSES:
            raise ValueError(f"Unsupported status {self.status!r}.")
        if self.status in {"adjusted", "blocked"} and not self.reason:
            raise ValueError("reason is required when status is 'adjusted' or 'blocked'.")
        if self.approved_size_usd < 0:
            raise ValueError("approved_size_usd must not be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation.to_dict(),
            "status": self.status,
            "approved_size_usd": round(self.approved_size_usd, 2),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RiskDecision":
        return cls(
            recommendation=PositionRecommendation.from_dict(payload["recommendation"]),
            status=str(payload["status"]),
            approved_size_usd=float(payload["approved_size_usd"]),
            reason=str(payload.get("reason", "")),
        )
