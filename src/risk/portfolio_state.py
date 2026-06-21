"""Current portfolio composition for the live risk engine.

No broker integration exists yet, so this state is a hand-maintained snapshot
(persisted to data/portfolio_state.json by default) rather than a live feed.
Update the file as your real allocations change; the risk engine reads
whatever is on disk at evaluation time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from risk.portfolio_risk_governor import calculate_drawdown_pct
from risk.risk_config import RiskEngineConfig


DEFAULT_PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")


@dataclass(frozen=True)
class PortfolioState:
    total_value_usd: float
    peak_value_usd: float
    cash_usd: float
    core_usd: float
    growth_usd: float
    speculative_usd: float

    def __post_init__(self) -> None:
        if self.total_value_usd < 0 or self.peak_value_usd < 0:
            raise ValueError("Portfolio values must not be negative.")

    @property
    def drawdown_pct(self) -> float:
        return calculate_drawdown_pct(self.total_value_usd, self.peak_value_usd)

    def bucket_usd(self, bucket: str) -> float:
        mapping = {"core": self.core_usd, "growth": self.growth_usd, "speculative": self.speculative_usd}
        if bucket not in mapping:
            raise ValueError(f"Unsupported bucket {bucket!r}.")
        return mapping[bucket]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioState":
        return cls(
            total_value_usd=float(payload["total_value_usd"]),
            peak_value_usd=float(payload["peak_value_usd"]),
            cash_usd=float(payload["cash_usd"]),
            core_usd=float(payload["core_usd"]),
            growth_usd=float(payload["growth_usd"]),
            speculative_usd=float(payload["speculative_usd"]),
        )

    @classmethod
    def from_config_targets(cls, config: RiskEngineConfig) -> "PortfolioState":
        total = config.total_portfolio_value_usd
        targets = config.bucket_targets
        return cls(
            total_value_usd=total,
            peak_value_usd=total,
            cash_usd=total * targets.cash_pct / 100,
            core_usd=total * targets.core_pct / 100,
            growth_usd=total * targets.growth_pct / 100,
            speculative_usd=total * targets.speculative_pct / 100,
        )


def load_portfolio_state(
    path: Path = DEFAULT_PORTFOLIO_STATE_PATH,
    config: RiskEngineConfig | None = None,
) -> PortfolioState:
    """Load portfolio state from disk, defaulting to config bucket targets."""

    path = Path(path)
    if path.exists():
        return PortfolioState.from_dict(json.loads(path.read_text()))
    return PortfolioState.from_config_targets(config or RiskEngineConfig())


def save_portfolio_state(state: PortfolioState, path: Path = DEFAULT_PORTFOLIO_STATE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def apply_decision_to_state(state: PortfolioState, decision_dict: dict[str, Any]) -> PortfolioState:
    """Return new state reflecting an approved/adjusted buy or a sell/trim being filled.

    This only updates bucket and cash balances; total_value_usd / peak_value_usd
    should be refreshed separately from real holdings marks.
    """

    bucket = decision_dict["recommendation"]["bucket"]
    action = decision_dict["recommendation"]["action"]
    size = float(decision_dict["approved_size_usd"])
    if decision_dict["status"] == "blocked" or size <= 0:
        return state

    bucket_values = {"core": state.core_usd, "growth": state.growth_usd, "speculative": state.speculative_usd}
    if action == "buy":
        bucket_values[bucket] += size
        cash_usd = state.cash_usd - size
    else:
        bucket_values[bucket] = max(0.0, bucket_values[bucket] - size)
        cash_usd = state.cash_usd + size

    return PortfolioState(
        total_value_usd=state.total_value_usd,
        peak_value_usd=state.peak_value_usd,
        cash_usd=cash_usd,
        core_usd=bucket_values["core"],
        growth_usd=bucket_values["growth"],
        speculative_usd=bucket_values["speculative"],
    )
