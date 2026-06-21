"""Live risk engine configuration.

Bucket targets and caps are config, not constants: edit config/risk_config.json
rather than these dataclass defaults. The defaults here only apply when no
config file is present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RISK_CONFIG_PATH = Path("config/risk_config.json")


@dataclass(frozen=True)
class BucketTargets:
    core_pct: float = 45.0
    growth_pct: float = 30.0
    speculative_pct: float = 12.0
    cash_pct: float = 13.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RiskEngineConfig:
    total_portfolio_value_usd: float = 330_000.0
    bucket_targets: BucketTargets = field(default_factory=BucketTargets)
    speculative_max_pct: float = 12.0
    cash_buffer_min_pct: float = 10.0
    growth_position_max_pct: float = 5.0
    speculative_position_max_pct: float = 2.0
    portfolio_drawdown_circuit_breaker_pct: float = 25.0
    portfolio_drawdown_recovery_pct: float = 15.0
    equity_gap_buffer_pct: float = 5.0
    growth_default_position_usd: float = 5_000.0
    speculative_default_position_usd: float = 1_500.0
    core_default_position_usd: float = 5_000.0
    risk_per_trade_pct: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bucket_targets"] = self.bucket_targets.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RiskEngineConfig":
        bucket_payload = payload.get("bucket_targets", {})
        kwargs = {key: value for key, value in payload.items() if key != "bucket_targets"}
        return cls(bucket_targets=BucketTargets(**bucket_payload), **kwargs)


def load_risk_config(path: Path = DEFAULT_RISK_CONFIG_PATH) -> RiskEngineConfig:
    """Load risk config from JSON, falling back to defaults if absent."""

    path = Path(path)
    if not path.exists():
        return RiskEngineConfig()
    payload = json.loads(path.read_text())
    return RiskEngineConfig.from_dict(payload)


def write_default_risk_config(path: Path = DEFAULT_RISK_CONFIG_PATH) -> Path:
    """Write the default risk config to disk if it doesn't already exist."""

    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(RiskEngineConfig().to_dict(), indent=2, sort_keys=True) + "\n")
    return path
