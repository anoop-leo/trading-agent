"""Strategy parameter profiles for deterministic backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    min_rr_ratio: float
    min_volume_ratio: float
    require_4h_macd_bullish: bool
    require_price_above_1h_ema20: bool
    allocation_per_trade: float
    require_alignment: bool = True
    require_rr_ratio: bool = True
    require_volume_ratio: bool = True
    enable_bull_market_mode: bool = False
    bull_min_rr_ratio: float = 1.2
    bull_min_volume_ratio: float = 0.5
    bull_allow_pullback_alignment: bool = True

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


PROFILES: dict[str, StrategyProfile] = {
    "conservative": StrategyProfile(
        name="conservative",
        min_rr_ratio=2.5,
        min_volume_ratio=1.2,
        require_4h_macd_bullish=True,
        require_price_above_1h_ema20=True,
        allocation_per_trade=0.20,
    ),
    "balanced": StrategyProfile(
        name="balanced",
        min_rr_ratio=2.0,
        min_volume_ratio=1.0,
        require_4h_macd_bullish=True,
        require_price_above_1h_ema20=True,
        allocation_per_trade=0.25,
    ),
    "aggressive": StrategyProfile(
        name="aggressive",
        min_rr_ratio=1.5,
        min_volume_ratio=0.8,
        require_4h_macd_bullish=False,
        require_price_above_1h_ema20=False,
        allocation_per_trade=0.30,
    ),
}

PROFILE_NAMES = tuple(PROFILES.keys())


class StrategyProfileError(ValueError):
    """Raised when a requested strategy profile is not supported."""


def get_strategy_profile(name: str) -> StrategyProfile:
    normalized = name.lower()
    if normalized not in PROFILES:
        supported = ", ".join(PROFILE_NAMES)
        raise StrategyProfileError(f"Unsupported strategy profile {name!r}. Supported profiles: {supported}.")
    return PROFILES[normalized]
