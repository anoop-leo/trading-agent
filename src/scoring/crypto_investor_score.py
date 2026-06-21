"""Deterministic non-BTC crypto investor scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CRYPTO_INVESTOR_WEIGHTS = {
    "distance_from_200d_ma": 20,
    "weekly_rsi": 15,
    "monthly_trend": 15,
    "drawdown_from_cycle_high": 15,
    "volume_trend": 10,
    "volatility_risk": 10,
    "liquidity_proxy": 10,
    "thesis_risk_check": 5,
}


@dataclass(frozen=True)
class CryptoInvestorBand:
    investor_band: str
    accumulation_bias: str
    final_investor_action: str
    suggested_dca_multiplier: str


def score_distance_from_200d_ma(distance_pct: float | None) -> int:
    if distance_pct is None:
        return 0
    if distance_pct <= -30:
        return 20
    if distance_pct <= -15:
        return 16
    if -15 < distance_pct <= 10:
        return 10
    if distance_pct <= 30:
        return 5
    return 0


def score_weekly_rsi(value: float | None) -> int:
    if value is None:
        return 0
    if value < 30:
        return 15
    if value <= 40:
        return 13
    if value <= 50:
        return 9
    if value <= 60:
        return 7
    if value <= 70:
        return 3
    return 0


def score_monthly_trend(value: str | None) -> int:
    normalized = (value or "").upper()
    if normalized == "BELOW_EMA20":
        return 15
    if normalized == "NEAR_EMA20":
        return 10
    if normalized == "ABOVE_EMA20_EMA10_BELOW_EMA20":
        return 8
    if normalized == "BULLISH_ABOVE_EMA20":
        return 8
    if normalized == "EXTREMELY_EXTENDED":
        return 3
    return 0


def score_drawdown_from_cycle_high(drawdown_pct: float | None) -> int:
    if drawdown_pct is None:
        return 0
    if drawdown_pct <= -80:
        return 15
    if drawdown_pct <= -60:
        return 13
    if drawdown_pct <= -40:
        return 9
    if drawdown_pct <= -20:
        return 5
    return 2


def score_volume_trend(volume_ratio: float | None) -> int:
    if volume_ratio is None:
        return 0
    if volume_ratio >= 1.5:
        return 10
    if volume_ratio >= 1.0:
        return 7
    if volume_ratio >= 0.7:
        return 5
    return 2


def score_volatility_risk(atr_pct: float | None) -> int:
    if atr_pct is None:
        return 0
    if atr_pct < 3:
        return 10
    if atr_pct <= 5:
        return 8
    if atr_pct <= 8:
        return 4
    if atr_pct <= 12:
        return 2
    return 0


def score_liquidity_proxy(quote_volume_usd: float | None) -> int:
    if quote_volume_usd is None:
        return 0
    if quote_volume_usd > 500_000_000:
        return 10
    if quote_volume_usd >= 100_000_000:
        return 9
    if quote_volume_usd >= 25_000_000:
        return 8
    if quote_volume_usd >= 10_000_000:
        return 7
    if quote_volume_usd >= 1_000_000:
        return 4
    return 1


def score_thesis_risk(level: str | None) -> int:
    normalized = (level or "MODERATE").upper()
    if normalized == "LOW":
        return 5
    if normalized == "MODERATE":
        return 3
    if normalized == "HIGH":
        return 0
    return 3


def band_for_crypto_investor_score(score: int, thesis_risk_level: str = "MODERATE") -> CryptoInvestorBand:
    if score >= 85:
        band = CryptoInvestorBand(
            "STRONG_ACCUMULATION_ZONE",
            "HIGH",
            "ACCUMULATE_OPPORTUNISTICALLY",
            "1.25x to 2.0x normal DCA",
        )
    elif score >= 70:
        band = CryptoInvestorBand(
            "ACCUMULATION_ZONE",
            "MEDIUM_HIGH",
            "ACCUMULATE_SLOWLY",
            "1.0x to 1.25x normal DCA",
        )
    elif score >= 55:
        band = CryptoInvestorBand(
            "NEUTRAL_WATCH_ZONE",
            "MEDIUM",
            "NORMAL_DCA_ONLY",
            "0.75x to 1.0x normal DCA",
        )
    elif score >= 40:
        band = CryptoInvestorBand(
            "WEAK_ACCUMULATION_ZONE",
            "LOW",
            "WAIT_FOR_BETTER_PRICE_OR_CONFIRMATION",
            "0.25x to 0.75x normal DCA",
        )
    else:
        band = CryptoInvestorBand(
            "AVOID_ZONE",
            "VERY_LOW",
            "DO_NOT_ACCUMULATE",
            "0x normal DCA",
        )

    if thesis_risk_level.upper() == "HIGH" and band.investor_band == "STRONG_ACCUMULATION_ZONE":
        return CryptoInvestorBand(
            "ACCUMULATION_ZONE",
            "MEDIUM_HIGH",
            "ACCUMULATE_SLOWLY",
            "1.0x to 1.25x normal DCA",
        )
    return band


def calculate_crypto_investor_score(factor_scores: dict[str, dict[str, Any]]) -> int:
    return int(round(sum(float(item.get("score", 0)) for item in factor_scores.values())))


def factor_payload(value: Any, score: int, weight: int, value_key: str = "value") -> dict[str, Any]:
    return {
        value_key: value,
        "score": score,
        "weight": weight,
    }
