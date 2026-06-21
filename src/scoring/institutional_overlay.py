"""Institutional portfolio overlay scoring for BTC Investor Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RISK_PROFILE_DEFAULTS = {
    "CONSERVATIVE": {"target_btc_allocation_pct": 1.0, "max_btc_allocation_pct": 2.0},
    "BALANCED": {"target_btc_allocation_pct": 2.0, "max_btc_allocation_pct": 5.0},
    "AGGRESSIVE": {"target_btc_allocation_pct": 5.0, "max_btc_allocation_pct": 10.0},
    "HIGH_CONVICTION_CRYPTO": {"target_btc_allocation_pct": 10.0, "max_btc_allocation_pct": 20.0},
}

TREND_SCORES = {
    "VERY_STRONG": 100,
    "STRONG": 85,
    "IMPROVING": 70,
    "POSITIVE": 70,
    "NEUTRAL": 50,
    "MIXED": 50,
    "WEAKENING": 30,
    "NEGATIVE": 25,
    "VERY_WEAK": 0,
}


@dataclass(frozen=True)
class InstitutionalOverlayInput:
    portfolio_risk_profile: str = "BALANCED"
    current_btc_allocation_pct: float | None = None
    target_btc_allocation_pct: float | None = None
    max_btc_allocation_pct: float | None = None
    rebalance_threshold_pct: float = 25.0
    store_of_value_inputs: dict[str, str | None] | None = None
    network_adoption_inputs: dict[str, str | None] | None = None


@dataclass(frozen=True)
class InstitutionalOverlay:
    portfolio_risk_profile: str
    current_btc_allocation_pct: float | None
    target_btc_allocation_pct: float
    max_btc_allocation_pct: float
    portfolio_risk_budget_score: int
    store_of_value_thesis_score: int
    portfolio_discipline_score: int
    network_adoption_score: int
    institutional_score: int
    rebalance_signal: str
    allocation_guidance: str
    institutional_rationale: list[str]
    fallback_fields: list[str]
    dca_cap_multiplier: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "portfolio_risk_profile": self.portfolio_risk_profile,
            "current_btc_allocation_pct": self.current_btc_allocation_pct,
            "target_btc_allocation_pct": self.target_btc_allocation_pct,
            "max_btc_allocation_pct": self.max_btc_allocation_pct,
            "portfolio_risk_budget_score": self.portfolio_risk_budget_score,
            "store_of_value_thesis_score": self.store_of_value_thesis_score,
            "portfolio_discipline_score": self.portfolio_discipline_score,
            "network_adoption_score": self.network_adoption_score,
            "institutional_score": self.institutional_score,
            "rebalance_signal": self.rebalance_signal,
            "allocation_guidance": self.allocation_guidance,
            "institutional_rationale": self.institutional_rationale,
        }


def calculate_institutional_overlay(inputs: InstitutionalOverlayInput) -> InstitutionalOverlay:
    profile = inputs.portfolio_risk_profile.upper()
    if profile not in RISK_PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported portfolio risk profile: {inputs.portfolio_risk_profile}")

    target_pct = _resolve_allocation_value(inputs.target_btc_allocation_pct, profile, "target_btc_allocation_pct")
    max_pct = _resolve_allocation_value(inputs.max_btc_allocation_pct, profile, "max_btc_allocation_pct")
    if max_pct < target_pct:
        raise ValueError("max_btc_allocation_pct must be greater than or equal to target_btc_allocation_pct.")

    risk_budget_score, rebalance_signal, risk_rationale, dca_cap = score_portfolio_risk_budget(
        current_btc_allocation_pct=inputs.current_btc_allocation_pct,
        target_btc_allocation_pct=target_pct,
        max_btc_allocation_pct=max_pct,
        rebalance_threshold_pct=inputs.rebalance_threshold_pct,
    )
    discipline_score = score_portfolio_discipline(
        current_btc_allocation_pct=inputs.current_btc_allocation_pct,
        target_btc_allocation_pct=target_pct,
        max_btc_allocation_pct=max_pct,
        rebalance_threshold_pct=inputs.rebalance_threshold_pct,
    )
    store_score, store_fallbacks, store_rationale = score_store_of_value_thesis(inputs.store_of_value_inputs)
    adoption_score, adoption_fallbacks, adoption_rationale = score_network_adoption(inputs.network_adoption_inputs)
    institutional_score = int(
        round(
            (risk_budget_score * 0.35)
            + (store_score * 0.25)
            + (discipline_score * 0.25)
            + (adoption_score * 0.15)
        )
    )
    rationale = [
        *risk_rationale,
        *store_rationale,
        *adoption_rationale,
        f"Institutional overlay score is {institutional_score}.",
    ]
    return InstitutionalOverlay(
        portfolio_risk_profile=profile,
        current_btc_allocation_pct=inputs.current_btc_allocation_pct,
        target_btc_allocation_pct=target_pct,
        max_btc_allocation_pct=max_pct,
        portfolio_risk_budget_score=risk_budget_score,
        store_of_value_thesis_score=store_score,
        portfolio_discipline_score=discipline_score,
        network_adoption_score=adoption_score,
        institutional_score=institutional_score,
        rebalance_signal=rebalance_signal,
        allocation_guidance=_allocation_guidance(rebalance_signal),
        institutional_rationale=rationale,
        fallback_fields=sorted(set(store_fallbacks + adoption_fallbacks)),
        dca_cap_multiplier=dca_cap,
    )


def score_portfolio_risk_budget(
    current_btc_allocation_pct: float | None,
    target_btc_allocation_pct: float,
    max_btc_allocation_pct: float,
    rebalance_threshold_pct: float = 25.0,
) -> tuple[int, str, list[str], str | None]:
    del rebalance_threshold_pct
    if current_btc_allocation_pct is None:
        return (
            50,
            "UNKNOWN_ALLOCATION",
            ["Current BTC allocation is unknown; allocation guidance is limited."],
            None,
        )
    if current_btc_allocation_pct >= max_btc_allocation_pct:
        return (
            0,
            "OVER_ALLOCATED",
            ["BTC allocation is above the max risk budget."],
            "0.0x to 0.5x normal DCA",
        )
    if current_btc_allocation_pct > target_btc_allocation_pct:
        return (
            40,
            _allocation_signal(current_btc_allocation_pct, target_btc_allocation_pct, max_btc_allocation_pct),
            ["BTC allocation is above target, so extra accumulation is capped."],
            "0.5x to 1.0x normal DCA",
        )
    return (
        100,
        _allocation_signal(current_btc_allocation_pct, target_btc_allocation_pct, max_btc_allocation_pct),
        ["BTC allocation is below target, so calculated accumulation guidance can be used."],
        None,
    )


def score_portfolio_discipline(
    current_btc_allocation_pct: float | None,
    target_btc_allocation_pct: float,
    max_btc_allocation_pct: float,
    rebalance_threshold_pct: float = 25.0,
) -> int:
    if current_btc_allocation_pct is None:
        return 50
    if current_btc_allocation_pct >= max_btc_allocation_pct:
        return 0

    threshold = rebalance_threshold_pct / 100.0
    lower_bound = target_btc_allocation_pct * (1.0 - threshold)
    upper_bound = target_btc_allocation_pct * (1.0 + threshold)
    if lower_bound <= current_btc_allocation_pct <= upper_bound:
        return 75
    if current_btc_allocation_pct < lower_bound:
        return 100
    return 40


def score_store_of_value_thesis(inputs: dict[str, str | None] | None = None) -> tuple[int, list[str], list[str]]:
    if not inputs:
        return 50, ["store_of_value_thesis"], ["Store-of-value thesis data is unavailable; neutral fallback score used."]

    factors = {
        "btc_supply_scarcity": inputs.get("btc_supply_scarcity", "STRONG"),
        "etf_flow_trend": inputs.get("etf_flow_trend"),
        "long_term_holder_supply_trend": inputs.get("long_term_holder_supply_trend"),
        "exchange_reserve_trend": inputs.get("exchange_reserve_trend"),
        "macro_debasement_proxy": inputs.get("macro_debasement_proxy"),
    }
    optional_values = [value for key, value in factors.items() if key != "btc_supply_scarcity"]
    if all(value is None for value in optional_values):
        return 50, ["store_of_value_thesis"], ["Store-of-value thesis data is unavailable; neutral fallback score used."]
    return _average_factor_score(factors), [], ["Store-of-value thesis score reflects available institutional inputs."]


def score_network_adoption(inputs: dict[str, str | None] | None = None) -> tuple[int, list[str], list[str]]:
    if not inputs or all(value is None for value in inputs.values()):
        return 50, ["network_adoption"], ["Network adoption data is unavailable; neutral fallback score used."]

    factors = {
        "active_address_trend": inputs.get("active_address_trend"),
        "transaction_fee_trend": inputs.get("transaction_fee_trend"),
        "hashrate_trend": inputs.get("hashrate_trend"),
        "etf_adoption_trend": inputs.get("etf_adoption_trend"),
    }
    return _average_factor_score(factors), [], ["Network adoption score reflects available adoption inputs."]


def _allocation_signal(current_pct: float, target_pct: float, max_pct: float, rebalance_threshold_pct: float = 25.0) -> str:
    if current_pct >= max_pct:
        return "OVER_ALLOCATED"
    lower_bound = target_pct * (1.0 - (rebalance_threshold_pct / 100.0))
    upper_bound = target_pct * (1.0 + (rebalance_threshold_pct / 100.0))
    if lower_bound <= current_pct <= upper_bound:
        return "NEAR_TARGET"
    if current_pct < lower_bound:
        return "BELOW_TARGET"
    return "ABOVE_TARGET"


def _allocation_guidance(rebalance_signal: str) -> str:
    if rebalance_signal == "UNKNOWN_ALLOCATION":
        return "Current BTC allocation is unknown; only base DCA is allowed until allocation is provided."
    if rebalance_signal == "BELOW_TARGET":
        return "BTC is below target allocation; calculated accumulation guidance can be used."
    if rebalance_signal == "NEAR_TARGET":
        return "BTC is near target allocation; maintain disciplined policy DCA."
    if rebalance_signal == "ABOVE_TARGET":
        return "BTC is above target allocation; extra accumulation is capped."
    if rebalance_signal == "OVER_ALLOCATED":
        return "BTC is above max allocation; pause extra accumulation and review rebalancing."
    return ""


def _average_factor_score(factors: dict[str, str | None]) -> int:
    scores = [_score_trend(value) for value in factors.values() if value is not None]
    if not scores:
        return 50
    return int(round(sum(scores) / len(scores)))


def _score_trend(value: str | None) -> int:
    if value is None:
        return 50
    return TREND_SCORES.get(value.upper(), 50)


def _resolve_allocation_value(value: float | None, profile: str, key: str) -> float:
    if value is not None:
        return float(value)
    return RISK_PROFILE_DEFAULTS[profile][key]
