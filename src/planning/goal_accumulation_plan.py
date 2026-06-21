"""Goal-based BTC accumulation planning for the Investor Agent."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


TAX_WARNING = "Large BTC sales may create taxable capital gains. Consult a tax professional before executing any sale."


@dataclass(frozen=True)
class GoalAccumulationInput:
    current_btc: float = 1.13059494
    target_btc: float = 2.0
    current_price: float | None = None
    monthly_dca_usd: float | None = None
    lump_sum_available_usd: float | None = None
    target_sell_price: float = 500000.0
    planned_sell_btc: float = 1.0
    retain_btc: float = 1.0
    accumulation_score: int = 0
    accumulation_band: str = "EXPENSIVE"
    margin_of_safety_score: int = 0
    mvrv_value: float | None = None
    mvrv_missing: bool = True
    fear_and_greed_value: float | None = None
    thesis_risk_level: str = "LOW"
    final_investor_action: str = "INSUFFICIENT_DATA"
    rebalance_signal: str = "UNKNOWN_ALLOCATION"
    current_btc_allocation_pct: float | None = None
    target_btc_allocation_pct: float | None = None
    max_btc_allocation_pct: float | None = None
    institutional_score: int = 0
    reference_price_for_dip: float | None = None
    distance_from_200d_ma_pct: float | None = None
    drawdown_from_cycle_high_pct: float | None = None


def build_goal_accumulation_plan(inputs: GoalAccumulationInput) -> dict[str, Any]:
    remaining_btc = _max_decimal(_btc(inputs.target_btc) - _btc(inputs.current_btc), Decimal("0"))
    bucket_amounts = _bucket_amounts(remaining_btc)
    goal_rationale = _goal_rationale(inputs)
    if inputs.monthly_dca_usd is None or inputs.monthly_dca_usd <= 0:
        goal_rationale.append("Monthly DCA amount not provided.")

    current_price = _optional_money(inputs.current_price)
    usd_needed = None
    if current_price is not None:
        usd_needed = _money_float(remaining_btc * Decimal(str(current_price)))

    estimated_months = None
    if usd_needed is not None and inputs.monthly_dca_usd is not None and inputs.monthly_dca_usd > 0:
        estimated_months = round(usd_needed / inputs.monthly_dca_usd, 2)

    planned_sell_btc = _btc(inputs.planned_sell_btc)
    target_sell_price = _money(inputs.target_sell_price)
    dip_context = _dip_context(inputs)
    allocation_gate = _allocation_gate(inputs)
    return {
        "current_btc": _btc_float(_btc(inputs.current_btc)),
        "target_btc": _btc_float(_btc(inputs.target_btc)),
        "remaining_btc_needed": _btc_float(remaining_btc),
        "current_price": current_price,
        "estimated_usd_needed_at_current_price": usd_needed,
        "monthly_dca_usd": _optional_money(inputs.monthly_dca_usd),
        "lump_sum_available_usd": _optional_money(inputs.lump_sum_available_usd),
        "estimated_months_to_target": estimated_months,
        "target_sell_price": _money_float(target_sell_price),
        "planned_sell_btc": _btc_float(planned_sell_btc),
        "retain_btc": _btc_float(_btc(inputs.retain_btc)),
        "estimated_sale_proceeds": _money_float(target_sell_price * planned_sell_btc),
        "accumulation_mode": "PATIENT_ACCUMULATION",
        "allocation_gate": allocation_gate,
        "reference_price_for_dip": _optional_money(inputs.reference_price_for_dip),
        "current_drawdown_from_reference_pct": dip_context["current_drawdown_from_reference_pct"],
        "dip_trigger_met": dip_context["dip_trigger_met"],
        "tranche_plan": _tranche_plan(inputs, bucket_amounts),
        "lump_sum_plan": _lump_sum_plan(inputs, remaining_btc),
        "sell_plan": _sell_plan(target_sell_price, planned_sell_btc),
        "tax_warning": TAX_WARNING,
        "goal_rationale": goal_rationale,
    }


def _tranche_plan(inputs: GoalAccumulationInput, bucket_amounts: dict[str, Decimal]) -> list[dict[str, Any]]:
    base = _base_bucket(inputs, bucket_amounts["base_dca"])
    opportunistic = _opportunistic_bucket(inputs, bucket_amounts["opportunistic"])
    dip_reserve = _dip_reserve_bucket(inputs, bucket_amounts["dip_reserve"])
    deep_value = _deep_value_bucket(inputs, bucket_amounts["deep_value_reserve"])
    return [base, opportunistic, dip_reserve, deep_value]


def _base_bucket(inputs: GoalAccumulationInput, target_btc: Decimal) -> dict[str, Any]:
    status = "ACTIVE"
    allowed = True
    reason = "Small base DCA is allowed for patient accumulation."
    if _is_high_thesis_risk(inputs) or inputs.rebalance_signal == "OVER_ALLOCATED":
        status = "PAUSED"
        allowed = False
        reason = "Base DCA is paused because thesis risk or allocation risk is elevated."
    elif inputs.accumulation_score < 30:
        status = "LOCKED"
        allowed = False
        reason = "Base DCA requires an accumulation score of FAIR or better."
    elif inputs.final_investor_action == "INSUFFICIENT_DATA":
        reason = "Insufficient data; only small base DCA is allowed."

    return _bucket(
        "BASE_DCA",
        10,
        target_btc,
        status,
        allowed,
        [
            "Accumulation score is FAIR or better",
            "Thesis risk is not HIGH",
        ],
        reason,
    )


def _opportunistic_bucket(inputs: GoalAccumulationInput, target_btc: Decimal) -> dict[str, Any]:
    status = "LOCKED"
    allowed = False
    reason = "Reserved for strong Investor Agent signals."
    if _larger_buckets_blocked_by_allocation_gate(inputs):
        reason = "Current BTC allocation is unknown; opportunistic accumulation is locked until portfolio risk budget can be checked."
    elif _locks_non_base_buckets(inputs):
        reason = _non_base_lock_reason(inputs)
    elif _opportunistic_conditions_met(inputs):
        if inputs.accumulation_score < 80:
            status = "PARTIALLY_AVAILABLE"
            reason = "Opportunistic bucket is partially available; maximum deployment is 25% of this bucket at a time."
        else:
            status = "ACTIVE"
            reason = "Opportunistic bucket is available because investor signals and valuation confirmation are strong."
        allowed = True

    return _bucket(
        "OPPORTUNISTIC",
        35,
        target_btc,
        status,
        allowed,
        [
            "Accumulation score >= 75",
            "Margin of safety score >= 65",
            "MVRV is available and not expensive",
            "Fear & Greed <= 30",
            "Thesis risk is not HIGH",
            "BTC allocation is not above max portfolio risk budget",
        ],
        reason,
    )


def _dip_reserve_bucket(inputs: GoalAccumulationInput, target_btc: Decimal) -> dict[str, Any]:
    status = "LOCKED"
    allowed = False
    reason = "Dip reserve requires a reference price and confirmed drawdown trigger."
    dip_context = _dip_context(inputs)
    if _larger_buckets_blocked_by_allocation_gate(inputs):
        reason = "Current BTC allocation is unknown; dip reserve is locked until portfolio risk budget can be checked."
    elif _locks_non_base_buckets(inputs):
        reason = _non_base_lock_reason(inputs)
    elif dip_context["dip_trigger_met"]:
        status = "PARTIALLY_AVAILABLE"
        allowed = True
        reason = "Dip reserve is partially available because BTC has reached a staged decline trigger."

    bucket = _bucket(
        "DIP_RESERVE",
        35,
        target_btc,
        status,
        allowed,
        [
            "Deploy 25% of dip reserve if BTC falls 15%",
            "Deploy another 25% if BTC falls 25%",
            "Deploy another 25% if BTC falls 35%",
            "Keep final 25% unless Investor Score also rises above 80",
            "Thesis risk is not HIGH",
        ],
        reason,
    )
    bucket["reference_price_for_dip"] = _optional_money(inputs.reference_price_for_dip)
    bucket["current_drawdown_from_reference_pct"] = dip_context["current_drawdown_from_reference_pct"]
    bucket["dip_trigger_met"] = dip_context["dip_trigger_met"]
    bucket["available_bucket_pct_now"] = dip_context["available_bucket_pct_now"] if allowed else 0
    return bucket


def _deep_value_bucket(inputs: GoalAccumulationInput, target_btc: Decimal) -> dict[str, Any]:
    status = "LOCKED"
    allowed = False
    reason = "Deep value reserve is locked until rare high-margin-of-safety conditions appear."
    if _larger_buckets_blocked_by_allocation_gate(inputs):
        reason = "Current BTC allocation is unknown; deep value accumulation is locked until portfolio risk budget can be checked."
    elif _locks_non_base_buckets(inputs):
        reason = _non_base_lock_reason(inputs)
    elif _deep_value_conditions_met(inputs):
        status = "ACTIVE"
        allowed = True
        reason = "Deep value reserve is available because valuation, sentiment, and margin of safety are unusually strong."

    return _bucket(
        "DEEP_VALUE_RESERVE",
        20,
        target_btc,
        status,
        allowed,
        [
            "Accumulation score >= 80",
            "Margin of safety score >= 70",
            "MVRV < 1.5 or clearly cheap",
            "Fear & Greed <= 25",
            "Price is at least 25% below 200D MA or drawdown from cycle high is greater than 50%",
            "Thesis risk is LOW or MEDIUM",
        ],
        reason,
    )


def _bucket(
    name: str,
    pct: int,
    target_btc: Decimal,
    status: str,
    allowed_now: bool,
    trigger_conditions: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "bucket_name": name,
        "bucket_pct": pct,
        "target_btc": _btc_float(target_btc),
        "status": status,
        "trigger_conditions": trigger_conditions,
        "allowed_now": allowed_now,
        "reason": reason,
    }


def _sell_plan(target_sell_price: Decimal, planned_sell_btc: Decimal) -> list[dict[str, Any]]:
    trigger_prices = [
        target_sell_price * Decimal("0.9"),
        target_sell_price,
        target_sell_price * Decimal("1.1"),
        target_sell_price * Decimal("1.2"),
    ]
    tranche_btc = _quantize_btc(planned_sell_btc / Decimal("4"))
    tranches = [tranche_btc, tranche_btc, tranche_btc]
    tranches.append(_quantize_btc(planned_sell_btc - sum(tranches, Decimal("0"))))
    return [
        {
            "trigger_price": _money_float(price),
            "sell_btc": _btc_float(amount),
            "estimated_proceeds": _money_float(price * amount),
            "purpose": "Future staged sale framework; not a current execution instruction.",
        }
        for price, amount in zip(trigger_prices, tranches)
    ]


def _lump_sum_plan(inputs: GoalAccumulationInput, remaining_btc: Decimal) -> dict[str, Any]:
    max_btc = Decimal("0")
    max_usd = None
    allowed = False
    reason = "Lump-sum accumulation remains locked until high-confidence valuation conditions appear."
    if not _allocation_known(inputs):
        reason = "Current BTC allocation is unknown; lump-sum planning is disabled until portfolio risk budget is known."
    elif inputs.institutional_score < 70:
        reason = "Institutional score is below 70, so lump-sum planning remains disabled."
    elif inputs.final_investor_action == "INSUFFICIENT_DATA":
        reason = "Insufficient data; do not recommend lump sum."
    elif inputs.mvrv_missing:
        reason = "MVRV is missing, so lump-sum accumulation remains locked."
    elif inputs.rebalance_signal == "OVER_ALLOCATED":
        reason = "BTC allocation is already above risk budget. Accumulation should pause or slow."
    elif _is_high_thesis_risk(inputs):
        reason = "High thesis risk detected; accumulation is paused until risk normalizes."
    elif (
        _allocation_below_target(inputs)
        and inputs.accumulation_score >= 80
        and inputs.margin_of_safety_score >= 70
        and _mvrv_cheap(inputs)
    ):
        allowed = True
        max_btc = _quantize_btc(remaining_btc * Decimal("0.25"))
        reason = "Partial lump-sum planning is limited to 25% of remaining BTC needed."
        if inputs.lump_sum_available_usd is not None and inputs.current_price is not None and inputs.current_price > 0:
            affordable_btc = _quantize_btc(_money(inputs.lump_sum_available_usd) / _money(inputs.current_price))
            max_btc = min(max_btc, affordable_btc)
    if inputs.current_price is not None:
        max_usd = _money_float(max_btc * _money(inputs.current_price))
    return {
        "allowed_now": allowed,
        "max_btc": _btc_float(max_btc),
        "max_usd_at_current_price": max_usd,
        "reason": reason,
    }


def _goal_rationale(inputs: GoalAccumulationInput) -> list[str]:
    rationale = [
        "User already owns a meaningful BTC position, so the plan favors patient accumulation.",
        "Most remaining BTC target is reserved for better valuation, dips, or stronger investor signals.",
    ]
    if inputs.final_investor_action == "INSUFFICIENT_DATA":
        rationale.append("Insufficient data; only small base DCA is allowed.")
    if not _allocation_known(inputs):
        rationale.append("Current BTC allocation is unknown; larger accumulation buckets remain locked.")
    if inputs.mvrv_missing:
        rationale.append("MVRV is missing, so larger accumulation buckets remain locked.")
    if inputs.rebalance_signal == "OVER_ALLOCATED":
        rationale.append("BTC allocation is already above risk budget. Accumulation should pause or slow.")
    if _is_high_thesis_risk(inputs):
        rationale.append("High thesis risk detected; accumulation is paused until risk normalizes.")
    return rationale


def _locks_non_base_buckets(inputs: GoalAccumulationInput) -> bool:
    return (
        inputs.final_investor_action == "INSUFFICIENT_DATA"
        or inputs.mvrv_missing
        or inputs.rebalance_signal == "OVER_ALLOCATED"
        or _is_high_thesis_risk(inputs)
    )


def _non_base_lock_reason(inputs: GoalAccumulationInput) -> str:
    if _is_high_thesis_risk(inputs):
        return "High thesis risk detected; accumulation is paused until risk normalizes."
    if inputs.rebalance_signal == "OVER_ALLOCATED":
        return "BTC allocation is already above risk budget. Accumulation should pause or slow."
    if inputs.final_investor_action == "INSUFFICIENT_DATA":
        return "Insufficient data; only small base DCA is allowed."
    if inputs.mvrv_missing:
        return "MVRV is missing, so larger accumulation buckets remain locked."
    return "Bucket remains locked."


def _opportunistic_conditions_met(inputs: GoalAccumulationInput) -> bool:
    return (
        _allocation_known_below_max(inputs)
        and inputs.accumulation_score >= 75
        and inputs.margin_of_safety_score >= 65
        and _mvrv_available_not_expensive(inputs)
        and _fear_and_greed_at_or_below(inputs, 30)
        and not _is_high_thesis_risk(inputs)
        and inputs.rebalance_signal != "OVER_ALLOCATED"
    )


def _deep_value_conditions_met(inputs: GoalAccumulationInput) -> bool:
    return (
        _allocation_known_below_max(inputs)
        and inputs.accumulation_score >= 80
        and inputs.margin_of_safety_score >= 70
        and _mvrv_cheap(inputs)
        and _fear_and_greed_at_or_below(inputs, 25)
        and _deep_value_price_condition(inputs)
        and inputs.thesis_risk_level.upper() in {"LOW", "MEDIUM"}
    )


def _dip_trigger_met(inputs: GoalAccumulationInput) -> bool:
    return bool(_dip_context(inputs)["dip_trigger_met"]) and not _is_high_thesis_risk(inputs)


def _deep_value_price_condition(inputs: GoalAccumulationInput) -> bool:
    below_ma = inputs.distance_from_200d_ma_pct is not None and inputs.distance_from_200d_ma_pct <= -25
    deep_drawdown = inputs.drawdown_from_cycle_high_pct is not None and inputs.drawdown_from_cycle_high_pct <= -50
    return below_ma or deep_drawdown


def _mvrv_available_not_expensive(inputs: GoalAccumulationInput) -> bool:
    return not inputs.mvrv_missing and inputs.mvrv_value is not None and inputs.mvrv_value <= 3.0


def _mvrv_cheap(inputs: GoalAccumulationInput) -> bool:
    return not inputs.mvrv_missing and inputs.mvrv_value is not None and inputs.mvrv_value < 1.5


def _fear_and_greed_at_or_below(inputs: GoalAccumulationInput, threshold: float) -> bool:
    return inputs.fear_and_greed_value is not None and inputs.fear_and_greed_value <= threshold


def _is_high_thesis_risk(inputs: GoalAccumulationInput) -> bool:
    return inputs.thesis_risk_level.upper() == "HIGH"


def _allocation_gate(inputs: GoalAccumulationInput) -> dict[str, Any]:
    allocation_known = _allocation_known(inputs)
    larger_buckets_allowed = allocation_known and _allocation_known_below_max(inputs)
    reason = "Current BTC allocation is unknown."
    if allocation_known and not larger_buckets_allowed:
        reason = "Current BTC allocation is at or above the max risk budget."
    elif larger_buckets_allowed:
        reason = "Current BTC allocation is known and below the max risk budget."
    return {
        "allocation_known": allocation_known,
        "larger_buckets_allowed": larger_buckets_allowed,
        "reason": reason,
    }


def _larger_buckets_blocked_by_allocation_gate(inputs: GoalAccumulationInput) -> bool:
    return not _allocation_known(inputs)


def _allocation_known(inputs: GoalAccumulationInput) -> bool:
    return inputs.current_btc_allocation_pct is not None


def _allocation_known_below_max(inputs: GoalAccumulationInput) -> bool:
    if not _allocation_known(inputs):
        return False
    if inputs.max_btc_allocation_pct is None:
        return inputs.rebalance_signal != "OVER_ALLOCATED"
    return float(inputs.current_btc_allocation_pct) < float(inputs.max_btc_allocation_pct)


def _allocation_below_target(inputs: GoalAccumulationInput) -> bool:
    if not _allocation_known(inputs) or inputs.target_btc_allocation_pct is None:
        return False
    return float(inputs.current_btc_allocation_pct) < float(inputs.target_btc_allocation_pct)


def _dip_context(inputs: GoalAccumulationInput) -> dict[str, Any]:
    drawdown_pct = None
    trigger_met = False
    available_bucket_pct_now = 0
    if inputs.reference_price_for_dip is not None and inputs.current_price is not None and inputs.reference_price_for_dip > 0:
        drawdown_pct = round(((inputs.current_price - inputs.reference_price_for_dip) / inputs.reference_price_for_dip) * 100, 4)
        if drawdown_pct <= -15:
            trigger_met = True
            available_bucket_pct_now = 25
        if drawdown_pct <= -25:
            available_bucket_pct_now = 50
        if drawdown_pct <= -35:
            available_bucket_pct_now = 75
        if drawdown_pct <= -35 and inputs.accumulation_score > 80:
            available_bucket_pct_now = 100
    return {
        "current_drawdown_from_reference_pct": drawdown_pct,
        "dip_trigger_met": trigger_met,
        "available_bucket_pct_now": available_bucket_pct_now,
    }


def _bucket_amounts(remaining_btc: Decimal) -> dict[str, Decimal]:
    base = _quantize_btc(remaining_btc * Decimal("0.10"))
    opportunistic = _quantize_btc(remaining_btc * Decimal("0.35"))
    dip_reserve = _quantize_btc(remaining_btc * Decimal("0.35"))
    deep_value = _quantize_btc(remaining_btc - base - opportunistic - dip_reserve)
    return {
        "base_dca": base,
        "opportunistic": opportunistic,
        "dip_reserve": dip_reserve,
        "deep_value_reserve": deep_value,
    }


def _btc(value: float | Decimal) -> Decimal:
    return _quantize_btc(Decimal(str(value)))


def _money(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _optional_money(value: float | None) -> float | None:
    if value is None:
        return None
    return _money_float(_money(value))


def _quantize_btc(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _btc_float(value: Decimal) -> float:
    return float(_quantize_btc(value))


def _money_float(value: Decimal) -> float:
    return float(_money(value))


def _max_decimal(left: Decimal, right: Decimal) -> Decimal:
    return left if left >= right else right
