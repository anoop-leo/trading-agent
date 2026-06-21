"""BTC Investor Agent V1 with margin-of-safety scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents.btc.cycle_phase_agent import (
    CyclePhaseInput,
    classify_btc_cycle_phase,
    cycle_adjusted_action,
    cycle_adjusted_dca_multiplier,
)
from data.fear_greed_provider import FearGreedProvider
from data.liquidity_provider import LiquidityProvider
from data.mvrv_provider import MVRVProvider
from data.thesis_risk_provider import ThesisRiskProvider
from planning.goal_accumulation_plan import GoalAccumulationInput, build_goal_accumulation_plan
from scoring.accumulation_score import (
    AccumulationBand,
    WEIGHTS,
    band_for_accumulation_score,
    calculate_accumulation_score,
    factor_payload,
    score_distance_from_200d_ma,
    score_drawdown_from_cycle_high,
    score_fear_and_greed,
    score_liquidity_condition,
    score_monthly_trend,
    score_mvrv,
    score_thesis_risk,
    score_weekly_rsi,
)
from scoring.institutional_overlay import InstitutionalOverlayInput, calculate_institutional_overlay
from scoring.margin_of_safety import calculate_margin_of_safety_score, margin_of_safety_band
from trading_agent.data import BinanceKlineProvider, DataLoadError
from trading_agent.indicators import calculate_ema, calculate_rsi


INVESTOR_REPORT_FILENAME = "investor_accumulation_report.json"
MVRV_MISSING_RATIONALE = "MVRV is unavailable, so aggressive accumulation is capped until valuation is confirmed."
MVRV_ELEVATED_RATIONALE = "MVRV is elevated, so accumulation is capped despite other positive signals."


@dataclass(frozen=True)
class InvestorAgentConfig:
    symbol: str = "BTC"
    output_dir: Path = Path("outputs")
    offline: bool = False
    request_timeout_seconds: float = 10.0
    binance_base_url: str = "https://api.binance.com"
    portfolio_risk_profile: str = "BALANCED"
    current_btc_allocation_pct: float | None = None
    target_btc_allocation_pct: float | None = None
    max_btc_allocation_pct: float | None = None
    rebalance_threshold_pct: float = 25.0
    current_btc: float = 1.13059494
    target_btc: float = 2.0
    target_sell_price: float = 500000.0
    planned_sell_btc: float = 1.0
    retain_btc: float = 1.0
    monthly_dca_usd: float | None = None
    lump_sum_available_usd: float | None = None
    reference_price_for_dip: float | None = None


@dataclass(frozen=True)
class MarketMetrics:
    price: float | None
    ma200: float | None
    ma200w: float | None
    distance_from_200d_ma_pct: float | None
    weekly_rsi: float | None
    monthly_ema20: float | None
    monthly_trend: str
    drawdown_from_cycle_high_pct: float | None
    missing_fields: list[str]


@dataclass(frozen=True)
class ConfidenceAdjustment:
    accumulation_band: str
    dca_intensity: str
    suggested_dca_multiplier: str
    confidence_adjusted_dca_multiplier: str
    confidence_adjustments: list[str]
    rationale_warnings: list[str]


class InvestorAgentError(ValueError):
    """Raised when Investor Agent configuration is invalid."""


class BTCInvestorAgent:
    """Long-term BTC accumulation guidance agent.

    This agent intentionally avoids trading labels and broker integrations.
    """

    def __init__(
        self,
        config: InvestorAgentConfig,
        market_data_provider: BinanceKlineProvider | None = None,
        fear_greed_provider: FearGreedProvider | None = None,
        mvrv_provider: MVRVProvider | None = None,
        liquidity_provider: LiquidityProvider | None = None,
        thesis_risk_provider: ThesisRiskProvider | None = None,
    ) -> None:
        symbol = config.symbol.upper()
        if symbol != "BTC":
            raise InvestorAgentError("Investor Agent V1 supports BTC only.")
        self.config = InvestorAgentConfig(
            symbol=symbol,
            output_dir=Path(config.output_dir),
            offline=config.offline,
            request_timeout_seconds=config.request_timeout_seconds,
            binance_base_url=config.binance_base_url.rstrip("/"),
            portfolio_risk_profile=config.portfolio_risk_profile.upper(),
            current_btc_allocation_pct=config.current_btc_allocation_pct,
            target_btc_allocation_pct=config.target_btc_allocation_pct,
            max_btc_allocation_pct=config.max_btc_allocation_pct,
            rebalance_threshold_pct=config.rebalance_threshold_pct,
            current_btc=config.current_btc,
            target_btc=config.target_btc,
            target_sell_price=config.target_sell_price,
            planned_sell_btc=config.planned_sell_btc,
            retain_btc=config.retain_btc,
            monthly_dca_usd=config.monthly_dca_usd,
            lump_sum_available_usd=config.lump_sum_available_usd,
            reference_price_for_dip=config.reference_price_for_dip,
        )
        self.market_data_provider = market_data_provider or BinanceKlineProvider(
            base_url=self.config.binance_base_url,
            timeout_seconds=self.config.request_timeout_seconds,
        )
        self.fear_greed_provider = fear_greed_provider or FearGreedProvider(timeout_seconds=self.config.request_timeout_seconds)
        self.mvrv_provider = mvrv_provider or MVRVProvider()
        self.liquidity_provider = liquidity_provider or LiquidityProvider()
        self.thesis_risk_provider = thesis_risk_provider or ThesisRiskProvider()

    def run(self) -> dict[str, Any]:
        market = self._market_metrics()
        fear_greed = self.fear_greed_provider.fetch(offline=self.config.offline)
        mvrv = self.mvrv_provider.fetch(offline=self.config.offline)
        liquidity = self.liquidity_provider.fetch(offline=self.config.offline)
        thesis_risk = self.thesis_risk_provider.fetch(offline=self.config.offline)

        factor_scores = self._factor_scores(market, fear_greed, mvrv, liquidity, thesis_risk)
        accumulation_score = calculate_accumulation_score(factor_scores)
        band = band_for_accumulation_score(accumulation_score, thesis_risk["level"])
        confidence_adjustment = _apply_mvrv_confidence_gate(band, mvrv["value"], mvrv.get("missing", False))
        margin_score = calculate_margin_of_safety_score(factor_scores)
        institutional_overlay = calculate_institutional_overlay(
            InstitutionalOverlayInput(
                portfolio_risk_profile=self.config.portfolio_risk_profile,
                current_btc_allocation_pct=self.config.current_btc_allocation_pct,
                target_btc_allocation_pct=self.config.target_btc_allocation_pct,
                max_btc_allocation_pct=self.config.max_btc_allocation_pct,
                rebalance_threshold_pct=self.config.rebalance_threshold_pct,
            )
        )
        missing_fields, fallback_fields = self._data_quality_fields(market, fear_greed, mvrv, liquidity, thesis_risk)
        fallback_fields = sorted(set(fallback_fields + institutional_overlay.fallback_fields))
        weighted_data_gaps = _weighted_missing_or_fallback_points(missing_fields, fallback_fields)
        confidence_adjusted_dca_multiplier, dca_adjustments = _apply_final_dca_caps(
            confidence_adjustment.confidence_adjusted_dca_multiplier,
            thesis_risk["level"],
            institutional_overlay.dca_cap_multiplier,
        )
        final_investor_action = _final_investor_action(
            accumulation_score=accumulation_score,
            institutional_score=institutional_overlay.institutional_score,
            margin_of_safety_score=margin_score,
            thesis_risk_level=thesis_risk["level"],
            rebalance_signal=institutional_overlay.rebalance_signal,
            weighted_data_gaps=weighted_data_gaps,
        )
        cycle_overlay = classify_btc_cycle_phase(
            CyclePhaseInput(
                price=market.price,
                ma200=market.ma200,
                ma200w=market.ma200w,
                monthly_ema20=market.monthly_ema20,
                weekly_rsi=market.weekly_rsi,
                mvrv=mvrv["value"],
                drawdown_from_cycle_high_pct=market.drawdown_from_cycle_high_pct,
                fear_and_greed=fear_greed["value"],
            )
        )
        cycle_dca_multiplier = cycle_adjusted_dca_multiplier(
            confidence_adjusted_dca_multiplier,
            cycle_overlay["cycle_phase"],
        )
        cycle_action = cycle_adjusted_action(final_investor_action, cycle_overlay["cycle_phase"])
        goal_plan = build_goal_accumulation_plan(
            GoalAccumulationInput(
                current_btc=self.config.current_btc,
                target_btc=self.config.target_btc,
                current_price=market.price,
                monthly_dca_usd=self.config.monthly_dca_usd,
                lump_sum_available_usd=self.config.lump_sum_available_usd,
                target_sell_price=self.config.target_sell_price,
                planned_sell_btc=self.config.planned_sell_btc,
                retain_btc=self.config.retain_btc,
                accumulation_score=accumulation_score,
                accumulation_band=confidence_adjustment.accumulation_band,
                margin_of_safety_score=margin_score,
                mvrv_value=mvrv["value"],
                mvrv_missing=mvrv.get("missing", False),
                fear_and_greed_value=fear_greed["value"],
                thesis_risk_level=thesis_risk["level"],
                final_investor_action=final_investor_action,
                rebalance_signal=institutional_overlay.rebalance_signal,
                current_btc_allocation_pct=institutional_overlay.current_btc_allocation_pct,
                target_btc_allocation_pct=institutional_overlay.target_btc_allocation_pct,
                max_btc_allocation_pct=institutional_overlay.max_btc_allocation_pct,
                institutional_score=institutional_overlay.institutional_score,
                reference_price_for_dip=self.config.reference_price_for_dip,
                distance_from_200d_ma_pct=market.distance_from_200d_ma_pct,
                drawdown_from_cycle_high_pct=market.drawdown_from_cycle_high_pct,
            )
        )
        payload = {
            "agent": "BTC_INVESTOR",
            "symbol": "BTC",
            "generated_at": datetime.now(UTC).isoformat(),
            "accumulation_score": accumulation_score,
            "accumulation_band": confidence_adjustment.accumulation_band,
            "margin_of_safety_score": margin_score,
            "margin_of_safety_band": margin_of_safety_band(margin_score),
            "dca_intensity": confidence_adjustment.dca_intensity,
            "suggested_dca_multiplier": confidence_adjustment.suggested_dca_multiplier,
            "confidence_adjusted_dca_multiplier": confidence_adjusted_dca_multiplier,
            "confidence_adjustments": confidence_adjustment.confidence_adjustments + dca_adjustments,
            "final_investor_action": final_investor_action,
            "cycle_overlay": cycle_overlay,
            "cycle_adjusted_dca_multiplier": cycle_dca_multiplier,
            "cycle_adjusted_action": cycle_action,
            "institutional_overlay": institutional_overlay.to_payload(),
            "allocation_gate": goal_plan["allocation_gate"],
            "goal_plan": goal_plan,
            "thesis_risk": {
                "level": thesis_risk["level"],
                "flags": thesis_risk["flags"],
            },
            "factor_scores": factor_scores,
            "rationale": self._rationale(
                factor_scores,
                confidence_adjustment.accumulation_band,
                margin_score,
                thesis_risk,
                confidence_adjustment.rationale_warnings,
            ),
            "data_quality": {
                "missing_fields": missing_fields,
                "fallback_fields": fallback_fields,
                "confidence": _data_quality_confidence(missing_fields, fallback_fields),
            },
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / INVESTOR_REPORT_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return payload

    def _market_metrics(self) -> MarketMetrics:
        if self.config.offline:
            return MarketMetrics(None, None, None, None, None, None, "", None, _market_missing_fields())

        missing_fields: list[str] = []
        daily = _safe_fetch(self.market_data_provider, "BTCUSDT", "1d", 1000)
        weekly = _safe_fetch(self.market_data_provider, "BTCUSDT", "1w", 300)
        monthly = _safe_fetch(self.market_data_provider, "BTCUSDT", "1M", 120)

        price = None
        ma200 = None
        ma200w = None
        distance_pct = None
        drawdown_pct = None
        if daily is None or len(daily) < 200:
            missing_fields.extend(["distance_from_200d_ma", "drawdown_from_cycle_high"])
        else:
            price = float(daily.iloc[-1]["close"])
            ma200 = float(daily["close"].rolling(200, min_periods=200).mean().iloc[-1])
            distance_pct = round(((price - ma200) / ma200) * 100, 4) if ma200 > 0 else None
            cycle_high = float(daily.tail(min(len(daily), 1460))["high"].max())
            drawdown_pct = round(((price - cycle_high) / cycle_high) * 100, 4) if cycle_high > 0 else None

        weekly_rsi = None
        if weekly is None or len(weekly) < 20:
            missing_fields.append("weekly_rsi")
        else:
            weekly_close = weekly["close"].astype(float)
            weekly_rsi = float(calculate_rsi(weekly_close, 14).iloc[-1])
            if len(weekly_close) >= 200:
                ma200w = float(weekly_close.rolling(200, min_periods=200).mean().iloc[-1])

        monthly_trend = ""
        monthly_ema20 = None
        if monthly is None or len(monthly) < 25:
            missing_fields.append("monthly_trend")
        else:
            monthly_ema20 = float(calculate_ema(monthly["close"].astype(float), 20).iloc[-1])
            monthly_trend = classify_monthly_trend(monthly)

        return MarketMetrics(price, ma200, ma200w, distance_pct, weekly_rsi, monthly_ema20, monthly_trend, drawdown_pct, missing_fields)

    def _factor_scores(
        self,
        market: MarketMetrics,
        fear_greed: dict[str, Any],
        mvrv: dict[str, Any],
        liquidity: dict[str, Any],
        thesis_risk: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        mvrv_payload = factor_payload(mvrv["value"], score_mvrv(mvrv["value"]), WEIGHTS["mvrv"])
        mvrv_payload["source"] = mvrv.get("source")
        mvrv_payload["as_of"] = mvrv.get("as_of")
        return {
            "distance_from_200d_ma": factor_payload(
                market.distance_from_200d_ma_pct,
                score_distance_from_200d_ma(market.distance_from_200d_ma_pct),
                WEIGHTS["distance_from_200d_ma"],
                "value_pct",
            ),
            "mvrv": mvrv_payload,
            "weekly_rsi": factor_payload(market.weekly_rsi, score_weekly_rsi(market.weekly_rsi), WEIGHTS["weekly_rsi"]),
            "fear_and_greed": factor_payload(
                fear_greed["value"],
                score_fear_and_greed(fear_greed["value"]),
                WEIGHTS["fear_and_greed"],
            ),
            "monthly_trend": factor_payload(
                market.monthly_trend,
                score_monthly_trend(market.monthly_trend),
                WEIGHTS["monthly_trend"],
            ),
            "drawdown_from_cycle_high": factor_payload(
                market.drawdown_from_cycle_high_pct,
                score_drawdown_from_cycle_high(market.drawdown_from_cycle_high_pct),
                WEIGHTS["drawdown_from_cycle_high"],
                "value_pct",
            ),
            "liquidity_condition": factor_payload(
                liquidity["value"],
                score_liquidity_condition(liquidity["value"]),
                WEIGHTS["liquidity_condition"],
            ),
            "thesis_risk_check": factor_payload(
                thesis_risk["level"],
                score_thesis_risk(thesis_risk["level"]),
                WEIGHTS["thesis_risk_check"],
            ),
        }

    def _data_quality_fields(
        self,
        market: MarketMetrics,
        fear_greed: dict[str, Any],
        mvrv: dict[str, Any],
        liquidity: dict[str, Any],
        thesis_risk: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        missing_fields = list(market.missing_fields)
        fallback_fields = []
        for name, provider_result in (
            ("fear_and_greed", fear_greed),
            ("mvrv", mvrv),
            ("liquidity_condition", liquidity),
            ("thesis_risk", thesis_risk),
        ):
            if provider_result.get("missing"):
                missing_fields.append(name)
            if provider_result.get("fallback"):
                fallback_fields.append(name)
            if name == "mvrv" and provider_result.get("source") == "cache/manual":
                fallback_fields.append("mvrv_cache_used")
        return sorted(set(missing_fields)), sorted(set(fallback_fields))

    def _rationale(
        self,
        factor_scores: dict[str, dict[str, Any]],
        accumulation_band: str,
        margin_score: int,
        thesis_risk: dict[str, Any],
        confidence_warnings: list[str],
    ) -> list[str]:
        rationale = [
            f"Accumulation band is {accumulation_band} with a score of {calculate_accumulation_score(factor_scores)}.",
            f"Margin of safety score is {margin_score}, based on valuation, trend discount, drawdown, and sentiment.",
        ]
        distance = factor_scores["distance_from_200d_ma"]["value_pct"]
        if distance is not None:
            rationale.append(f"BTC is {distance}% from its 200D moving average.")
        rationale.extend(confidence_warnings)
        if thesis_risk["level"] == "HIGH":
            rationale.append("Structural thesis risk is HIGH, so accumulation guidance is capped at FAIR.")
        elif thesis_risk["flags"]:
            rationale.append("Structural risk flags are present and should be reviewed before increasing DCA.")
        return rationale


def run_investor_agent(
    config: InvestorAgentConfig,
    market_data_provider: BinanceKlineProvider | None = None,
    fear_greed_provider: FearGreedProvider | None = None,
    mvrv_provider: MVRVProvider | None = None,
    liquidity_provider: LiquidityProvider | None = None,
    thesis_risk_provider: ThesisRiskProvider | None = None,
) -> dict[str, Any]:
    agent = BTCInvestorAgent(
        config,
        market_data_provider=market_data_provider,
        fear_greed_provider=fear_greed_provider,
        mvrv_provider=mvrv_provider,
        liquidity_provider=liquidity_provider,
        thesis_risk_provider=thesis_risk_provider,
    )
    return agent.run()


def classify_monthly_trend(monthly: pd.DataFrame) -> str:
    close = monthly["close"].astype(float)
    ema10 = calculate_ema(close, 10)
    ema20 = calculate_ema(close, 20)
    latest_close = float(close.iloc[-1])
    latest_ema10 = float(ema10.iloc[-1])
    latest_ema20 = float(ema20.iloc[-1])
    if latest_ema20 <= 0:
        return ""
    distance_pct = ((latest_close - latest_ema20) / latest_ema20) * 100
    if latest_close < latest_ema20:
        return "BELOW_EMA20"
    if abs(distance_pct) <= 5:
        return "NEAR_EMA20"
    if latest_ema10 < latest_ema20:
        return "ABOVE_EMA20_EMA10_BELOW_EMA20"
    if distance_pct > 60:
        return "EXTREMELY_EXTENDED"
    return "BULLISH_ABOVE_EMA20"


def _safe_fetch(provider: BinanceKlineProvider, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    try:
        return provider.fetch_ohlcv(symbol, interval, limit)
    except (DataLoadError, OSError, TimeoutError):
        return None


def _market_missing_fields() -> list[str]:
    return [
        "distance_from_200d_ma",
        "weekly_rsi",
        "monthly_trend",
        "drawdown_from_cycle_high",
    ]


def _data_quality_confidence(missing_fields: list[str], fallback_fields: list[str]) -> str:
    if not missing_fields and len(fallback_fields) <= 1:
        return "HIGH"
    if len(missing_fields) <= 2:
        return "MEDIUM"
    return "LOW"


def _weighted_missing_or_fallback_points(missing_fields: list[str], fallback_fields: list[str]) -> int:
    field_weights = {
        "distance_from_200d_ma": WEIGHTS["distance_from_200d_ma"],
        "mvrv": WEIGHTS["mvrv"],
        "weekly_rsi": WEIGHTS["weekly_rsi"],
        "fear_and_greed": WEIGHTS["fear_and_greed"],
        "monthly_trend": WEIGHTS["monthly_trend"],
        "drawdown_from_cycle_high": WEIGHTS["drawdown_from_cycle_high"],
        "liquidity_condition": WEIGHTS["liquidity_condition"],
        "thesis_risk": WEIGHTS["thesis_risk_check"],
        "store_of_value_thesis": 25,
        "network_adoption": 15,
    }
    data_gap_fields = set(missing_fields + fallback_fields)
    return sum(weight for field, weight in field_weights.items() if field in data_gap_fields)


def _apply_final_dca_caps(
    multiplier: str,
    thesis_risk_level: str,
    institutional_cap: str | None,
) -> tuple[str, list[str]]:
    capped = multiplier
    adjustments: list[str] = []
    if thesis_risk_level.upper() == "HIGH":
        capped = _cap_dca_multiplier_at_0_5(capped)
        adjustments.append("High thesis risk; capped max DCA multiplier at 0.5x.")
    if institutional_cap == "0.0x to 0.5x normal DCA":
        capped = _cap_dca_multiplier_at_0_5(capped)
        adjustments.append("BTC allocation above max risk budget; capped max DCA multiplier at 0.5x.")
    elif institutional_cap == "0.5x to 1.0x normal DCA":
        capped = _cap_dca_multiplier_at_1_0(capped)
        adjustments.append("BTC allocation above target; capped max DCA multiplier at 1.0x.")
    return capped, adjustments


def _final_investor_action(
    accumulation_score: int,
    institutional_score: int,
    margin_of_safety_score: int,
    thesis_risk_level: str,
    rebalance_signal: str,
    weighted_data_gaps: int,
) -> str:
    if thesis_risk_level.upper() == "HIGH":
        return "PAUSE_EXTRA_DCA"
    if rebalance_signal == "OVER_ALLOCATED":
        return "REBALANCE_WARNING"
    if weighted_data_gaps > 50:
        return "INSUFFICIENT_DATA"
    if accumulation_score >= 80 and institutional_score >= 70 and margin_of_safety_score >= 70:
        return "AGGRESSIVE_DCA_ALLOWED"
    if accumulation_score >= 60 and institutional_score >= 50:
        return "INCREASE_DCA_GRADUALLY"
    if 30 <= accumulation_score < 60:
        return "NORMAL_DCA"
    return "PAUSE_EXTRA_DCA"


def _apply_mvrv_confidence_gate(
    band: AccumulationBand,
    mvrv_value: float | None,
    mvrv_missing: bool,
) -> ConfidenceAdjustment:
    if mvrv_missing or mvrv_value is None:
        return ConfidenceAdjustment(
            accumulation_band=band.accumulation_band,
            dca_intensity=band.dca_intensity,
            suggested_dca_multiplier=band.suggested_dca_multiplier,
            confidence_adjusted_dca_multiplier=_cap_dca_multiplier_at_1_25(band.suggested_dca_multiplier),
            confidence_adjustments=["MVRV missing; capped max DCA multiplier at 1.25x."],
            rationale_warnings=[MVRV_MISSING_RATIONALE],
        )

    if float(mvrv_value) > 3.0:
        capped_band = band
        if band.accumulation_band in {"GOOD_ACCUMULATION", "AGGRESSIVE_ACCUMULATION"}:
            capped_band = AccumulationBand("FAIR", "NORMAL_DCA", "0.5x to 1.0x normal DCA")
        return ConfidenceAdjustment(
            accumulation_band=capped_band.accumulation_band,
            dca_intensity=capped_band.dca_intensity,
            suggested_dca_multiplier=_cap_dca_multiplier_at_1_0(capped_band.suggested_dca_multiplier),
            confidence_adjusted_dca_multiplier=_cap_dca_multiplier_at_1_0(capped_band.suggested_dca_multiplier),
            confidence_adjustments=["MVRV elevated; capped max DCA multiplier at 1.0x."],
            rationale_warnings=[MVRV_ELEVATED_RATIONALE],
        )

    return ConfidenceAdjustment(
        accumulation_band=band.accumulation_band,
        dca_intensity=band.dca_intensity,
        suggested_dca_multiplier=band.suggested_dca_multiplier,
        confidence_adjusted_dca_multiplier=band.suggested_dca_multiplier,
        confidence_adjustments=[],
        rationale_warnings=[],
    )


def _cap_dca_multiplier_at_1_25(multiplier: str) -> str:
    if multiplier in {"1.0x to 1.5x normal DCA", "1.5x to 2.5x normal DCA"}:
        return "1.0x to 1.25x normal DCA"
    return multiplier


def _cap_dca_multiplier_at_1_0(multiplier: str) -> str:
    if multiplier in {"1.0x to 1.5x normal DCA", "1.5x to 2.5x normal DCA"}:
        return "0.5x to 1.0x normal DCA"
    return multiplier


def _cap_dca_multiplier_at_0_5(multiplier: str) -> str:
    if multiplier == "0.0x to 0.25x normal DCA":
        return multiplier
    return "0.0x to 0.5x normal DCA"
