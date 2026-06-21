"""Live gate that turns a PositionRecommendation into a RiskDecision.

This is the only module allowed to size a position. Investor agents and the
signal engine only ever propose; this module approves, trims, or blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from decision.recommendation import PositionRecommendation, RiskDecision
from risk.portfolio_risk_governor import PortfolioRiskGovernor, PositionSizeResult
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


_POSITION_CAP_PCT_BY_BUCKET = {
    "growth": "growth_position_max_pct",
    "speculative": "speculative_position_max_pct",
}


@dataclass(frozen=True)
class LiveRiskEngine:
    config: RiskEngineConfig

    def evaluate(self, recommendation: PositionRecommendation, state: PortfolioState) -> RiskDecision:
        if recommendation.action == "hold":
            return RiskDecision(recommendation, "approved", 0.0)
        if recommendation.action in ("sell", "trim"):
            return RiskDecision(recommendation, "approved", recommendation.suggested_size_usd)

        return self._evaluate_buy(recommendation, state)

    def _evaluate_buy(self, recommendation: PositionRecommendation, state: PortfolioState) -> RiskDecision:
        drawdown_pct = state.drawdown_pct
        if drawdown_pct > self.config.portfolio_drawdown_circuit_breaker_pct and recommendation.bucket != "core":
            return RiskDecision(
                recommendation,
                "blocked",
                0.0,
                (
                    f"Portfolio drawdown circuit breaker active: {drawdown_pct:.1f}% drawdown exceeds the "
                    f"{self.config.portfolio_drawdown_circuit_breaker_pct:.1f}% threshold. Only core/index buys "
                    "are allowed until drawdown recovers below "
                    f"{self.config.portfolio_drawdown_recovery_pct:.1f}% or this is manually cleared."
                ),
            )

        size = recommendation.suggested_size_usd
        reasons: list[str] = []

        cap_attr = _POSITION_CAP_PCT_BY_BUCKET.get(recommendation.bucket)
        if cap_attr is not None:
            cap_pct = getattr(self.config, cap_attr)
            cap_usd = state.total_value_usd * cap_pct / 100
            if size > cap_usd:
                size = cap_usd
                reasons.append(f"trimmed to the {cap_pct:.1f}% single-position cap for {recommendation.bucket}")

        if recommendation.bucket == "speculative":
            bucket_cap_usd = state.total_value_usd * self.config.speculative_max_pct / 100
            room = max(0.0, bucket_cap_usd - state.speculative_usd)
            if size > room:
                size = room
                reasons.append(f"trimmed to stay within the {self.config.speculative_max_pct:.1f}% speculative bucket cap")

        cash_floor_usd = state.total_value_usd * self.config.cash_buffer_min_pct / 100
        cash_room = max(0.0, state.cash_usd - cash_floor_usd)
        if size > cash_room:
            size = cash_room
            reasons.append(f"trimmed to preserve the {self.config.cash_buffer_min_pct:.1f}% cash buffer")

        size = max(0.0, round(size, 2))
        if size <= 0:
            return RiskDecision(
                recommendation,
                "blocked",
                0.0,
                "; ".join(reasons) or "No room under bucket and cash-buffer caps.",
            )
        if size < round(recommendation.suggested_size_usd, 2):
            return RiskDecision(recommendation, "adjusted", size, "; ".join(reasons))
        return RiskDecision(recommendation, "approved", size)

    def position_size_for_trade(
        self,
        *,
        entry_price: float,
        stop_price: float | None,
        available_cash: float,
        asset_class: str,
        equity: float | None = None,
        fee_rate: float = 0.0,
        atr: float | None = None,
        atr_ma: float | None = None,
    ) -> PositionSizeResult:
        """Per-trade stop-based sizing, extended with an equity gap buffer.

        Equities can gap overnight or over a weekend in a way 24/7 crypto markets
        don't; this widens the effective stop distance for equities by
        config.equity_gap_buffer_pct before sizing, so the same risk-per-trade
        budget buys a smaller position when the honored fill could be worse
        than the stop price.
        """

        stop = stop_price
        if asset_class == "equity" and stop is not None and stop < entry_price:
            gap_buffer = self.config.equity_gap_buffer_pct / 100
            stop = stop_price - (entry_price - stop_price) * gap_buffer

        governor = PortfolioRiskGovernor(
            initial_equity=equity if equity is not None else available_cash,
            risk_per_trade=self.config.risk_per_trade_pct / 100,
        )
        return governor.position_size_details(
            entry_price=entry_price,
            stop_price=stop,
            available_cash=available_cash,
            fee_rate=fee_rate,
            atr=atr,
            atr_ma=atr_ma,
        )
