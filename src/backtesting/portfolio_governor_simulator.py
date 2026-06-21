"""Regime-gated trend holding with a portfolio risk governor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtesting.regime_gated_trend_holding_simulator import RegimeGatedTrendHoldingSimulator
from backtesting.trade_simulator import ENTRY_DECISIONS, PortfolioSnapshot, _optional_float
from risk.portfolio_risk_governor import PortfolioRiskGovernor, RiskState
from strategy.trend_holding_engine import MacroRegime, determine_macro_regime, determine_trend_state


class PortfolioGovernorSimulator(RegimeGatedTrendHoldingSimulator):
    """Apply account-level drawdown, volatility, and risk-per-trade controls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.governor = PortfolioRiskGovernor(initial_equity=self.initial_capital)
        self.risk_state_counts = {state.value: 0 for state in RiskState}
        self.position_size_samples: list[float] = []
        self.runner_size_samples: list[float] = []
        self.defensive_mode_hours = 0.0
        self._last_risk_timestamp: pd.Timestamp | None = None
        self._last_risk_state = RiskState.NORMAL
        self._last_portfolio_stop_active = False

    def process_signal(self, signal: dict[str, Any]) -> PortfolioSnapshot:
        timestamp = self._format_signal_timestamp(signal)
        timestamp_value = pd.Timestamp(timestamp)
        price = float(signal["price"])
        final_decision = str(signal["final_decision"])
        market_regime = signal.get("market_regime")
        trend_score = int(signal.get("trend_score", 10))
        macro_regime = determine_macro_regime(signal)
        signal["macro_regime"] = macro_regime.value
        self.regime_period_counts[macro_regime.value] += 1
        risk_snapshot = self._update_risk(timestamp_value, price)
        signal["risk_state"] = risk_snapshot.risk_state
        signal["portfolio_drawdown_pct"] = risk_snapshot.current_drawdown_pct
        self.last_rejected_entry_reasons = []

        if self.position_size > 0 and self.runner_active and self.governor.portfolio_stop_active:
            self._close_long(timestamp, price, "PORTFOLIO_STOP", signal)

        if self._should_audit_entry(final_decision):
            rejected_reasons = self._entry_rejection_reasons(signal, timestamp_value)
            self.entry_attempts += 1
            if rejected_reasons:
                self._record_rejected_entry(rejected_reasons)
            else:
                self._open_long(timestamp, price, signal)

        if self.position_size > 0:
            self.governor.update_equity(self.cash + (self.position_size * price))
            if not self.trend_holding_allowed:
                exit_reason = self._exit_reason(signal, timestamp_value, price, market_regime, trend_score)
                if exit_reason is not None:
                    self._close_long(timestamp, price, exit_reason, signal)
            else:
                self._update_runner_high(price)
                if self.entry_stop_loss is not None and price <= self.entry_stop_loss:
                    self._close_long(timestamp, price, "STOP_LOSS", signal)
                else:
                    self._process_profit_targets(timestamp, price, macro_regime)
                    runner_exit_reason = self._runner_exit_reason(signal) if self.runner_active else None
                    if runner_exit_reason is not None and self.position_size > 0:
                        self._close_long(timestamp, price, runner_exit_reason, signal)

        snapshot = self.snapshot(timestamp, price)
        self.equity_curve.append(snapshot)
        self.governor.update_equity(snapshot.current_equity)
        return snapshot

    def _entry_rejection_reasons(self, signal: dict[str, Any], timestamp: pd.Timestamp) -> list[str]:
        reasons = super()._entry_rejection_reasons(signal, timestamp)
        final_decision = str(signal["final_decision"])
        if final_decision in ENTRY_DECISIONS and not any(
            reason in reasons for reason in ("not_buy_decision", "already_holding", "cooldown_active")
        ):
            entry_price = float(signal["price"]) * (1 + self.slippage_rate)
            stop_loss = _optional_float(signal.get("stop_loss"))
            details = self.governor.position_size_details(
                entry_price=entry_price,
                stop_price=stop_loss,
                available_cash=self.cash,
                fee_rate=self.fee_rate,
                atr=_optional_float(signal.get("atr")),
                atr_ma=_optional_float(signal.get("atr_ma")),
            )
            if details.position_size <= 0:
                reasons.append("invalid_stop_distance")
        return reasons

    def _open_long(self, timestamp: str, price: float, signal: dict[str, Any]) -> None:
        self._reset_execution_cost_tracking()
        self.governor.update_equity(self.cash)
        adjusted_entry_price = price * (1 + self.slippage_rate)
        stop_loss = _optional_float(signal.get("stop_loss"))
        details = self.governor.position_size_details(
            entry_price=adjusted_entry_price,
            stop_price=stop_loss,
            available_cash=self.cash,
            fee_rate=self.fee_rate,
            atr=_optional_float(signal.get("atr")),
            atr_ma=_optional_float(signal.get("atr_ma")),
        )
        if details.position_size <= 0:
            self._record_rejected_entry(["invalid_stop_distance"])
            return

        entry_notional = details.position_size * adjusted_entry_price
        entry_fee = entry_notional * self.fee_rate
        total_cost = entry_notional + entry_fee
        if total_cost > self.cash:
            self._record_rejected_entry(["portfolio_governor_block"])
            return

        self.position_size = details.position_size
        self.entry_signal_price = price
        self.entry_price = adjusted_entry_price
        self.entry_timestamp = timestamp
        self.entry_fee = entry_fee
        self.entry_total_cost = total_cost
        self.entry_stop_loss = stop_loss
        self.initial_risk = (
            adjusted_entry_price - self.entry_stop_loss
            if self.entry_stop_loss is not None and self.entry_stop_loss < adjusted_entry_price
            else 0.0
        )
        self.entry_target_1 = self._entry_target(signal)
        self.highest_price = price
        self.trailing_stop = None
        self.trailing_active = False
        self.entry_audit = {
            "open_reason": "portfolio_governor_entry_gates_passed",
            "signal_entry_price": price,
            "actual_entry_price": adjusted_entry_price,
            "entry_slippage_cost": self.position_size * (adjusted_entry_price - price),
            "entry_decision": str(signal["final_decision"]),
            "entry_alignment": str(signal.get("alignment", "")),
            "entry_rr_ratio": float(signal.get("rr_ratio", 0.0)),
            "entry_volume_ratio": float(signal.get("volume_ratio", 0.0)),
            "entry_market_regime": str(signal.get("market_regime", "")),
            "entry_regime_tag": self._regime_tag(signal),
            "entry_bull_mode_active": self._bull_market_mode_active(signal),
            "entry_daily_setup": str(signal.get("daily_setup", "")),
            "entry_4h_price": float(signal.get("four_hour_price", 0.0)),
            "entry_4h_ema20": float(signal.get("four_hour_ema20", 0.0)),
            "entry_stop_loss": self.entry_stop_loss,
            "entry_target_1": self.entry_target_1,
            "initial_risk": self.initial_risk,
            "entry_stop_type": str(signal.get("stop_type", "FIXED") or "FIXED"),
            "entry_stop_distance_pct": _optional_float(signal.get("stop_distance_pct")),
            "entry_stop_distance_atr": _optional_float(signal.get("stop_distance_atr")),
            "entry_atr": _optional_float(signal.get("atr")),
            "entry_stop_candidates": signal.get("stop_candidates") or {},
            "rejected_entry_reasons": [],
            "risk_state": self.governor.risk_state.value,
            "portfolio_drawdown_pct": self.governor.current_drawdown_pct,
            "allocation_multiplier": details.allocation_multiplier,
            "volatility_factor": details.volatility_factor,
            "position_size_fraction": details.position_size_fraction,
        }
        self.cash -= total_cost
        self.accepted_entries += 1
        self.position_size_samples.append(details.position_size_fraction * 100)

        self.initial_position_size = self.position_size
        self.original_entry_total_cost = self.entry_total_cost
        self.partial_realized_pnl = 0.0
        self.partial_exits = []
        self.tp1_done = False
        self.tp2_done = False
        self.runner_active = False
        self.runner_activation_timestamp = None
        self.runner_highest_close = price
        self.tp1_price = self.engine.tp1_price(self.entry_price, self.initial_risk)
        self.tp2_price = self.engine.tp2_price(self.entry_price, self.initial_risk)
        self.entry_audit["tp1_price"] = self.tp1_price
        self.entry_audit["tp2_price"] = self.tp2_price
        self.entry_audit["entry_trend_state"] = determine_trend_state(signal).value

        macro_regime = determine_macro_regime(signal)
        self.trend_holding_allowed = (
            macro_regime == MacroRegime.STRONG_BULL
            and self.governor.trend_holding_enabled
        )
        self.pending_runner_disabled = False
        if macro_regime == MacroRegime.STRONG_BULL and not self.trend_holding_allowed:
            self.runner_disabled_count += 1
        self.active_runner_fraction = self.engine.config.runner_fraction if self.trend_holding_allowed else 0.0
        self.entry_portfolio_drawdown_pct = self.governor.current_drawdown_pct
        self.entry_audit["macro_regime"] = macro_regime.value
        self.entry_audit["trend_holding_allowed"] = self.trend_holding_allowed
        self.entry_audit["runner_fraction"] = self.active_runner_fraction

    def _process_profit_targets(
        self,
        timestamp: str,
        price: float,
        macro_regime: MacroRegime | None = None,
    ) -> None:
        if self.entry_price is None or self.initial_risk <= 0 or self.initial_position_size <= 0:
            return
        if not self.tp1_done and self.tp1_price is not None and price >= self.tp1_price:
            self._partial_close(timestamp, price, self.engine.config.tp1_close_fraction, "TP1_2R")
            self.tp1_done = True
            self.tp1_hit_count += 1
            self.entry_stop_loss = self.entry_price
            self.entry_audit["entry_stop_loss"] = self.entry_stop_loss
        if not self.tp2_done and self.tp2_price is not None and price >= self.tp2_price:
            self.tp2_done = True
            self.tp2_hit_count += 1
            self.governor.update_equity(self.cash + (self.position_size * float(price)))
            current_regime = macro_regime or MacroRegime(str(self.entry_audit.get("macro_regime", MacroRegime.RANGE.value)))
            if current_regime != MacroRegime.STRONG_BULL or not self.governor.runner_enabled:
                self._partial_close(timestamp, price, self.engine.config.tp2_close_fraction, "TP2_4R")
                self.trend_holding_allowed = False
                self.pending_runner_disabled = True
                self.runner_disabled_count += 1
                return
            runner_fraction = self.engine.config.runner_fraction
            self.active_runner_fraction = runner_fraction
            close_fraction = max(0.0, 1.0 - self.engine.config.tp1_close_fraction - runner_fraction)
            self._partial_close(timestamp, price, close_fraction, "TP2_4R")
            self.runner_active = True
            self.runner_activation_count += 1
            self.runner_activation_timestamp = timestamp
            self.runner_highest_close = price
            self.runner_size_samples.append(runner_fraction * 100)
            self.entry_audit["runner_fraction"] = runner_fraction

    def execution_summary(self, total_decisions: int) -> dict[str, Any]:
        summary = super().execution_summary(total_decisions)
        summary.update(
            {
                "risk_state_counts": self.risk_state_counts.copy(),
                "average_position_size": _round_or_none(_mean(self.position_size_samples)),
                "average_runner_size": _round_or_none(_mean(self.runner_size_samples)),
                "portfolio_stop_count": self.governor.portfolio_stop_count,
                "defensive_mode_hours": round(self.defensive_mode_hours, 2),
            }
        )
        return summary

    def _update_risk(self, timestamp: pd.Timestamp, price: float):
        equity = self.cash + (self.position_size * float(price))
        self._accumulate_defensive_hours(timestamp)
        snapshot = self.governor.update_equity(equity)
        self.risk_state_counts[snapshot.risk_state] += 1
        self._last_risk_timestamp = timestamp
        self._last_risk_state = RiskState(snapshot.risk_state)
        self._last_portfolio_stop_active = snapshot.portfolio_stop_active
        return snapshot

    def _accumulate_defensive_hours(self, timestamp: pd.Timestamp) -> None:
        if self._last_risk_timestamp is None:
            return
        if self._last_risk_state in {RiskState.DEFENSIVE, RiskState.CAPITAL_PRESERVATION} or self._last_portfolio_stop_active:
            elapsed = (timestamp - self._last_risk_timestamp).total_seconds() / 3600
            self.defensive_mode_hours += max(0.0, elapsed)

    @staticmethod
    def _regime_tag(signal: dict[str, Any]) -> str:
        from backtesting.trade_simulator import _regime_tag

        return _regime_tag(signal)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
