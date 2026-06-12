"""Regime-gated Trend Holding simulator for Phase 1.14 backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtesting.trade_simulator import PortfolioSnapshot, _optional_float
from backtesting.trend_holding_simulator import TrendHoldingSimulator
from strategy.trend_holding_engine import MacroRegime, determine_macro_regime


class RegimeGatedTrendHoldingSimulator(TrendHoldingSimulator):
    """Activate trend runners only in strong bull regimes with drawdown controls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.portfolio_peak_equity = self.initial_capital
        self.current_portfolio_drawdown_pct = 0.0
        self.runner_disabled_count = 0
        self.regime_period_counts = {
            MacroRegime.STRONG_BULL.value: 0,
            MacroRegime.BULL.value: 0,
            MacroRegime.RANGE.value: 0,
            MacroRegime.BEAR.value: 0,
        }
        self.trend_holding_allowed = False
        self.pending_runner_disabled = False
        self.active_runner_fraction = self.engine.config.runner_fraction

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
        self._update_portfolio_drawdown(price)
        self.last_rejected_entry_reasons = []

        if self._should_audit_entry(final_decision):
            rejected_reasons = self._entry_rejection_reasons(signal, timestamp_value)
            self.entry_attempts += 1
            if rejected_reasons:
                self._record_rejected_entry(rejected_reasons)
            else:
                self._open_long(timestamp, price, signal)

        if self.position_size > 0:
            self._update_portfolio_drawdown(price)
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
        self._record_snapshot_drawdown(snapshot.current_equity)
        return snapshot

    def _open_long(self, timestamp: str, price: float, signal: dict[str, Any]) -> None:
        super()._open_long(timestamp, price, signal)
        macro_regime = determine_macro_regime(signal)
        self.trend_holding_allowed = (
            macro_regime == MacroRegime.STRONG_BULL
            and self.current_portfolio_drawdown_pct <= 20.0
        )
        self.pending_runner_disabled = False
        if macro_regime == MacroRegime.STRONG_BULL and self.current_portfolio_drawdown_pct > 20.0:
            self.runner_disabled_count += 1
        self.active_runner_fraction = self._runner_fraction_for_drawdown(self.current_portfolio_drawdown_pct)
        self.entry_portfolio_drawdown_pct = self.current_portfolio_drawdown_pct
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
            self._update_portfolio_drawdown(price)
            drawdown_for_runner = max(self.current_portfolio_drawdown_pct, self.entry_portfolio_drawdown_pct)
            current_regime = macro_regime or MacroRegime(str(self.entry_audit.get("macro_regime", MacroRegime.RANGE.value)))
            if current_regime != MacroRegime.STRONG_BULL or drawdown_for_runner > 20.0:
                self._partial_close(timestamp, price, self.engine.config.tp2_close_fraction, "TP2_4R")
                self.trend_holding_allowed = False
                self.pending_runner_disabled = True
                self.runner_disabled_count += 1
                return
            runner_fraction = self._runner_fraction_for_drawdown(drawdown_for_runner)
            self.active_runner_fraction = runner_fraction
            close_fraction = max(0.0, 1.0 - self.engine.config.tp1_close_fraction - runner_fraction)
            self._partial_close(timestamp, price, close_fraction, "TP2_4R")
            self.runner_active = True
            self.runner_activation_count += 1
            self.runner_activation_timestamp = timestamp
            self.runner_highest_close = price
            self.entry_audit["runner_fraction"] = runner_fraction

    def _runner_exit_reason(self, signal: dict[str, Any]) -> str | None:
        daily_ema20 = _optional_float(signal.get("daily_ema20"))
        daily_ema50 = _optional_float(signal.get("daily_ema50"))
        if daily_ema20 is not None and daily_ema50 is not None and daily_ema20 < daily_ema50:
            return "RUNNER_EMA_EXIT"
        if str(signal.get("daily_macd") or signal.get("macd") or "").lower() == "bearish":
            return "RUNNER_MACD_EXIT"
        trailing_stop = self.engine.runner_trailing_stop(self.runner_highest_close)
        price = _optional_float(signal.get("price"))
        if trailing_stop is not None and price is not None and price <= trailing_stop:
            return "RUNNER_TRAILING_STOP"
        return None

    def _runner_fraction_for_drawdown(self, drawdown_pct: float) -> float:
        if drawdown_pct > 20.0:
            return 0.0
        if drawdown_pct > 15.0:
            return self.engine.config.runner_fraction * 0.5
        return self.engine.config.runner_fraction

    def _update_portfolio_drawdown(self, price: float) -> None:
        equity = self.cash + (self.position_size * float(price))
        self._record_snapshot_drawdown(equity)

    def _record_snapshot_drawdown(self, equity: float) -> None:
        self.portfolio_peak_equity = max(self.portfolio_peak_equity, float(equity))
        if self.portfolio_peak_equity <= 0:
            self.current_portfolio_drawdown_pct = 0.0
            return
        self.current_portfolio_drawdown_pct = max(
            0.0,
            ((self.portfolio_peak_equity - float(equity)) / self.portfolio_peak_equity) * 100,
        )

    def execution_summary(self, total_decisions: int) -> dict[str, Any]:
        summary = super().execution_summary(total_decisions)
        summary.update(
            {
                "runner_disabled_count": self.runner_disabled_count,
                "strong_bull_periods": self.regime_period_counts[MacroRegime.STRONG_BULL.value],
                "bull_periods": self.regime_period_counts[MacroRegime.BULL.value],
                "range_periods": self.regime_period_counts[MacroRegime.RANGE.value],
                "bear_periods": self.regime_period_counts[MacroRegime.BEAR.value],
            }
        )
        return summary

    def _reset_trend_holding_state(self) -> None:
        super()._reset_trend_holding_state()
        self.trend_holding_allowed = False
        self.pending_runner_disabled = False
        self.active_runner_fraction = self.engine.config.runner_fraction
        self.entry_portfolio_drawdown_pct = 0.0
