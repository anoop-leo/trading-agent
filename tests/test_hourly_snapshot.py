import unittest

from monitoring.hourly_snapshot import compute_hourly_update
from monitoring.monitoring_config import MonitoringConfig
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


def _raw_state(total: float, **overrides: float) -> PortfolioState:
    defaults = dict(
        total_value_usd=total, peak_value_usd=total,
        cash_usd=total * 0.6, core_usd=total * 0.26, growth_usd=total * 0.10, speculative_usd=total * 0.04,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


class ComputeHourlyUpdateTests(unittest.TestCase):
    def test_peak_comes_from_history_not_seeded_to_current(self) -> None:
        history = [{"timestamp": "2026-06-20T00:00:00+00:00", "total_value_usd": 350000.0, "prices": {}}]
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, {}, {}, "2026-06-22T00:00:00+00:00",
        )
        self.assertEqual(result.state.peak_value_usd, 350000.0)
        self.assertGreater(result.drawdown_pct, 0.0)

    def test_empty_history_falls_back_to_current_as_peak(self) -> None:
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), [], {}, {}, "2026-06-22T00:00:00+00:00",
        )
        self.assertEqual(result.state.peak_value_usd, 330000.0)
        self.assertEqual(result.drawdown_pct, 0.0)

    def test_history_point_is_appended_with_correct_shape(self) -> None:
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), [], {}, {"BTC": 64000.0}, "2026-06-22T00:00:00+00:00",
        )
        self.assertEqual(result.history_point["timestamp"], "2026-06-22T00:00:00+00:00")
        self.assertEqual(result.history_point["total_value_usd"], 330000.0)
        self.assertEqual(result.history_point["prices"], {"BTC": 64000.0})

    def test_drawdown_alert_fires_on_crossing(self) -> None:
        history = [{"timestamp": "2026-06-20T00:00:00+00:00", "total_value_usd": 440000.0, "prices": {}}]
        result, new_state = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, {}, {}, "2026-06-22T00:00:00+00:00",
        )
        self.assertTrue(any("drawdown" in a.lower() for a in result.alerts))
        self.assertIsNotNone(new_state["drawdown_tier"])

    def test_drawdown_alert_does_not_refire_next_hour_at_same_tier(self) -> None:
        history = [{"timestamp": "2026-06-20T00:00:00+00:00", "total_value_usd": 440000.0, "prices": {}}]
        _, alert_state = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, {}, {}, "2026-06-22T00:00:00+00:00",
        )
        result2, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, alert_state, {}, "2026-06-22T01:00:00+00:00",
        )
        self.assertFalse(any("drawdown" in a.lower() for a in result2.alerts))

    def test_speculative_near_cap_alert_fires(self) -> None:
        config = RiskEngineConfig(speculative_max_pct=12.0)
        state = _raw_state(330000.0, speculative_usd=36500.0)  # 11.06% -- close to 12% cap
        result, new_state = compute_hourly_update(
            state, config, MonitoringConfig(), [], {}, {}, "2026-06-22T00:00:00+00:00",
        )
        self.assertTrue(any("bucket" in a.lower() for a in result.alerts))
        self.assertTrue(new_state["speculative_near_cap"])

    def test_no_bucket_alert_when_well_under_cap(self) -> None:
        config = RiskEngineConfig(speculative_max_pct=12.0)
        state = _raw_state(330000.0, speculative_usd=11907.96)
        result, _ = compute_hourly_update(
            state, config, MonitoringConfig(), [], {}, {}, "2026-06-22T00:00:00+00:00",
        )
        self.assertFalse(any("bucket" in a.lower() for a in result.alerts))

    def test_position_move_alert_fires_against_24h_old_price(self) -> None:
        history = [
            {"timestamp": "2026-06-21T00:30:00+00:00", "total_value_usd": 330000.0, "prices": {"SUI": 0.60}},
        ]
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, {}, {"SUI": 0.71}, "2026-06-22T00:00:00+00:00",
        )
        self.assertTrue(any("SUI" in a for a in result.alerts))

    def test_position_move_alert_silent_when_no_matching_history_point(self) -> None:
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), [], {}, {"SUI": 0.71}, "2026-06-22T00:00:00+00:00",
        )
        self.assertFalse(any("SUI" in a for a in result.alerts))

    def test_position_move_below_threshold_is_silent(self) -> None:
        history = [{"timestamp": "2026-06-21T00:30:00+00:00", "total_value_usd": 330000.0, "prices": {"SUI": 0.68}}]
        result, _ = compute_hourly_update(
            _raw_state(330000.0), RiskEngineConfig(), MonitoringConfig(), history, {}, {"SUI": 0.71}, "2026-06-22T00:00:00+00:00",
        )
        self.assertFalse(any("SUI" in a for a in result.alerts))


if __name__ == "__main__":
    unittest.main()
