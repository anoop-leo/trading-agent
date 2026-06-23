import unittest

from monitoring.daily_digest import build_daily_digest_text
from monitoring.monitoring_config import MonitoringConfig
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


def _state(**overrides: float) -> PortfolioState:
    defaults = dict(
        total_value_usd=330784.86, peak_value_usd=330784.86,
        cash_usd=200096.45, core_usd=86993.45, growth_usd=31787.00, speculative_usd=11907.96,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


class BuildDailyDigestTextTests(unittest.TestCase):
    def test_includes_total_value(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("330,784.86", text)

    def test_no_history_reports_not_enough_data(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("not enough history", text)

    def test_24h_change_computed_when_yesterday_known(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, 320000.0, None)
        self.assertIn("+10,784.86", text)

    def test_drawdown_line_shows_tripped_when_over_threshold(self) -> None:
        state = _state(total_value_usd=240000.0, peak_value_usd=330000.0)
        text = build_daily_digest_text(state, RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("TRIPPED", text)

    def test_drawdown_line_silent_on_tripped_when_under_threshold(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertNotIn("TRIPPED", text)

    def test_btc_progress_included_when_quantity_given(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, 1.3476492)
        self.assertIn("BTC core", text)
        self.assertIn("1.3476492", text)

    def test_watchlist_section_included_and_sorted_descending(self) -> None:
        watchlist = {"scores": {"MRVL": 16, "MSFT": 59}, "bands": {"MRVL": "AVOID_ZONE", "MSFT": "NEUTRAL_WATCH_ZONE"}}
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), watchlist, None, None)
        msft_index = text.index("MSFT")
        mrvl_index = text.index("MRVL")
        self.assertLess(msft_index, mrvl_index)
        self.assertIn("11 pts away", text)

    def test_in_zone_status_shown_for_score_above_threshold(self) -> None:
        watchlist = {"scores": {"MSFT": 75}, "bands": {"MSFT": "ACCUMULATION_ZONE"}}
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), watchlist, None, None)
        self.assertIn("IN ZONE", text)

    def test_cro_blind_spot_note_always_present(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("CRO", text)

    def test_bucket_room_to_cap_shown_for_speculative_only(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("to cap", text)
        self.assertIn("no bucket cap", text)


if __name__ == "__main__":
    unittest.main()
