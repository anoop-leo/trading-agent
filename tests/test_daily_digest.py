import unittest

from monitoring.daily_digest import build_crypto_accumulation_lines, build_daily_digest_text
from monitoring.monitoring_config import MonitoringConfig
from risk.portfolio_state import PortfolioState
from risk.risk_config import RiskEngineConfig


def _crypto() -> dict:
    return {
        "accumulation_zone_threshold": 70,
        "scores": {
            "BTC": {
                "symbol": "BTC", "role": "core", "bucket": "core", "held": True,
                "score": 80, "band": "GOOD_ACCUMULATION", "confidence": "HIGH",
                "distance_to_zone": 0, "in_zone": True,
                "drivers": {"mvrv": 2.0, "fear_greed": 40, "cycle_phase": "ACCUMULATION"},
                "cap_room_usd": 1000.0, "cap_note": "core bucket has $1,000 of room under caps",
            },
            "ADA": {
                "symbol": "ADA", "role": "held", "bucket": "speculative", "held": True,
                "score": 55, "band": "NEUTRAL", "confidence": "HIGH",
                "distance_to_zone": 15, "in_zone": False, "drivers": None,
            },
            "HYPE": {
                "symbol": "HYPE", "role": "watchlist", "bucket": "speculative", "held": False,
                "score": 72, "band": "GOOD", "confidence": "MEDIUM",
                "distance_to_zone": 0, "in_zone": True, "drivers": None,
                "cap_room_usd": 0.0, "cap_note": "speculative bucket cannot take a buy now -- at cap",
            },
        },
        "excluded": [{"symbol": "CRO", "reason": "CRO excluded (unfetchable)."}],
    }


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

    def test_bucket_room_to_cap_shown_for_speculative_only(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None)
        self.assertIn("to cap", text)
        self.assertIn("no bucket cap", text)

    def test_crypto_block_absent_when_no_crypto_data(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None, None)
        self.assertNotIn("Crypto accumulation", text)

    def test_crypto_block_included_when_crypto_data_present(self) -> None:
        text = build_daily_digest_text(_state(), RiskEngineConfig(), MonitoringConfig(), None, None, None, _crypto())
        self.assertIn("Crypto accumulation", text)
        self.assertIn("BTC", text)
        self.assertIn("MVRV", text)  # BTC drivers shown


class BuildCryptoAccumulationLinesTests(unittest.TestCase):
    def test_empty_when_no_scores(self) -> None:
        self.assertEqual(build_crypto_accumulation_lines(None, 70), [])
        self.assertEqual(build_crypto_accumulation_lines({"scores": {}}, 70), [])

    def test_btc_in_zone_shows_status_and_drivers(self) -> None:
        text = "\n".join(build_crypto_accumulation_lines(_crypto(), 70))
        self.assertIn("IN ZONE", text)
        self.assertIn("MVRV 2.0", text)
        self.assertIn("ACCUMULATION", text)

    def test_held_alt_shows_distance_to_zone(self) -> None:
        text = "\n".join(build_crypto_accumulation_lines(_crypto(), 70))
        self.assertIn("15 pts away", text)

    def test_watchlist_role_grouped_and_labelled(self) -> None:
        text = "\n".join(build_crypto_accumulation_lines(_crypto(), 70))
        self.assertIn("Watchlist (not held)", text)
        self.assertIn("HYPE", text)

    def test_excluded_blind_spot_note_rendered_when_present(self) -> None:
        # The excluded note is data-driven: it renders whatever the payload carries.
        text = "\n".join(build_crypto_accumulation_lines(_crypto(), 70))
        self.assertIn("CRO", text)

    def test_no_excluded_note_when_excluded_empty(self) -> None:
        payload = _crypto()
        payload["excluded"] = []
        text = "\n".join(build_crypto_accumulation_lines(payload, 70))
        self.assertNotIn("CRO", text)


if __name__ == "__main__":
    unittest.main()
