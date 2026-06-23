import unittest

from monitoring.daily_scan import compute_watchlist_alerts


class ComputeWatchlistAlertsTests(unittest.TestCase):
    def test_fires_for_symbol_crossing_into_zone(self) -> None:
        alerts, new_state = compute_watchlist_alerts({"MSFT": 72, "MRVL": 16}, {}, accumulation_zone_threshold=70)
        self.assertEqual(len(alerts), 1)
        self.assertIn("MSFT", alerts[0])
        self.assertTrue(new_state["watchlist_MSFT_in_zone"])
        self.assertFalse(new_state["watchlist_MRVL_in_zone"])

    def test_no_alerts_when_nothing_crosses(self) -> None:
        alerts, _ = compute_watchlist_alerts({"MSFT": 59, "MU": 58}, {}, accumulation_zone_threshold=70)
        self.assertEqual(alerts, [])

    def test_does_not_refire_for_symbol_already_in_zone(self) -> None:
        _, state = compute_watchlist_alerts({"MSFT": 72}, {}, accumulation_zone_threshold=70)
        alerts, _ = compute_watchlist_alerts({"MSFT": 75}, state, accumulation_zone_threshold=70)
        self.assertEqual(alerts, [])

    def test_multiple_symbols_can_fire_in_one_run(self) -> None:
        alerts, _ = compute_watchlist_alerts({"MSFT": 72, "NVDA": 80}, {}, accumulation_zone_threshold=70)
        self.assertEqual(len(alerts), 2)


if __name__ == "__main__":
    unittest.main()
