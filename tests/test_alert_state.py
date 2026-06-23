from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from risk.alert_state import (
    evaluate_btc_target_alert,
    evaluate_bucket_cap_alert,
    evaluate_drawdown_alert,
    evaluate_position_move_alert,
    evaluate_watchlist_accumulation_alert,
    load_alert_state,
    save_alert_state,
)


class AlertStatePersistenceTests(unittest.TestCase):
    def test_load_missing_file_returns_empty_dict(self) -> None:
        with TemporaryDirectory() as temp_dir:
            self.assertEqual(load_alert_state(Path(temp_dir) / "missing.json"), {})

    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "alert_state.json"
            save_alert_state({"drawdown_tier": "drawdown_10"}, path)
            self.assertEqual(load_alert_state(path), {"drawdown_tier": "drawdown_10"})


class DrawdownAlertTests(unittest.TestCase):
    def test_fires_on_first_crossing_into_10_pct(self) -> None:
        message, new_state = evaluate_drawdown_alert({}, 11.0)
        self.assertIsNotNone(message)
        self.assertIn("-10%", message)
        self.assertEqual(new_state["drawdown_tier"], "drawdown_10")

    def test_does_not_refire_while_remaining_in_same_tier(self) -> None:
        _, state = evaluate_drawdown_alert({}, 11.0)
        message, _ = evaluate_drawdown_alert(state, 12.0)
        self.assertIsNone(message)

    def test_fires_again_on_crossing_up_to_a_worse_tier(self) -> None:
        _, state = evaluate_drawdown_alert({}, 11.0)
        message, new_state = evaluate_drawdown_alert(state, 21.0)
        self.assertIsNotNone(message)
        self.assertIn("-20%", message)
        self.assertEqual(new_state["drawdown_tier"], "drawdown_20")

    def test_tripped_tier_at_25_percent(self) -> None:
        message, new_state = evaluate_drawdown_alert({}, 26.0)
        self.assertIn("TRIPPED", message)
        self.assertEqual(new_state["drawdown_tier"], "drawdown_25_tripped")

    def test_approaching_tier_between_23_and_25(self) -> None:
        message, new_state = evaluate_drawdown_alert({}, 23.5)
        self.assertIn("approaching", message)
        self.assertEqual(new_state["drawdown_tier"], "drawdown_25_approaching")

    def test_recovering_then_recrossing_fires_again(self) -> None:
        _, state = evaluate_drawdown_alert({}, 11.0)  # cross into 10
        _, state = evaluate_drawdown_alert(state, 5.0)  # recover to none
        self.assertIsNone(state["drawdown_tier"])
        message, _ = evaluate_drawdown_alert(state, 11.0)  # cross into 10 again
        self.assertIsNotNone(message)

    def test_no_alert_below_any_threshold(self) -> None:
        message, new_state = evaluate_drawdown_alert({}, 3.0)
        self.assertIsNone(message)
        self.assertIsNone(new_state["drawdown_tier"])


class BucketCapAlertTests(unittest.TestCase):
    def test_fires_on_crossing_90_pct_of_cap(self) -> None:
        message, new_state = evaluate_bucket_cap_alert({}, "speculative", 36000.0, 39694.18)
        self.assertIsNotNone(message)
        self.assertTrue(new_state["speculative_near_cap"])

    def test_does_not_refire_while_remaining_near_cap(self) -> None:
        _, state = evaluate_bucket_cap_alert({}, "speculative", 36000.0, 39694.18)
        message, _ = evaluate_bucket_cap_alert(state, "speculative", 37000.0, 39694.18)
        self.assertIsNone(message)

    def test_no_alert_below_90_pct(self) -> None:
        message, new_state = evaluate_bucket_cap_alert({}, "speculative", 11907.96, 39694.18)
        self.assertIsNone(message)
        self.assertFalse(new_state["speculative_near_cap"])

    def test_refires_after_dropping_back_below_then_crossing_again(self) -> None:
        _, state = evaluate_bucket_cap_alert({}, "speculative", 36000.0, 39694.18)
        _, state = evaluate_bucket_cap_alert(state, "speculative", 20000.0, 39694.18)
        message, _ = evaluate_bucket_cap_alert(state, "speculative", 36000.0, 39694.18)
        self.assertIsNotNone(message)


class WatchlistAccumulationAlertTests(unittest.TestCase):
    def test_fires_on_crossing_into_accumulation_zone(self) -> None:
        message, new_state = evaluate_watchlist_accumulation_alert({}, "MSFT", 72)
        self.assertIsNotNone(message)
        self.assertIn("MSFT", message)
        self.assertTrue(new_state["watchlist_MSFT_in_zone"])

    def test_no_alert_below_threshold(self) -> None:
        message, _ = evaluate_watchlist_accumulation_alert({}, "MRVL", 16)
        self.assertIsNone(message)

    def test_does_not_refire_while_remaining_in_zone(self) -> None:
        _, state = evaluate_watchlist_accumulation_alert({}, "MSFT", 72)
        message, _ = evaluate_watchlist_accumulation_alert(state, "MSFT", 75)
        self.assertIsNone(message)

    def test_independent_per_symbol(self) -> None:
        _, state = evaluate_watchlist_accumulation_alert({}, "MSFT", 72)
        message, _ = evaluate_watchlist_accumulation_alert(state, "NVDA", 80)
        self.assertIsNotNone(message)


class BtcTargetAlertTests(unittest.TestCase):
    def test_fires_on_reaching_target(self) -> None:
        message, new_state = evaluate_btc_target_alert({}, 2.0, 2.0)
        self.assertIsNotNone(message)
        self.assertTrue(new_state["btc_target_reached"])

    def test_no_alert_below_target(self) -> None:
        message, _ = evaluate_btc_target_alert({}, 1.5, 2.0)
        self.assertIsNone(message)

    def test_does_not_refire_once_reached(self) -> None:
        _, state = evaluate_btc_target_alert({}, 2.0, 2.0)
        message, _ = evaluate_btc_target_alert(state, 2.1, 2.0)
        self.assertIsNone(message)


class PositionMoveAlertTests(unittest.TestCase):
    def test_fires_when_move_exceeds_threshold(self) -> None:
        message, new_state = evaluate_position_move_alert({}, "SUI", 18.0, 15.0, "2026-06-22")
        self.assertIsNotNone(message)
        self.assertIn("up", message)
        self.assertEqual(new_state["position_move_SUI_last_alert_date"], "2026-06-22")

    def test_fires_for_negative_moves_too(self) -> None:
        message, _ = evaluate_position_move_alert({}, "SUI", -18.0, 15.0, "2026-06-22")
        self.assertIn("down", message)

    def test_no_alert_below_threshold(self) -> None:
        message, _ = evaluate_position_move_alert({}, "SUI", 5.0, 15.0, "2026-06-22")
        self.assertIsNone(message)

    def test_does_not_refire_same_day(self) -> None:
        _, state = evaluate_position_move_alert({}, "SUI", 18.0, 15.0, "2026-06-22")
        message, _ = evaluate_position_move_alert(state, "SUI", 19.0, 15.0, "2026-06-22")
        self.assertIsNone(message)

    def test_refires_next_day(self) -> None:
        _, state = evaluate_position_move_alert({}, "SUI", 18.0, 15.0, "2026-06-22")
        message, _ = evaluate_position_move_alert(state, "SUI", 18.0, 15.0, "2026-06-23")
        self.assertIsNotNone(message)


if __name__ == "__main__":
    unittest.main()
