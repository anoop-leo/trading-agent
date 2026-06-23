from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from risk.equity_history import (
    append_equity_history_point,
    compute_real_peak_value_usd,
    find_point_near,
    load_equity_history,
)


class EquityHistoryTests(unittest.TestCase):
    def test_load_missing_file_returns_empty_list(self) -> None:
        with TemporaryDirectory() as temp_dir:
            self.assertEqual(load_equity_history(Path(temp_dir) / "missing.jsonl"), [])

    def test_append_and_load_round_trips(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "equity_history.jsonl"
            append_equity_history_point({"timestamp": "2026-06-20T00:00:00+00:00", "total_value_usd": 300000.0}, path)
            append_equity_history_point({"timestamp": "2026-06-21T00:00:00+00:00", "total_value_usd": 310000.0}, path)

            points = load_equity_history(path)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[1]["total_value_usd"], 310000.0)

    def test_real_peak_is_max_of_history_and_current(self) -> None:
        history = [{"total_value_usd": 300000.0}, {"total_value_usd": 350000.0}, {"total_value_usd": 320000.0}]
        self.assertEqual(compute_real_peak_value_usd(history, 330000.0), 350000.0)

    def test_real_peak_falls_back_to_current_when_history_empty(self) -> None:
        self.assertEqual(compute_real_peak_value_usd([], 330000.0), 330000.0)

    def test_real_peak_uses_current_when_it_is_the_new_high(self) -> None:
        history = [{"total_value_usd": 300000.0}]
        self.assertEqual(compute_real_peak_value_usd(history, 340000.0), 340000.0)

    def test_find_point_near_returns_closest_within_tolerance(self) -> None:
        history = [
            {"timestamp": "2026-06-20T10:00:00+00:00", "total_value_usd": 300000.0},
            {"timestamp": "2026-06-21T09:30:00+00:00", "total_value_usd": 310000.0},
        ]
        found = find_point_near(history, "2026-06-21T10:00:00+00:00", tolerance_seconds=3600.0)
        self.assertEqual(found["total_value_usd"], 310000.0)

    def test_find_point_near_returns_none_outside_tolerance(self) -> None:
        history = [{"timestamp": "2026-06-18T10:00:00+00:00", "total_value_usd": 300000.0}]
        found = find_point_near(history, "2026-06-21T10:00:00+00:00", tolerance_seconds=3600.0)
        self.assertIsNone(found)

    def test_find_point_near_empty_history_returns_none(self) -> None:
        self.assertIsNone(find_point_near([], "2026-06-21T10:00:00+00:00"))


if __name__ == "__main__":
    unittest.main()
