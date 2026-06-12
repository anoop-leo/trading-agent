from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trading_agent.journal import (
    JournalError,
    build_journal_entry,
    evaluate_journal_entries,
    load_journal,
    update_signal_journal,
)


def payload(**overrides) -> dict:
    values = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "symbol": "BTCUSDT",
        "setup": "PULLBACK",
        "final_decision": "BUY WATCH",
        "price": 100.0,
    }
    values.update(overrides)
    return values


def price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                    "2024-01-08T00:00:00Z",
                    "2024-01-31T00:00:00Z",
                ]
            ),
            "close": [100.0, 103.0, 106.0, 97.0, 120.0],
        }
    )


class JournalTests(unittest.TestCase):
    def test_build_journal_entry_uses_required_signal_fields(self) -> None:
        entry = build_journal_entry(payload())

        self.assertEqual(entry["timestamp"], "2024-01-01T00:00:00+00:00")
        self.assertEqual(entry["symbol"], "BTCUSDT")
        self.assertEqual(entry["setup"], "PULLBACK")
        self.assertEqual(entry["final_decision"], "BUY WATCH")
        self.assertEqual(entry["price"], 100)
        self.assertIn("plus_1_day", entry["evaluations"])

    def test_build_journal_entry_rejects_missing_fields(self) -> None:
        with self.assertRaises(JournalError):
            build_journal_entry({"symbol": "BTCUSDT"})

    def test_evaluate_journal_entries_scores_due_horizons(self) -> None:
        entry = build_journal_entry(payload())
        count = evaluate_journal_entries([entry], price_frame(), "BTCUSDT")

        self.assertEqual(count, 4)
        self.assertEqual(entry["evaluations"]["plus_1_day"]["price"], 103)
        self.assertEqual(entry["evaluations"]["plus_1_day"]["price_change_pct"], 3.0)
        self.assertEqual(entry["evaluations"]["plus_1_day"]["outcome"], "FAVORABLE")
        self.assertEqual(entry["evaluations"]["plus_7_day"]["outcome"], "UNFAVORABLE")
        self.assertEqual(entry["evaluations"]["plus_30_day"]["price"], 120)

    def test_defensive_decision_favors_lower_future_price(self) -> None:
        entry = build_journal_entry(payload(final_decision="AVOID LONG"))
        evaluate_journal_entries([entry], price_frame(), "BTCUSDT")

        self.assertEqual(entry["evaluations"]["plus_7_day"]["outcome"], "FAVORABLE")
        self.assertEqual(entry["evaluations"]["plus_1_day"]["outcome"], "UNFAVORABLE")

    def test_future_horizon_remains_pending_without_enough_data(self) -> None:
        entry = build_journal_entry(payload())
        short_frame = price_frame().iloc[:2]

        count = evaluate_journal_entries([entry], short_frame, "BTCUSDT")

        self.assertEqual(count, 1)
        self.assertIsNone(entry["evaluations"]["plus_3_day"])

    def test_update_signal_journal_writes_and_deduplicates_entries(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            path, first_status = update_signal_journal(payload(), price_frame(), output_dir)
            _same_path, second_status = update_signal_journal(payload(), price_frame(), output_dir)
            entries = load_journal(path)

            self.assertEqual(path.name, "signal_journal.json")
            self.assertTrue(first_status["inserted"])
            self.assertFalse(second_status["inserted"])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["evaluations"]["plus_3_day"]["price"], 106)


if __name__ == "__main__":
    unittest.main()
