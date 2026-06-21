import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.false_avoid_analysis import (
    CORRECT_AVOID,
    FALSE_AVOID,
    FalseAvoidAnalysisConfig,
    evaluate_watch_long_candidate,
    run_false_avoid_analysis,
)


class FalseAvoidAnalysisTests(unittest.TestCase):
    def test_false_avoid_classified_when_price_rallies_without_prior_drop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_json(
                output_dir / "signal_journal.json",
                [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "symbol": "BTCUSDT",
                        "setup": "RANGE_BOUND",
                        "final_decision": "AVOID LONG",
                        "price": 100.0,
                        "evaluations": {
                            "plus_1_day": {"price_change_pct": 1.0},
                            "plus_3_day": {"price_change_pct": 3.0},
                        },
                    }
                ],
            )

            payload = run_false_avoid_analysis(FalseAvoidAnalysisConfig(output_dir=output_dir))
            signal = payload["signal_journal_enriched"]["signals"][0]

            self.assertEqual(signal["avoid_classification"], FALSE_AVOID)
            self.assertEqual(payload["false_avoid_analysis"]["metrics"]["false_avoid_count"], 1)

    def test_correct_avoid_classified_when_price_drops(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_json(
                output_dir / "signal_journal.json",
                [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "symbol": "BTCUSDT",
                        "setup": "BEAR_TREND",
                        "final_decision": "AVOID LONG",
                        "price": 100.0,
                        "evaluations": {
                            "plus_1_day": {"price_change_pct": -2.5},
                            "plus_3_day": {"price_change_pct": -1.0},
                        },
                    }
                ],
            )

            payload = run_false_avoid_analysis(FalseAvoidAnalysisConfig(output_dir=output_dir))
            signal = payload["signal_journal_enriched"]["signals"][0]

            self.assertEqual(signal["avoid_classification"], CORRECT_AVOID)
            self.assertEqual(payload["false_avoid_analysis"]["metrics"]["correct_avoid_count"], 1)

    def test_watch_long_trigger_requires_reversal_evidence(self) -> None:
        result = evaluate_watch_long_candidate(
            {
                "setup": "TREND_FOLLOWING",
                "price": 103.0,
                "ema20": 101.0,
                "ema50": 100.0,
                "rsi": 51.0,
                "four_hour_macd": "bullish",
                "support": 100.5,
                "resistance": 110.0,
                "market_regime": "BULL",
                "daily_setup": "TREND_FOLLOWING",
            }
        )

        self.assertTrue(result["triggered"])
        self.assertEqual(result["candidate_state"], "WATCH LONG")

    def test_watch_long_trigger_blocks_missing_indicator_snapshot(self) -> None:
        result = evaluate_watch_long_candidate(
            {
                "setup": "RANGE_BOUND",
                "price": 103.0,
                "market_regime": "BULL",
            }
        )

        self.assertFalse(result["triggered"])
        self.assertIn("ema20", result["missing_fields"])

    def test_run_writes_required_artifacts_and_reads_shadow_signals(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_csv(
                output_dir / "shadow_signals.csv",
                [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "symbol": "BTCUSDT",
                        "setup": "RANGE_BOUND",
                        "final_decision": "AVOID LONG",
                        "price": "100",
                        "rejection_reasons": "bearish_alignment|low_rr_ratio",
                    }
                ],
            )
            _write_csv(
                output_dir / "shadow_equity_curve.csv",
                [
                    {"timestamp": "2026-01-02T00:00:00+00:00", "price": "101"},
                    {"timestamp": "2026-01-04T00:00:00+00:00", "price": "103"},
                ],
            )
            _write_json(output_dir / "shadow_summary_30d.json", {"total_trades": 0, "rejected_signal_count": 1})

            payload = run_false_avoid_analysis(FalseAvoidAnalysisConfig(output_dir=output_dir))

            self.assertEqual(payload["false_avoid_analysis"]["metrics"]["total_avoid_signals"], 1)
            self.assertTrue((output_dir / "false_avoid_analysis.json").exists())
            self.assertTrue((output_dir / "watch_long_candidate_backtest.json").exists())
            self.assertTrue((output_dir / "signal_journal_enriched.json").exists())


def _write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
