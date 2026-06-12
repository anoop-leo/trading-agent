import unittest

import pandas as pd

from risk.structure_stop_engine import (
    StructureStopEngine,
    calculate_atr,
    detect_support_zone,
    find_recent_swing_low,
)


def frame_from_lows(lows: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(lows), freq="h", tz="UTC")
    low = pd.Series(lows, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": low + 1.0,
            "high": low + 4.0,
            "low": low,
            "close": low + 2.0,
            "volume": 1000.0,
        }
    )


class StructureStopEngineTests(unittest.TestCase):
    def test_calculates_atr_from_true_range(self) -> None:
        frame = frame_from_lows([96.0, 97.0, 98.0, 97.0])

        self.assertEqual(calculate_atr(frame, period=3), 4.0)

    def test_finds_recent_swing_low_surrounded_by_higher_lows(self) -> None:
        frame = frame_from_lows([10.0, 9.0, 8.0, 9.0, 10.0, 7.0, 8.0, 9.0])

        self.assertEqual(find_recent_swing_low(frame, lookback=8)["swing_low"], 7.0)

    def test_detects_clustered_support_zone(self) -> None:
        frame = frame_from_lows(
            [
                104.0,
                102.0,
                95.0,
                102.0,
                104.0,
                103.0,
                101.0,
                95.2,
                101.0,
                103.0,
                104.0,
                102.0,
                94.9,
                102.0,
                104.0,
                103.0,
                101.0,
                95.1,
                101.0,
                103.0,
            ]
        )

        zone = detect_support_zone(frame, lookback=20)

        self.assertIsNotNone(zone)
        self.assertEqual(zone.support_low, 94.9)
        self.assertEqual(zone.support_high, 95.2)
        self.assertEqual(zone.strength, 8)

    def test_atr_stop_floor_widens_too_close_swing_stop(self) -> None:
        engine = StructureStopEngine("swing_low", swing_lookback=5, atr_period=5)
        frame = frame_from_lows([100.0, 99.0, 98.0, 99.0, 100.0])

        result = engine.evaluate(frame, entry_price=100.0, fixed_stop=99.0, support=98.0)

        self.assertEqual(result.stop_type, "SWING_LOW")
        self.assertEqual(result.stop_price, 94.0)
        self.assertTrue(result.atr_floor_applied)
        self.assertEqual(result.stop_distance_atr, 1.5)

    def test_support_zone_stop_has_priority_over_swing_low_and_atr(self) -> None:
        engine = StructureStopEngine("support_zone", swing_lookback=20, support_lookback=20, atr_period=5)
        frame = frame_from_lows(
            [
                104.0,
                102.0,
                95.0,
                102.0,
                104.0,
                103.0,
                101.0,
                95.2,
                101.0,
                103.0,
                104.0,
                102.0,
                94.9,
                102.0,
                104.0,
                103.0,
                101.0,
                95.1,
                101.0,
                103.0,
            ]
        )

        result = engine.evaluate(frame, entry_price=110.0, fixed_stop=109.0)

        self.assertEqual(result.stop_type, "SUPPORT_ZONE")
        self.assertEqual(result.support_zone.support_low, 94.9)
        expected_support_stop = round(result.support_zone.support_low - (0.5 * calculate_atr(frame, period=5)), 4)
        self.assertEqual(result.stop_price, expected_support_stop)
        self.assertEqual(result.candidates["swing_low"]["stop_type"], "SWING_LOW")

    def test_support_zone_request_falls_back_to_swing_low_when_no_zone_exists(self) -> None:
        engine = StructureStopEngine("support_zone", swing_lookback=5, support_lookback=5, atr_period=5)
        frame = frame_from_lows([100.0, 99.0, 96.0, 99.0, 100.0])

        result = engine.evaluate(frame, entry_price=110.0)

        self.assertEqual(result.stop_type, "SWING_LOW")
        expected_swing_stop = round(96.0 - (0.5 * calculate_atr(frame, period=5)), 4)
        self.assertEqual(result.stop_price, expected_swing_stop)

    def test_atr_stop_selection_uses_one_and_half_atr(self) -> None:
        engine = StructureStopEngine("atr", swing_lookback=5, atr_period=5)
        frame = frame_from_lows([90.0, 91.0, 92.0, 93.0, 94.0])

        result = engine.evaluate(frame, entry_price=100.0, fixed_stop=99.0, support=90.0)

        self.assertEqual(result.stop_type, "ATR")
        self.assertEqual(result.stop_price, 94.0)
        self.assertEqual(result.candidates["fixed"]["stop_price"], 99.0)
        self.assertIsNone(result.candidates["support_zone"]["stop_price"])

    def test_fixed_stop_does_not_fall_back_to_atr_when_missing(self) -> None:
        engine = StructureStopEngine("fixed", swing_lookback=5, atr_period=5)
        frame = frame_from_lows([90.0, 91.0, 92.0, 93.0, 94.0])

        result = engine.evaluate(frame, entry_price=100.0, fixed_stop=None, support=90.0)

        self.assertEqual(result.stop_type, "FIXED")
        self.assertIsNone(result.stop_price)
        self.assertEqual(result.candidates["atr"]["stop_price"], 94.0)


if __name__ == "__main__":
    unittest.main()
