import unittest

import numpy as np
import pandas as pd

from trading_agent.indicators import (
    IndicatorError,
    add_indicators,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_volume_average,
)


class IndicatorTests(unittest.TestCase):
    def test_calculate_ema_has_expected_initial_nulls(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        ema = calculate_ema(series, span=3)

        self.assertTrue(pd.isna(ema.iloc[1]))
        self.assertFalse(pd.isna(ema.iloc[2]))

    def test_calculate_rsi_handles_flat_series(self) -> None:
        rsi = calculate_rsi(pd.Series([10.0] * 20), window=14)

        self.assertEqual(rsi.iloc[-1], 50)

    def test_calculate_rsi_detects_uptrend(self) -> None:
        rsi = calculate_rsi(pd.Series(np.arange(1.0, 40.0)), window=14)

        self.assertGreater(rsi.iloc[-1], 99)

    def test_calculate_macd_returns_components(self) -> None:
        macd = calculate_macd(pd.Series(np.arange(1.0, 80.0)))

        self.assertEqual(list(macd.columns), ["macd", "macd_signal", "macd_histogram"])
        self.assertAlmostEqual(macd.iloc[-1]["macd_histogram"], macd.iloc[-1]["macd"] - macd.iloc[-1]["macd_signal"])

    def test_calculate_bollinger_bands_on_constant_series(self) -> None:
        bands = calculate_bollinger_bands(pd.Series([5.0] * 25), window=20)

        self.assertEqual(bands.iloc[-1]["bb_middle"], 5.0)
        self.assertEqual(bands.iloc[-1]["bb_upper"], 5.0)
        self.assertEqual(bands.iloc[-1]["bb_lower"], 5.0)

    def test_calculate_volume_average(self) -> None:
        volume_average = calculate_volume_average(pd.Series([1.0, 2.0, 3.0]), window=2)

        self.assertTrue(pd.isna(volume_average.iloc[0]))
        self.assertEqual(volume_average.iloc[-1], 2.5)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(IndicatorError):
            calculate_ema(pd.Series([1.0, 2.0]), span=0)

    def test_add_indicators_preserves_ohlcv_and_adds_columns(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.arange(220.0),
                "high": np.arange(221.0, 441.0),
                "low": np.arange(220.0) - 1,
                "close": np.arange(220.0) + 0.5,
                "volume": np.full(220, 100.0),
            }
        )

        enriched = add_indicators(frame)

        self.assertIn("ema_20", enriched.columns)
        self.assertIn("rsi_14", enriched.columns)
        self.assertIn("macd_histogram", enriched.columns)
        self.assertIn("volume_ma_20", enriched.columns)
        self.assertEqual(enriched.iloc[-1]["timestamp"], timestamps[-1])


if __name__ == "__main__":
    unittest.main()
