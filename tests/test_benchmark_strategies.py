import unittest

import numpy as np
import pandas as pd

from backtesting.benchmark_strategies import run_buy_and_hold_benchmark, run_daily_ema200_benchmark


def ohlcv_frame(close: list[float], freq: str = "1D") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(close), freq=freq, tz="UTC")
    close_array = np.array(close, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close_array,
            "high": close_array + 1.0,
            "low": close_array - 1.0,
            "close": close_array,
            "volume": np.full(len(close), 1000.0),
        }
    )


class BenchmarkStrategiesTests(unittest.TestCase):
    def test_buy_and_hold_tracks_underlying_return(self) -> None:
        result = run_buy_and_hold_benchmark("BTCUSDT", ohlcv_frame([100.0, 110.0, 120.0]), 10000.0)

        self.assertEqual(result.name, "buy_and_hold")
        self.assertEqual(result.metrics["total_return_pct"], 20.0)
        self.assertEqual(result.metrics["max_drawdown_pct"], 0.0)
        self.assertEqual(result.trades, [])

    def test_daily_ema200_enters_above_ema_and_exits_below_ema(self) -> None:
        close = [100.0] * 210 + [120.0] * 20 + [80.0] * 20

        result = run_daily_ema200_benchmark("BTCUSDT", ohlcv_frame(close), 10000.0)

        self.assertEqual(result.name, "daily_ema200")
        self.assertGreaterEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0]["exit_reason"], "EMA200_EXIT")
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("profit_factor", result.metrics)

    def test_daily_ema200_uses_warmup_before_requested_start(self) -> None:
        close = [100.0] * 210 + [120.0] * 20 + [80.0] * 20

        result = run_daily_ema200_benchmark(
            "BTCUSDT",
            ohlcv_frame(close),
            10000.0,
            start="2024-08-01",
            end="2024-09-01",
        )

        self.assertGreaterEqual(result.start_date, "2024-08-01")
        self.assertIn("total_return_pct", result.metrics)


if __name__ == "__main__":
    unittest.main()
