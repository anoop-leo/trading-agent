from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from data.equity_data_adapter import (
    ProviderAttempt,
    load_equity_data,
    resample_ohlcv,
    validate_equity_frame,
    write_data_provider_diagnostics,
)
from trading_agent.data import DataLoadError


def frame(start: str = "2020-01-01", periods: int = 5, freq: str = "1h") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = [100.0 + index for index in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000.0] * periods,
        }
    )


class EquityDataAdapterTests(unittest.TestCase):
    def test_load_equity_data_falls_back_after_provider_failure(self) -> None:
        def fake_fetch(provider, *_args, **_kwargs):
            if provider == "yahoo":
                raise DataLoadError("HTTP 422")
            return frame(periods=30000)

        with patch("data.equity_data_adapter._fetch_provider", side_effect=fake_fetch):
            result = load_equity_data(
                "SPY",
                start="2020-01-01",
                end="2024-01-01",
                timeframes=("1h",),
                cache_dir=None,
                providers=("yahoo", "stooq"),
            )

        self.assertEqual(result.providers["1h"], "stooq")
        self.assertEqual(result.attempts[0].provider, "yahoo")
        self.assertEqual(result.attempts[0].status, "failed")
        self.assertEqual(result.attempts[1].provider, "stooq")
        self.assertEqual(result.attempts[1].status, "success")

    def test_generates_four_hour_from_one_hour(self) -> None:
        hourly = frame(periods=8)

        four_hour = resample_ohlcv(hourly, "4h")

        self.assertEqual(len(four_hour), 2)
        self.assertEqual(four_hour.iloc[0]["open"], 100.0)
        self.assertEqual(four_hour.iloc[0]["high"], 104.0)
        self.assertEqual(four_hour.iloc[1]["close"], 107.0)

    def test_validation_checks_required_quality_rules(self) -> None:
        clean = frame(start="2020-01-01", periods=365 * 4, freq="1D")

        validation = validate_equity_frame(symbol="SPY", timeframe="1d", frame=clean, provider="stooq")

        self.assertEqual(validation.validation, "passed")
        self.assertTrue(validation.minimum_3_years_history)
        broken = pd.concat([clean.iloc[[0]], clean.iloc[[0]]], ignore_index=True)
        broken_validation = validate_equity_frame(symbol="SPY", timeframe="1d", frame=broken, provider="stooq")
        self.assertEqual(broken_validation.validation, "failed")
        self.assertIn("duplicate timestamps found", broken_validation.errors)

    def test_writes_provider_diagnostics(self) -> None:
        with patch("data.equity_data_adapter._fetch_provider", return_value=frame(periods=365 * 4)):
            result = load_equity_data(
                "QQQ",
                start="2020-01-01",
                end="2024-01-01",
                timeframes=("1d",),
                cache_dir=None,
                providers=("stooq",),
            )

        with TemporaryDirectory() as temp_dir:
            path = write_data_provider_diagnostics(Path(temp_dir), [result])

            self.assertTrue(path.exists())
            self.assertIn("QQQ", path.read_text())
            self.assertIn("stooq", path.read_text())


if __name__ == "__main__":
    unittest.main()
