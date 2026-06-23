import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import URLError

from data.equity_fundamentals_provider import EquityFundamentalsProvider, fetch_equity_fundamentals


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class EquityFundamentalsProviderTests(unittest.TestCase):
    def test_paces_overview_and_cash_flow_calls(self) -> None:
        sleeps: list[float] = []

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            del timeout
            if "function=OVERVIEW" in request.full_url:
                return FakeResponse({"Symbol": "AAPL", "MarketCapitalization": "100"})
            return FakeResponse({"annualReports": [{"operatingCashflow": "10", "capitalExpenditures": "1"}]})

        fetch_equity_fundamentals(
            "AAPL",
            provider=EquityFundamentalsProvider(
                opener=opener,
                environ={"ALPHA_VANTAGE_API_KEY": "test-key"},
                request_interval_seconds=12.0,
                sleep_fn=sleeps.append,
            ),
        )

        self.assertEqual(sleeps, [12.0])

    def test_zero_interval_skips_sleep(self) -> None:
        sleeps: list[float] = []

        def opener(request: object, timeout: float = 0) -> FakeResponse:
            del timeout
            if "function=OVERVIEW" in request.full_url:
                return FakeResponse({"Symbol": "AAPL", "MarketCapitalization": "100"})
            return FakeResponse({"annualReports": [{"operatingCashflow": "10", "capitalExpenditures": "1"}]})

        fetch_equity_fundamentals(
            "AAPL",
            provider=EquityFundamentalsProvider(
                opener=opener,
                environ={"ALPHA_VANTAGE_API_KEY": "test-key"},
                request_interval_seconds=0,
                sleep_fn=sleeps.append,
            ),
        )

        self.assertEqual(sleeps, [])

    def test_alpha_vantage_overview_and_cash_flow_are_combined(self) -> None:
        def opener(request: object, timeout: float = 0) -> FakeResponse:
            del timeout
            if "function=OVERVIEW" in request.full_url:
                return FakeResponse(
                    {
                        "Symbol": "AAPL",
                        "AssetType": "Common Stock",
                        "Sector": "TECHNOLOGY",
                        "TrailingPE": "30.5",
                        "ForwardPE": "27.1",
                        "PriceToBookRatio": "45.2",
                        "PEGRatio": "2.1",
                        "ReturnOnEquityTTM": "1.5",
                        "QuarterlyRevenueGrowthYOY": "0.08",
                        "QuarterlyEarningsGrowthYOY": "0.12",
                        "DividendYield": "0.005",
                        "Beta": "1.2",
                        "MarketCapitalization": "3000000000000",
                    }
                )
            self.assertIn("function=CASH_FLOW", request.full_url)
            return FakeResponse(
                {"annualReports": [{"operatingCashflow": "110000000000", "capitalExpenditures": "10000000000"}]}
            )

        payload = fetch_equity_fundamentals(
            "AAPL",
            provider=EquityFundamentalsProvider(
                opener=opener, environ={"ALPHA_VANTAGE_API_KEY": "test-key"}, request_interval_seconds=0,
            ),
        )

        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["pe_forward"], 27.1)
        self.assertEqual(payload["price_to_book"], 45.2)
        self.assertEqual(payload["fcf_yield_pct"], round((110_000_000_000 - 10_000_000_000) / 3_000_000_000_000 * 100, 4))
        self.assertFalse(payload["missing"])
        self.assertEqual(payload["source"], "alpha_vantage")

    def test_no_api_key_falls_back_to_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "fundamentals_aapl.json"
            cache_path.write_text(json.dumps({"pe_trailing": 28.0, "source": "manual"}))

            payload = fetch_equity_fundamentals(
                "AAPL",
                provider=EquityFundamentalsProvider(
                    opener=_failing_opener, cache_dir=Path(temp_dir), environ={},
                ),
            )

        self.assertEqual(payload["pe_trailing"], 28.0)
        self.assertEqual(payload["source"], "cache/manual")
        self.assertFalse(payload["missing"])

    def test_missing_everything_returns_neutral_fallback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            payload = fetch_equity_fundamentals(
                "AAPL",
                provider=EquityFundamentalsProvider(
                    opener=_failing_opener, cache_dir=Path(temp_dir), environ={},
                ),
            )

        self.assertIsNone(payload["pe_trailing"])
        self.assertTrue(payload["missing"])
        self.assertTrue(payload["is_fallback"])
        self.assertEqual(payload["source"], "neutral_fallback")

    def test_offline_skips_network_and_uses_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "fundamentals_msft.json"
            cache_path.write_text(json.dumps({"pe_trailing": 32.0}))

            payload = fetch_equity_fundamentals(
                "MSFT",
                offline=True,
                provider=EquityFundamentalsProvider(
                    opener=_failing_opener, cache_dir=Path(temp_dir), environ={"ALPHA_VANTAGE_API_KEY": "key"},
                ),
            )

        self.assertEqual(payload["pe_trailing"], 32.0)


def _failing_opener(request: object, timeout: float = 0) -> FakeResponse:
    del request, timeout
    raise URLError("provider unavailable")


if __name__ == "__main__":
    unittest.main()
