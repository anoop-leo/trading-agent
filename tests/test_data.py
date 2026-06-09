import json
from urllib.parse import parse_qs, urlparse
import unittest

import pandas as pd

from trading_agent.data import BinanceKlineProvider, DataLoadError, normalize_klines


RAW_KLINES = [
    [1710003600000, "101.0", "105.0", "99.0", "104.0", "12.5", 0, 0, 0, 0, 0, 0],
    [1710000000000, "100.0", "102.0", "98.0", "101.0", "10.0", 0, 0, 0, 0, 0, 0],
]


class FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class DataTests(unittest.TestCase):
    def test_normalize_klines_sorts_and_converts_ohlcv(self) -> None:
        frame = normalize_klines(RAW_KLINES)

        self.assertEqual(list(frame.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["timestamp"]))
        self.assertEqual(frame.iloc[0]["close"], 101.0)
        self.assertEqual(frame.iloc[1]["volume"], 12.5)

    def test_normalize_klines_rejects_empty_payload(self) -> None:
        with self.assertRaises(DataLoadError):
            normalize_klines([])

    def test_provider_calls_public_klines_endpoint(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse(RAW_KLINES)

        provider = BinanceKlineProvider(base_url="https://example.test", timeout_seconds=3.0, opener=opener)
        frame = provider.fetch_ohlcv("btcusdt", "1h", 500)

        request, timeout = calls[0]
        parsed = urlparse(request.full_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/api/v3/klines")
        self.assertEqual(query["symbol"], ["BTCUSDT"])
        self.assertEqual(query["interval"], ["1h"])
        self.assertEqual(query["limit"], ["500"])
        self.assertEqual(timeout, 3.0)
        self.assertEqual(len(frame), 2)

    def test_provider_rejects_non_list_payload(self) -> None:
        provider = BinanceKlineProvider(opener=lambda *_args, **_kwargs: FakeResponse({"error": "bad"}))

        with self.assertRaises(DataLoadError):
            provider.fetch_ohlcv("BTCUSDT", "1h", 500)


if __name__ == "__main__":
    unittest.main()
