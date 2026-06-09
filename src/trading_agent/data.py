"""Market data loading for public Binance OHLCV candles."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from trading_agent.models import OHLCV_COLUMNS


BINANCE_KLINES_PATH = "/api/v3/klines"


class DataLoadError(RuntimeError):
    """Raised when market data cannot be fetched or normalized."""


UrlOpener = Callable[..., Any]


def validate_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the normalized OHLCV contract used by downstream modules."""

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataLoadError(f"OHLCV frame missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise DataLoadError("OHLCV frame is empty.")
    if frame[list(OHLCV_COLUMNS)].isnull().any().any():
        raise DataLoadError("OHLCV frame contains null values in required columns.")
    return frame


def normalize_klines(raw_klines: list[list[Any]]) -> pd.DataFrame:
    """Convert Binance kline arrays into the internal OHLCV DataFrame shape."""

    if not raw_klines:
        raise DataLoadError("Binance returned no kline rows.")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_klines):
        if len(item) < 6:
            raise DataLoadError(f"Binance kline row {index} has fewer than six fields.")
        rows.append(
            {
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return validate_ohlcv_frame(frame[list(OHLCV_COLUMNS)])


class BinanceKlineProvider:
    """Public Binance REST data provider for Phase 1 market data only."""

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout_seconds: float = 10.0,
        opener: UrlOpener = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        query = urlencode({"symbol": symbol.upper(), "interval": interval, "limit": limit})
        request = Request(
            f"{self.base_url}{BINANCE_KLINES_PATH}?{query}",
            headers={"User-Agent": "trading-agent-phase1/0.1"},
            method="GET",
        )

        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise DataLoadError(f"Binance request failed with HTTP {exc.code}.") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise DataLoadError(f"Binance request failed: {exc}.") from exc

        if status >= 400:
            raise DataLoadError(f"Binance request failed with HTTP {status}.")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DataLoadError("Binance returned invalid JSON.") from exc

        if not isinstance(payload, list):
            raise DataLoadError("Binance returned an unexpected payload shape.")

        return normalize_klines(payload)
