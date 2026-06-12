"""Market data loading for public OHLCV candles."""

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
BYBIT_KLINES_PATH = "/v5/market/kline"
BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
}


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
    """Convert exchange kline arrays into the internal OHLCV DataFrame shape."""

    if not raw_klines:
        raise DataLoadError("Exchange returned no kline rows.")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_klines):
        if len(item) < 6:
            raise DataLoadError(f"Exchange kline row {index} has fewer than six fields.")
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
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="raise"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return validate_ohlcv_frame(frame[list(OHLCV_COLUMNS)])


def _load_json_response(request: Request, opener: UrlOpener, timeout_seconds: float, source_name: str) -> Any:
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise DataLoadError(f"{source_name} request failed with HTTP {exc.code}.") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise DataLoadError(f"{source_name} request failed: {exc}.") from exc

    if status >= 400:
        raise DataLoadError(f"{source_name} request failed with HTTP {status}.")

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"{source_name} returned invalid JSON.") from exc


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

        payload = _load_json_response(request, self._opener, self.timeout_seconds, "Binance")
        if not isinstance(payload, list):
            raise DataLoadError("Binance returned an unexpected payload shape.")

        return normalize_klines(payload)


class BybitKlineProvider:
    """Public Bybit spot REST data provider for Phase 1 market data only."""

    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        timeout_seconds: float = 10.0,
        opener: UrlOpener = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        if interval not in BYBIT_INTERVALS:
            raise DataLoadError(f"Bybit does not support configured interval {interval!r}.")
        query = urlencode(
            {
                "category": "spot",
                "symbol": symbol.upper(),
                "interval": BYBIT_INTERVALS[interval],
                "limit": limit,
            }
        )
        request = Request(
            f"{self.base_url}{BYBIT_KLINES_PATH}?{query}",
            headers={"User-Agent": "trading-agent-phase1/0.1"},
            method="GET",
        )

        payload = _load_json_response(request, self._opener, self.timeout_seconds, "Bybit")
        if not isinstance(payload, dict):
            raise DataLoadError("Bybit returned an unexpected payload shape.")
        if payload.get("retCode") != 0:
            message = payload.get("retMsg", "unknown error")
            raise DataLoadError(f"Bybit request failed: {message}.")

        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise DataLoadError("Bybit returned an unexpected kline payload shape.")

        return normalize_klines(result["list"])
