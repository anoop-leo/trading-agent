"""Equity OHLCV adapter with deterministic provider fallback diagnostics."""

from __future__ import annotations

import csv
import hashlib
import http.cookiejar
import io
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import pandas as pd

from trading_agent.data import DataLoadError, _load_json_response
from trading_agent.models import OHLCV_COLUMNS


YAHOO_BASE_URL = "https://query1.finance.yahoo.com"
STOOQ_BASE_URL = "https://stooq.com"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co"
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
EQUITY_PROVIDERS = ("yahoo", "stooq", "alpha_vantage", "twelve_data")
YAHOO_INTERVALS = {"1h": "60m", "1d": "1d"}
STOOQ_INTERVALS = {"1h": "60", "1d": "d"}
ALPHA_INTERVALS = {"1h": "60min"}
TWELVE_INTERVALS = {"1h": "1h", "1d": "1day"}
MIN_HISTORY_DAYS = 365 * 3


class EquityDataAdapterError(DataLoadError):
    """Raised when all equity providers fail while preserving diagnostics."""

    def __init__(self, message: str, attempts: list["ProviderAttempt"]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class ProviderAttempt:
    symbol: str
    timeframe: str
    provider: str
    status: str
    rows: int
    error: str | None = None


@dataclass(frozen=True)
class TimeframeValidation:
    symbol: str
    timeframe: str
    rows: int
    provider: str
    validation: str
    start: str | None
    end: str | None
    duplicate_timestamps: int
    missing_timestamps: int
    sorted_ascending: bool
    minimum_3_years_history: bool
    errors: list[str]


@dataclass(frozen=True)
class EquityDataResult:
    symbol: str
    frames: dict[str, pd.DataFrame]
    providers: dict[str, str]
    validations: dict[str, TimeframeValidation]
    attempts: list[ProviderAttempt]

    def diagnostics_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "providers": self.providers,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "validation": {
                timeframe: asdict(validation)
                for timeframe, validation in self.validations.items()
            },
        }


def load_equity_data(
    symbol: str,
    start: str = "2018-01-01",
    end: str = "latest",
    timeframes: tuple[str, ...] = ("1h", "4h", "1d"),
    cache_dir: Path | None = Path("data/cache"),
    refresh_cache: bool = False,
    timeout_seconds: float = 10.0,
    providers: tuple[str, ...] = EQUITY_PROVIDERS,
) -> EquityDataResult:
    """Load equity OHLCV from provider fallback chain and validate each timeframe."""

    normalized_symbol = symbol.upper()
    requested = _ordered_timeframes(timeframes)
    download_timeframes = tuple(timeframe for timeframe in requested if timeframe != "4h")
    if "4h" in requested and "1h" not in download_timeframes:
        download_timeframes = ("1h", *download_timeframes)

    frames: dict[str, pd.DataFrame] = {}
    selected_providers: dict[str, str] = {}
    attempts: list[ProviderAttempt] = []
    for timeframe in download_timeframes:
        frame, provider, timeframe_attempts = _load_or_download_timeframe(
            symbol=normalized_symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            timeout_seconds=timeout_seconds,
            providers=providers,
        )
        frames[timeframe] = frame
        selected_providers[timeframe] = provider
        attempts.extend(timeframe_attempts)

    if "4h" in requested:
        if "1h" not in frames or frames["1h"].empty:
            raise DataLoadError(f"{normalized_symbol} 4h cannot be generated without 1h data.")
        frames["4h"] = resample_ohlcv(frames["1h"], "4h")
        selected_providers["4h"] = f"generated_from_1h:{selected_providers['1h']}"

    filtered_frames = {
        timeframe: _filter_date_range(frames[timeframe], start, end)
        for timeframe in requested
        if timeframe in frames
    }
    validations = {
        timeframe: validate_equity_frame(
            symbol=normalized_symbol,
            timeframe=timeframe,
            frame=filtered_frames[timeframe],
            provider=selected_providers.get(timeframe, "unknown"),
        )
        for timeframe in requested
        if timeframe in filtered_frames
    }
    return EquityDataResult(
        symbol=normalized_symbol,
        frames=filtered_frames,
        providers={timeframe: selected_providers[timeframe] for timeframe in filtered_frames},
        validations=validations,
        attempts=attempts,
    )


def download_equity_timeframe(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    timeout_seconds: float = 10.0,
    providers: tuple[str, ...] = EQUITY_PROVIDERS,
) -> pd.DataFrame:
    """Download one equity timeframe without reading or writing cache."""

    if interval == "4h":
        hourly = download_equity_timeframe(symbol, "1h", start, end, timeout_seconds, providers)
        return resample_ohlcv(hourly, "4h")
    frame, _provider, _attempts = _download_with_fallback(
        symbol=symbol.upper(),
        timeframe=interval,
        start=start,
        end=end,
        timeout_seconds=timeout_seconds,
        providers=providers,
    )
    return frame


def validate_equity_frame(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    provider: str,
) -> TimeframeValidation:
    errors: list[str] = []
    if frame.empty:
        errors.append("OHLCV frame is empty")
    missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"missing columns: {', '.join(missing_columns)}")

    if frame.empty or missing_columns:
        return TimeframeValidation(
            symbol=symbol,
            timeframe=timeframe,
            rows=int(len(frame)),
            provider=provider,
            validation="failed",
            start=None,
            end=None,
            duplicate_timestamps=0,
            missing_timestamps=0,
            sorted_ascending=False,
            minimum_3_years_history=False,
            errors=errors,
        )

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    duplicate_count = int(timestamps.duplicated().sum())
    missing_timestamp_count = int(timestamps.isna().sum())
    sorted_ascending = timestamps.is_monotonic_increasing
    start_ts = timestamps.min()
    end_ts = timestamps.max()
    minimum_history = bool((end_ts - start_ts).days >= MIN_HISTORY_DAYS)
    if duplicate_count:
        errors.append("duplicate timestamps found")
    if missing_timestamp_count:
        errors.append("missing timestamps found")
    if not sorted_ascending:
        errors.append("timestamps are not sorted ascending")
    if not minimum_history:
        errors.append("less than 3 years history")
    if frame[list(OHLCV_COLUMNS)].isnull().any().any():
        errors.append("OHLCV frame contains null values")

    return TimeframeValidation(
        symbol=symbol,
        timeframe=timeframe,
        rows=int(len(frame)),
        provider=provider,
        validation="passed" if not errors else "failed",
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
        duplicate_timestamps=duplicate_count,
        missing_timestamps=missing_timestamp_count,
        sorted_ascending=bool(sorted_ascending),
        minimum_3_years_history=minimum_history,
        errors=errors,
    )


def write_data_provider_diagnostics(output_dir: Path, results: list[EquityDataResult]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider_fallback_chain": list(EQUITY_PROVIDERS),
        "assets": {
            result.symbol: result.diagnostics_payload()
            for result in results
        },
    }
    path = output_dir / "data_provider_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def normalize_yahoo_chart(payload: dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise DataLoadError("Yahoo Finance returned an unexpected payload shape.")
    error = chart.get("error")
    if error:
        description = error.get("description") if isinstance(error, dict) else str(error)
        raise DataLoadError(f"Yahoo Finance request failed for {symbol} {interval}: {description}.")
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return _empty_ohlcv_frame()
    result = results[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    quote_payload = None
    if isinstance(indicators, dict) and isinstance(indicators.get("quote"), list) and indicators["quote"]:
        quote_payload = indicators["quote"][0]
    if not isinstance(timestamps, list) or not isinstance(quote_payload, dict):
        return _empty_ohlcv_frame()
    rows: list[dict[str, Any]] = []
    opens = quote_payload.get("open") or []
    highs = quote_payload.get("high") or []
    lows = quote_payload.get("low") or []
    closes = quote_payload.get("close") or []
    volumes = quote_payload.get("volume") or []
    for index, timestamp in enumerate(timestamps):
        values = (
            _list_value(opens, index),
            _list_value(highs, index),
            _list_value(lows, index),
            _list_value(closes, index),
            _list_value(volumes, index),
        )
        if any(value is None for value in values[:4]):
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(int(timestamp), unit="s", utc=True),
                "open": float(values[0]),
                "high": float(values[1]),
                "low": float(values[2]),
                "close": float(values[3]),
                "volume": float(values[4] or 0.0),
            }
        )
    return _normalize_rows(rows)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if timeframe != "4h":
        raise DataLoadError(f"Unsupported generated timeframe {timeframe!r}.")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").set_index("timestamp")
    resampled = data.resample("4h", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return _normalize_rows(resampled.dropna(subset=["open", "high", "low", "close"]).reset_index().to_dict("records"))


def _load_or_download_timeframe(
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cache_dir: Path | None,
    refresh_cache: bool,
    timeout_seconds: float,
    providers: tuple[str, ...],
) -> tuple[pd.DataFrame, str, list[ProviderAttempt]]:
    cache_path = _cache_path(cache_dir, symbol, timeframe)
    cached = _read_cache(cache_path) if cache_path is not None and cache_path.exists() and not refresh_cache else None
    if cached is not None and _cache_covers(cached, start, end):
        return _filter_date_range(cached, start, end), "cache", [
            ProviderAttempt(symbol=symbol, timeframe=timeframe, provider="cache", status="success", rows=len(cached))
        ]

    frame, provider, attempts = _download_with_fallback(symbol, timeframe, start, end, timeout_seconds, providers)
    if cached is not None and not cached.empty:
        frame = _merge_frames([cached, frame])
    if cache_path is not None:
        _write_cache(frame, cache_path)
    return _filter_date_range(frame, start, end), provider, attempts


def _download_with_fallback(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    timeout_seconds: float,
    providers: tuple[str, ...],
) -> tuple[pd.DataFrame, str, list[ProviderAttempt]]:
    attempts: list[ProviderAttempt] = []
    errors: list[str] = []
    for provider in providers:
        try:
            frame = _fetch_provider(provider, symbol, timeframe, start, end, timeout_seconds)
            if frame.empty:
                raise DataLoadError(f"{provider} returned no rows.")
            normalized = _normalize_rows(frame.to_dict("records"))
            attempts.append(ProviderAttempt(symbol, timeframe, provider, "success", len(normalized)))
            return normalized, provider, attempts
        except Exception as exc:  # noqa: BLE001 - diagnostics should record provider-specific failure.
            message = str(exc)
            attempts.append(ProviderAttempt(symbol, timeframe, provider, "failed", 0, message))
            errors.append(f"{provider}: {message}")
    raise EquityDataAdapterError(
        f"All equity providers failed for {symbol} {timeframe}: {' | '.join(errors)}",
        attempts,
    )


def _fetch_provider(
    provider: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    timeout_seconds: float,
) -> pd.DataFrame:
    if provider == "yahoo":
        return _fetch_yahoo(symbol, timeframe, start, end, timeout_seconds)
    if provider == "stooq":
        return _fetch_stooq(symbol, timeframe, start, end, timeout_seconds)
    if provider == "alpha_vantage":
        return _fetch_alpha_vantage(symbol, timeframe, start, end, timeout_seconds)
    if provider == "twelve_data":
        return _fetch_twelve_data(symbol, timeframe, start, end, timeout_seconds)
    raise DataLoadError(f"Unsupported equity provider {provider!r}.")


def _fetch_yahoo(symbol: str, timeframe: str, start: str, end: str, timeout_seconds: float) -> pd.DataFrame:
    if timeframe not in YAHOO_INTERVALS:
        raise DataLoadError(f"Yahoo Finance does not support {timeframe}.")
    start_ts = _timestamp(start)
    end_ts = _timestamp(end) if end != "latest" else pd.Timestamp.now(tz="UTC")
    if timeframe == "1h":
        earliest_intraday = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=729)
        if start_ts < earliest_intraday:
            start_ts = earliest_intraday
        frames = []
        current = start_ts
        while current < end_ts:
            chunk_end = min(current + pd.Timedelta(days=650), end_ts)
            frames.append(_fetch_yahoo_chunk(symbol, timeframe, current, chunk_end, timeout_seconds))
            current = chunk_end
        return _merge_frames(frames)
    return _fetch_yahoo_chunk(symbol, timeframe, start_ts, end_ts, timeout_seconds)


def _fetch_yahoo_chunk(
    symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    timeout_seconds: float,
) -> pd.DataFrame:
    query = urlencode(
        {
            "period1": int(start_ts.timestamp()),
            "period2": int(end_ts.timestamp()),
            "interval": YAHOO_INTERVALS[timeframe],
            "events": "history",
            "includePrePost": "false",
        }
    )
    request = Request(
        f"{YAHOO_BASE_URL}/v8/finance/chart/{quote(symbol.upper())}?{query}",
        headers={"User-Agent": "trading-agent-equity-validation/0.1"},
        method="GET",
    )
    payload = _load_json_response(request, urlopen, timeout_seconds, "Yahoo Finance")
    return normalize_yahoo_chart(payload, symbol, timeframe)


def _fetch_stooq(symbol: str, timeframe: str, start: str, end: str, timeout_seconds: float) -> pd.DataFrame:
    if timeframe not in STOOQ_INTERVALS:
        raise DataLoadError(f"Stooq does not support {timeframe}.")
    query = urlencode(
        {
            "s": _stooq_symbol(symbol),
            "i": STOOQ_INTERVALS[timeframe],
            "d1": _timestamp(start).strftime("%Y%m%d"),
            "d2": (_timestamp(end) if end != "latest" else pd.Timestamp.now(tz="UTC")).strftime("%Y%m%d"),
        }
    )
    url = f"{STOOQ_BASE_URL}/q/d/l/?{query}"
    body = _open_stooq_csv(url, timeout_seconds)
    return _parse_stooq_csv(body)


def _open_stooq_csv(url: str, timeout_seconds: float) -> str:
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def fetch(target: str, method: str = "GET", body: bytes | None = None) -> str:
        request = Request(
            target,
            data=body,
            headers={"User-Agent": "Mozilla/5.0 (compatible; trading-agent-equity-validation/0.1)"},
            method=method,
        )
        if body is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        with opener.open(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", "ignore")

    try:
        body = fetch(url)
        if _is_stooq_verification_page(body):
            challenge = _solve_stooq_challenge(body)
            fetch(
                f"{STOOQ_BASE_URL}/__verify",
                method="POST",
                body=urlencode(challenge).encode("utf-8"),
            )
            body = fetch(url)
        return body
    except OSError as exc:
        raise DataLoadError(f"Stooq request failed: {exc}.") from exc


def _is_stooq_verification_page(body: str) -> bool:
    return "This site requires JavaScript to verify your browser" in body and "/__verify" in body


def _solve_stooq_challenge(body: str) -> dict[str, str]:
    c_match = re.search(r'const c="([^"]+)"', body)
    d_match = re.search(r",d=(\d+),t=", body)
    if c_match is None or d_match is None:
        raise DataLoadError("Stooq verification challenge could not be parsed.")
    challenge = c_match.group(1)
    difficulty = int(d_match.group(1))
    prefix = "0" * difficulty
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{challenge}{nonce}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            return {"c": challenge, "n": str(nonce)}
        nonce += 1


def _fetch_alpha_vantage(symbol: str, timeframe: str, start: str, end: str, timeout_seconds: float) -> pd.DataFrame:
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise DataLoadError("ALPHA_VANTAGE_API_KEY is not configured.")
    if timeframe == "1h":
        query = urlencode(
            {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol.upper(),
                "interval": ALPHA_INTERVALS[timeframe],
                "outputsize": "full",
                "apikey": api_key,
            }
        )
        key = f"Time Series ({ALPHA_INTERVALS[timeframe]})"
    elif timeframe == "1d":
        query = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol.upper(),
                "outputsize": "full",
                "apikey": api_key,
            }
        )
        key = "Time Series (Daily)"
    else:
        raise DataLoadError(f"Alpha Vantage does not support {timeframe}.")
    request = Request(f"{ALPHA_VANTAGE_BASE_URL}/query?{query}", headers={"User-Agent": "trading-agent-equity-validation/0.1"})
    payload = _load_json_response(request, urlopen, timeout_seconds, "Alpha Vantage")
    if "Error Message" in payload or "Note" in payload:
        raise DataLoadError(str(payload.get("Error Message") or payload.get("Note")))
    series = payload.get(key)
    if not isinstance(series, dict):
        raise DataLoadError("Alpha Vantage returned no time series.")
    rows = []
    for timestamp, values in series.items():
        rows.append(
            {
                "timestamp": timestamp,
                "open": values.get("1. open"),
                "high": values.get("2. high"),
                "low": values.get("3. low"),
                "close": values.get("4. close"),
                "volume": values.get("6. volume") or values.get("5. volume") or 0,
            }
        )
    return _filter_date_range(_normalize_rows(rows), start, end)


def _fetch_twelve_data(symbol: str, timeframe: str, start: str, end: str, timeout_seconds: float) -> pd.DataFrame:
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        raise DataLoadError("TWELVE_DATA_API_KEY is not configured.")
    if timeframe not in TWELVE_INTERVALS:
        raise DataLoadError(f"Twelve Data does not support {timeframe}.")
    end_ts = _timestamp(end) if end != "latest" else pd.Timestamp.now(tz="UTC")
    query = urlencode(
        {
            "symbol": symbol.upper(),
            "interval": TWELVE_INTERVALS[timeframe],
            "start_date": _timestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
            "outputsize": 5000,
            "apikey": api_key,
        }
    )
    request = Request(f"{TWELVE_DATA_BASE_URL}/time_series?{query}", headers={"User-Agent": "trading-agent-equity-validation/0.1"})
    payload = _load_json_response(request, urlopen, timeout_seconds, "Twelve Data")
    if payload.get("status") == "error":
        raise DataLoadError(str(payload.get("message", "Twelve Data error")))
    values = payload.get("values")
    if not isinstance(values, list):
        raise DataLoadError("Twelve Data returned no values.")
    rows = [
        {
            "timestamp": item.get("datetime"),
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volume") or 0,
        }
        for item in values
    ]
    return _filter_date_range(_normalize_rows(rows), start, end)


def _parse_stooq_csv(body: str) -> pd.DataFrame:
    if "Access denied" in body:
        raise DataLoadError("Stooq access denied after verification.")
    if "<!DOCTYPE html>" in body or "<html" in body.lower():
        raise DataLoadError("Stooq returned HTML instead of CSV.")
    if not body.strip() or body.lower().startswith("no data"):
        return _empty_ohlcv_frame()
    rows = []
    for row in csv.DictReader(io.StringIO(body)):
        date = row.get("Date")
        time = row.get("Time")
        timestamp = f"{date} {time}" if time else date
        rows.append(
            {
                "timestamp": timestamp,
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "close": row.get("Close"),
                "volume": row.get("Volume") or 0,
            }
        )
    return _normalize_rows(rows)


def _normalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_ohlcv_frame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if frame.empty:
        return _empty_ohlcv_frame()
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)[list(OHLCV_COLUMNS)]


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
    filtered = filtered[filtered["timestamp"] >= _timestamp(start)]
    if end != "latest":
        filtered = filtered[filtered["timestamp"] <= _timestamp(end)]
    return filtered.sort_values("timestamp").reset_index(drop=True)[list(OHLCV_COLUMNS)]


def _cache_covers(frame: pd.DataFrame, start: str, end: str) -> bool:
    if frame.empty:
        return False
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    starts_before_request = timestamps.min() <= _timestamp(start)
    ends_after_request = end == "latest" or timestamps.max() >= _timestamp(end)
    return bool(starts_before_request and ends_after_request)


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return _empty_ohlcv_frame()
    return _normalize_rows(pd.concat(non_empty, ignore_index=True).to_dict("records"))


def _cache_path(cache_dir: Path | None, symbol: str, timeframe: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{symbol.upper()}_{timeframe}.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    return _normalize_rows(pd.read_csv(path).to_dict("records"))


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _ordered_timeframes(timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _stooq_symbol(symbol: str) -> str:
    normalized = symbol.lower()
    if "." in normalized:
        return normalized
    return f"{normalized}.us"


def _list_value(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLCV_COLUMNS))
