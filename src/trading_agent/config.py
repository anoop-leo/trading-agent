"""Configuration for the Phase 1 signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_SYMBOLS_BY_SOURCE = {
    "BINANCE": ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    "BYBIT": ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"),
}
SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT")
SUPPORTED_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
)
SUPPORTED_POSITION_MODES = ("NO_POSITION", "HOLDING")
SUPPORTED_MARKET_DATA_SOURCES = ("AUTO", "BINANCE", "BYBIT")
DEFAULT_TIMEFRAMES = ("1h", "4h", "1d")
DEFAULT_BINANCE_BASE_URL = "https://api.binance.com"
DEFAULT_BYBIT_BASE_URL = "https://api.bybit.com"


class ConfigurationError(ValueError):
    """Raised when local runtime configuration is invalid."""


@dataclass(frozen=True)
class AgentConfig:
    """Validated runtime configuration for a single Phase 1 run."""

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    history_limit: int = 500
    output_dir: Path = Path("outputs")
    binance_base_url: str = DEFAULT_BINANCE_BASE_URL
    bybit_base_url: str = DEFAULT_BYBIT_BASE_URL
    request_timeout_seconds: float = 10.0
    position_mode: str = "NO_POSITION"
    timeframes: tuple[str, ...] | list[str] | None = None
    market_data_source: str = "AUTO"
    resolved_market_data_source: str = field(init=False)

    def __post_init__(self) -> None:
        symbol = self.symbol.upper()
        position_mode = self.position_mode.upper()
        market_data_source = self.market_data_source.upper()
        timeframes = DEFAULT_TIMEFRAMES if self.timeframes is None else tuple(self.timeframes)
        output_dir = Path(self.output_dir)
        binance_base_url = self.binance_base_url.rstrip("/")
        bybit_base_url = self.bybit_base_url.rstrip("/")

        if symbol not in SUPPORTED_SYMBOLS:
            supported = ", ".join(SUPPORTED_SYMBOLS)
            raise ConfigurationError(f"Unsupported symbol {self.symbol!r}. Supported: {supported}.")
        if market_data_source not in SUPPORTED_MARKET_DATA_SOURCES:
            supported = ", ".join(SUPPORTED_MARKET_DATA_SOURCES)
            raise ConfigurationError(
                f"Unsupported market_data_source {self.market_data_source!r}. Supported: {supported}."
            )
        resolved_source = _resolve_market_data_source(symbol, market_data_source)
        if symbol not in SUPPORTED_SYMBOLS_BY_SOURCE[resolved_source]:
            raise ConfigurationError(f"{symbol} is not supported by {resolved_source} market data.")
        if self.interval not in SUPPORTED_INTERVALS:
            supported = ", ".join(SUPPORTED_INTERVALS)
            raise ConfigurationError(f"Unsupported interval {self.interval!r}. Supported: {supported}.")
        if not 200 <= self.history_limit <= 1000:
            raise ConfigurationError("history_limit must be between 200 and 1000 for EMA200-based signals.")
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("request_timeout_seconds must be greater than zero.")
        if not binance_base_url.startswith(("https://", "http://")):
            raise ConfigurationError("binance_base_url must be an http or https URL.")
        if not bybit_base_url.startswith(("https://", "http://")):
            raise ConfigurationError("bybit_base_url must be an http or https URL.")
        if position_mode not in SUPPORTED_POSITION_MODES:
            supported = ", ".join(SUPPORTED_POSITION_MODES)
            raise ConfigurationError(f"Unsupported position_mode {self.position_mode!r}. Supported: {supported}.")
        if not timeframes:
            raise ConfigurationError("timeframes must include at least one interval.")
        unsupported_timeframes = [timeframe for timeframe in timeframes if timeframe not in SUPPORTED_INTERVALS]
        if unsupported_timeframes:
            supported = ", ".join(SUPPORTED_INTERVALS)
            invalid = ", ".join(unsupported_timeframes)
            raise ConfigurationError(f"Unsupported timeframes {invalid}. Supported: {supported}.")
        if len(set(timeframes)) != len(timeframes):
            raise ConfigurationError("timeframes must not contain duplicates.")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position_mode", position_mode)
        object.__setattr__(self, "market_data_source", market_data_source)
        object.__setattr__(self, "resolved_market_data_source", resolved_source)
        object.__setattr__(self, "timeframes", timeframes)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "binance_base_url", binance_base_url)
        object.__setattr__(self, "bybit_base_url", bybit_base_url)


def _resolve_market_data_source(symbol: str, market_data_source: str) -> str:
    if market_data_source != "AUTO":
        return market_data_source
    if symbol == "HYPEUSDT":
        return "BYBIT"
    return "BINANCE"
