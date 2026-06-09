"""Configuration for the Phase 1 signal engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
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
DEFAULT_BINANCE_BASE_URL = "https://api.binance.com"


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
    request_timeout_seconds: float = 10.0
    position_mode: str = "NO_POSITION"

    def __post_init__(self) -> None:
        symbol = self.symbol.upper()
        position_mode = self.position_mode.upper()
        output_dir = Path(self.output_dir)
        base_url = self.binance_base_url.rstrip("/")

        if symbol not in SUPPORTED_SYMBOLS:
            supported = ", ".join(SUPPORTED_SYMBOLS)
            raise ConfigurationError(f"Unsupported symbol {self.symbol!r}. Supported: {supported}.")
        if self.interval not in SUPPORTED_INTERVALS:
            supported = ", ".join(SUPPORTED_INTERVALS)
            raise ConfigurationError(f"Unsupported interval {self.interval!r}. Supported: {supported}.")
        if not 200 <= self.history_limit <= 1000:
            raise ConfigurationError("history_limit must be between 200 and 1000 for EMA200-based signals.")
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("request_timeout_seconds must be greater than zero.")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError("binance_base_url must be an http or https URL.")
        if position_mode not in SUPPORTED_POSITION_MODES:
            supported = ", ".join(SUPPORTED_POSITION_MODES)
            raise ConfigurationError(f"Unsupported position_mode {self.position_mode!r}. Supported: {supported}.")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position_mode", position_mode)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "binance_base_url", base_url)
