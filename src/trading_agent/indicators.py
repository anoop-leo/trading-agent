"""Independent technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.data import validate_ohlcv_frame


class IndicatorError(ValueError):
    """Raised when an indicator cannot be calculated from the supplied data."""


def _require_positive_window(window: int, name: str = "window") -> None:
    if window <= 0:
        raise IndicatorError(f"{name} must be greater than zero.")


def calculate_ema(close: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average for a close-price series."""

    _require_positive_window(span, "span")
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Wilder-style RSI."""

    _require_positive_window(window)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50)
    return rsi


def calculate_macd(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Calculate MACD, signal, and histogram series."""

    _require_positive_window(fast_span, "fast_span")
    _require_positive_window(slow_span, "slow_span")
    _require_positive_window(signal_span, "signal_span")
    if fast_span >= slow_span:
        raise IndicatorError("fast_span must be smaller than slow_span.")

    fast_ema = close.ewm(span=fast_span, adjust=False).mean()
    slow_ema = close.ewm(span=slow_span, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": signal,
            "macd_histogram": macd - signal,
        }
    )


def calculate_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    deviations: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Band middle, upper, and lower series."""

    _require_positive_window(window)
    if deviations <= 0:
        raise IndicatorError("deviations must be greater than zero.")

    middle = close.rolling(window=window, min_periods=window).mean()
    standard_deviation = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + deviations * standard_deviation,
            "bb_lower": middle - deviations * standard_deviation,
        }
    )


def calculate_volume_average(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling average volume."""

    _require_positive_window(window)
    return volume.rolling(window=window, min_periods=window).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with Phase 1 indicator columns added."""

    enriched = validate_ohlcv_frame(frame).copy()
    close = enriched["close"]

    enriched["ema_20"] = calculate_ema(close, 20)
    enriched["ema_50"] = calculate_ema(close, 50)
    enriched["ema_200"] = calculate_ema(close, 200)
    enriched["rsi_14"] = calculate_rsi(close, 14)
    enriched = enriched.join(calculate_macd(close))
    enriched = enriched.join(calculate_bollinger_bands(close))
    enriched["volume_ma_20"] = calculate_volume_average(enriched["volume"], 20)

    return enriched
