"""Generic crypto investor agent for non-BTC assets such as Chainlink."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scoring.crypto_investor_score import (
    CRYPTO_INVESTOR_WEIGHTS,
    band_for_crypto_investor_score,
    calculate_crypto_investor_score,
    factor_payload,
    score_distance_from_200d_ma,
    score_drawdown_from_cycle_high,
    score_liquidity_proxy,
    score_monthly_trend,
    score_thesis_risk,
    score_volatility_risk,
    score_volume_trend,
    score_weekly_rsi,
)
from trading_agent.data import BinanceKlineProvider, BybitKlineProvider, DataLoadError
from trading_agent.indicators import calculate_ema, calculate_rsi


CRYPTO_INVESTOR_REPORT_TEMPLATE = "crypto_investor_{symbol}_report.json"
SUPPORTED_MARKET_DATA_SOURCES = {"BINANCE", "BYBIT"}

ASSET_PROFILES = {
    "ETH": {
        "name": "Ethereum",
        "base_asset": "ETH",
        "market_symbol": "ETHUSDT",
        "sector": "Smart contract platform",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"ETH", "ETHUSDT", "ETHEREUM"},
    },
    "SOL": {
        "name": "Solana",
        "base_asset": "SOL",
        "market_symbol": "SOLUSDT",
        "sector": "High-throughput smart contract platform",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"SOL", "SOLUSDT", "SOLANA"},
    },
    "XRP": {
        "name": "XRP",
        "base_asset": "XRP",
        "market_symbol": "XRPUSDT",
        "sector": "Payments and settlement network",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"XRP", "XRPUSDT", "RIPPLE"},
    },
    "AVAX": {
        "name": "Avalanche",
        "base_asset": "AVAX",
        "market_symbol": "AVAXUSDT",
        "sector": "Smart contract platform",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"AVAX", "AVAXUSDT", "AVALANCHE", "AVX"},
    },
    "LINK": {
        "name": "Chainlink",
        "base_asset": "LINK",
        "market_symbol": "LINKUSDT",
        "sector": "Decentralized oracle network",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"LINK", "LINKUSDT", "CHAINLINK", "CHAINLNK"},
    },
    "ONDO": {
        "name": "Ondo",
        "base_asset": "ONDO",
        "market_symbol": "ONDOUSDT",
        "sector": "Tokenized real-world assets",
        "market_data_source": "BINANCE",
        "derive_higher_timeframes_from_daily": False,
        "supported_aliases": {"ONDO", "ONDOUSDT"},
    },
    "HYPE": {
        "name": "Hyperliquid",
        "base_asset": "HYPE",
        "market_symbol": "HYPEUSDT",
        "sector": "Perpetual futures exchange token",
        "market_data_source": "BYBIT",
        "derive_higher_timeframes_from_daily": True,
        "supported_aliases": {"HYPE", "HYPEUSDT", "HYPER", "HYPERLIQUID"},
    },
}


@dataclass(frozen=True)
class CryptoInvestorConfig:
    symbol: str = "LINK"
    asset_name: str | None = None
    sector: str | None = None
    market_data_source: str | None = None
    output_dir: Path = Path("outputs")
    offline: bool = False
    request_timeout_seconds: float = 10.0
    binance_base_url: str = "https://api.binance.com"
    thesis_risk_level: str = "MODERATE"
    thesis_risk_flags: tuple[str, ...] = field(default_factory=tuple)
    current_price: float | None = None
    ma200: float | None = None
    weekly_rsi: float | None = None
    monthly_ema20: float | None = None
    monthly_trend: str | None = None
    recent_cycle_high: float | None = None
    quote_volume_usd: float | None = None
    average_quote_volume_usd: float | None = None
    atr_pct: float | None = None


@dataclass(frozen=True)
class CryptoMarketMetrics:
    price: float | None
    ma200: float | None
    distance_from_200d_ma_pct: float | None
    weekly_rsi: float | None
    monthly_ema20: float | None
    monthly_trend: str
    drawdown_from_cycle_high_pct: float | None
    volume_ratio: float | None
    atr_pct: float | None
    quote_volume_usd: float | None
    as_of: str | None
    missing_fields: list[str]


class CryptoInvestorAgentError(ValueError):
    """Raised when the generic crypto investor agent cannot analyze a symbol."""


class CryptoInvestorAgent:
    """Long-term accumulation analysis for non-BTC crypto assets.

    The agent is deterministic and market-data only. It produces investor
    guidance, not trading signals or broker actions.
    """

    def __init__(
        self,
        config: CryptoInvestorConfig,
        market_data_provider: BinanceKlineProvider | BybitKlineProvider | None = None,
    ) -> None:
        base_symbol = normalize_crypto_investor_symbol(config.symbol)
        profile = _asset_profile_for_symbol(base_symbol, config)

        self.config = CryptoInvestorConfig(
            symbol=base_symbol,
            asset_name=config.asset_name,
            sector=config.sector,
            market_data_source=config.market_data_source,
            output_dir=Path(config.output_dir),
            offline=config.offline,
            request_timeout_seconds=config.request_timeout_seconds,
            binance_base_url=config.binance_base_url.rstrip("/"),
            thesis_risk_level=config.thesis_risk_level.upper(),
            thesis_risk_flags=tuple(config.thesis_risk_flags),
            current_price=config.current_price,
            ma200=config.ma200,
            weekly_rsi=config.weekly_rsi,
            monthly_ema20=config.monthly_ema20,
            monthly_trend=config.monthly_trend,
            recent_cycle_high=config.recent_cycle_high,
            quote_volume_usd=config.quote_volume_usd,
            average_quote_volume_usd=config.average_quote_volume_usd,
            atr_pct=config.atr_pct,
        )
        self.asset_profile = profile
        self.market_data_provider = market_data_provider or self._default_market_data_provider()

    def run(self) -> dict[str, Any]:
        market = self._market_metrics()
        thesis_risk = self._thesis_risk()
        factor_scores = self._factor_scores(market, thesis_risk)
        investor_score = calculate_crypto_investor_score(factor_scores)
        band = band_for_crypto_investor_score(investor_score, thesis_risk["level"])
        missing_fields = sorted(set(market.missing_fields))
        data_confidence = _data_quality_confidence(missing_fields)
        final_action = band.final_investor_action
        liquidity_score = int(factor_scores["liquidity_proxy"]["score"])

        payload = {
            "agent": "CRYPTO_INVESTOR",
            "symbol": self.config.symbol,
            "market_symbol": self.asset_profile["market_symbol"],
            "generated_at": datetime.now(UTC).isoformat(),
            "asset_profile": {
                "name": self.asset_profile["name"],
                "base_asset": self.asset_profile["base_asset"],
                "sector": self.asset_profile["sector"],
                "market_data_source": self.asset_profile["market_data_source"],
            },
            "investor_score": investor_score,
            "investor_band": band.investor_band,
            "accumulation_bias": band.accumulation_bias,
            "suggested_dca_multiplier": band.suggested_dca_multiplier,
            "final_investor_action": final_action,
            "risk_level": _risk_level(thesis_risk["level"], market.atr_pct, liquidity_score, data_confidence),
            "market_snapshot": {
                "as_of": market.as_of,
                "price": market.price,
                "ma200": market.ma200,
                "distance_from_200d_ma_pct": market.distance_from_200d_ma_pct,
                "weekly_rsi": market.weekly_rsi,
                "monthly_ema20": market.monthly_ema20,
                "monthly_trend": market.monthly_trend,
                "drawdown_from_cycle_high_pct": market.drawdown_from_cycle_high_pct,
                "volume_ratio": market.volume_ratio,
                "atr_pct": market.atr_pct,
                "quote_volume_usd": market.quote_volume_usd,
            },
            "factor_scores": factor_scores,
            "thesis_risk": thesis_risk,
            "rationale": self._rationale(investor_score, band.investor_band, market, thesis_risk, data_confidence),
            "data_quality": {
                "missing_fields": missing_fields,
                "fallback_fields": [],
                "confidence": data_confidence,
            },
            "analysis_modules": [
                "distance_from_200d_ma",
                "weekly_rsi",
                "monthly_trend",
                "drawdown_from_cycle_high",
                "volume_trend",
                "volatility_risk",
                "liquidity_proxy",
                "thesis_risk_check",
            ],
            "safety": {
                "live_trading_enabled": False,
                "broker_order_endpoints_enabled": False,
                "output_is_investor_guidance_only": True,
            },
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.output_dir / CRYPTO_INVESTOR_REPORT_TEMPLATE.format(symbol=self.config.symbol.lower())
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    def _market_metrics(self) -> CryptoMarketMetrics:
        if _has_manual_market_inputs(self.config):
            return self._manual_market_metrics()

        if self.config.offline:
            return CryptoMarketMetrics(
                price=None,
                ma200=None,
                distance_from_200d_ma_pct=None,
                weekly_rsi=None,
                monthly_ema20=None,
                monthly_trend="",
                drawdown_from_cycle_high_pct=None,
                volume_ratio=None,
                atr_pct=None,
                quote_volume_usd=None,
                as_of=None,
                missing_fields=_market_missing_fields(),
            )

        symbol = str(self.asset_profile["market_symbol"])
        daily = _safe_fetch(self.market_data_provider, symbol, "1d", 1000)
        if self.asset_profile["derive_higher_timeframes_from_daily"]:
            weekly = _resample_ohlcv(daily, "W") if daily is not None else None
            monthly = _resample_ohlcv(daily, "ME") if daily is not None else None
        else:
            weekly = _safe_fetch(self.market_data_provider, symbol, "1w", 300)
            monthly = _safe_fetch(self.market_data_provider, symbol, "1M", 120)

        missing_fields: list[str] = []
        price = None
        ma200 = None
        distance_pct = None
        drawdown_pct = None
        volume_ratio = None
        atr_pct = None
        quote_volume_usd = None
        as_of = None

        if daily is None or daily.empty:
            missing_fields.extend(
                [
                    "price",
                    "distance_from_200d_ma",
                    "drawdown_from_cycle_high",
                    "volume_trend",
                    "volatility_risk",
                    "liquidity_proxy",
                ]
            )
        else:
            daily = daily.copy()
            price = _round(float(daily.iloc[-1]["close"]))
            as_of = pd.Timestamp(daily.iloc[-1]["timestamp"]).isoformat()
            latest_volume = float(daily.iloc[-1]["volume"])
            quote_volume_usd = _round(latest_volume * price)
            if len(daily) >= 200:
                ma200 = _round(float(daily["close"].astype(float).rolling(200, min_periods=200).mean().iloc[-1]))
                distance_pct = _round(((price - ma200) / ma200) * 100) if ma200 and ma200 > 0 else None
            else:
                missing_fields.append("distance_from_200d_ma")
            if len(daily) >= 30:
                cycle_high = float(daily.tail(min(len(daily), 1460))["high"].max())
                drawdown_pct = _round(((price - cycle_high) / cycle_high) * 100) if cycle_high > 0 else None
            else:
                missing_fields.append("drawdown_from_cycle_high")
            if len(daily) >= 20:
                quote_volume_series = daily["volume"].astype(float) * daily["close"].astype(float)
                average_quote_volume = float(quote_volume_series.rolling(20, min_periods=20).mean().iloc[-1])
                volume_ratio = _round(quote_volume_usd / average_quote_volume) if average_quote_volume > 0 else None
            else:
                missing_fields.append("volume_trend")
            if len(daily) >= 15:
                atr_pct = _calculate_atr_pct(daily)
            else:
                missing_fields.append("volatility_risk")
            if quote_volume_usd is None:
                missing_fields.append("liquidity_proxy")

        weekly_rsi = None
        if weekly is None or len(weekly) < 20:
            missing_fields.append("weekly_rsi")
        else:
            weekly_rsi = _round(float(calculate_rsi(weekly["close"].astype(float), 14).iloc[-1]))

        monthly_trend = ""
        monthly_ema20 = None
        if monthly is None or len(monthly) < 25:
            missing_fields.append("monthly_trend")
        else:
            monthly_close = monthly["close"].astype(float)
            monthly_ema20 = _round(float(calculate_ema(monthly_close, 20).iloc[-1]))
            monthly_trend = classify_monthly_trend(monthly)

        return CryptoMarketMetrics(
            price=price,
            ma200=ma200,
            distance_from_200d_ma_pct=distance_pct,
            weekly_rsi=weekly_rsi,
            monthly_ema20=monthly_ema20,
            monthly_trend=monthly_trend,
            drawdown_from_cycle_high_pct=drawdown_pct,
            volume_ratio=volume_ratio,
            atr_pct=atr_pct,
            quote_volume_usd=quote_volume_usd,
            as_of=as_of,
            missing_fields=missing_fields,
        )

    def _factor_scores(
        self,
        market: CryptoMarketMetrics,
        thesis_risk: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return {
            "distance_from_200d_ma": factor_payload(
                market.distance_from_200d_ma_pct,
                score_distance_from_200d_ma(market.distance_from_200d_ma_pct),
                CRYPTO_INVESTOR_WEIGHTS["distance_from_200d_ma"],
                "value_pct",
            ),
            "weekly_rsi": factor_payload(
                market.weekly_rsi,
                score_weekly_rsi(market.weekly_rsi),
                CRYPTO_INVESTOR_WEIGHTS["weekly_rsi"],
            ),
            "monthly_trend": factor_payload(
                market.monthly_trend,
                score_monthly_trend(market.monthly_trend),
                CRYPTO_INVESTOR_WEIGHTS["monthly_trend"],
            ),
            "drawdown_from_cycle_high": factor_payload(
                market.drawdown_from_cycle_high_pct,
                score_drawdown_from_cycle_high(market.drawdown_from_cycle_high_pct),
                CRYPTO_INVESTOR_WEIGHTS["drawdown_from_cycle_high"],
                "value_pct",
            ),
            "volume_trend": factor_payload(
                market.volume_ratio,
                score_volume_trend(market.volume_ratio),
                CRYPTO_INVESTOR_WEIGHTS["volume_trend"],
                "volume_ratio",
            ),
            "volatility_risk": factor_payload(
                market.atr_pct,
                score_volatility_risk(market.atr_pct),
                CRYPTO_INVESTOR_WEIGHTS["volatility_risk"],
                "atr_pct",
            ),
            "liquidity_proxy": factor_payload(
                market.quote_volume_usd,
                score_liquidity_proxy(market.quote_volume_usd),
                CRYPTO_INVESTOR_WEIGHTS["liquidity_proxy"],
                "quote_volume_usd",
            ),
            "thesis_risk_check": factor_payload(
                thesis_risk["level"],
                score_thesis_risk(thesis_risk["level"]),
                CRYPTO_INVESTOR_WEIGHTS["thesis_risk_check"],
            ),
        }

    def _manual_market_metrics(self) -> CryptoMarketMetrics:
        missing_fields: list[str] = []
        price = self.config.current_price
        ma200 = self.config.ma200
        monthly_ema20 = self.config.monthly_ema20
        distance_pct = _round(((price - ma200) / ma200) * 100) if price is not None and ma200 and ma200 > 0 else None
        drawdown_pct = (
            _round(((price - self.config.recent_cycle_high) / self.config.recent_cycle_high) * 100)
            if price is not None and self.config.recent_cycle_high and self.config.recent_cycle_high > 0
            else None
        )
        volume_ratio = (
            _round(self.config.quote_volume_usd / self.config.average_quote_volume_usd)
            if self.config.quote_volume_usd is not None
            and self.config.average_quote_volume_usd is not None
            and self.config.average_quote_volume_usd > 0
            else None
        )
        monthly_trend = self.config.monthly_trend or _classify_manual_monthly_trend(price, monthly_ema20)

        if price is None:
            missing_fields.append("price")
        if distance_pct is None:
            missing_fields.append("distance_from_200d_ma")
        if drawdown_pct is None:
            missing_fields.append("drawdown_from_cycle_high")
        if self.config.weekly_rsi is None:
            missing_fields.append("weekly_rsi")
        if not monthly_trend:
            missing_fields.append("monthly_trend")
        if volume_ratio is None:
            missing_fields.append("volume_trend")
        if self.config.quote_volume_usd is None:
            missing_fields.append("liquidity_proxy")
        if self.config.atr_pct is None:
            missing_fields.append("volatility_risk")

        return CryptoMarketMetrics(
            price=_round(price),
            ma200=_round(ma200),
            distance_from_200d_ma_pct=distance_pct,
            weekly_rsi=_round(self.config.weekly_rsi),
            monthly_ema20=_round(monthly_ema20),
            monthly_trend=monthly_trend,
            drawdown_from_cycle_high_pct=drawdown_pct,
            volume_ratio=volume_ratio,
            atr_pct=_round(self.config.atr_pct),
            quote_volume_usd=_round(self.config.quote_volume_usd),
            as_of=datetime.now(UTC).isoformat(),
            missing_fields=missing_fields,
        )

    def _thesis_risk(self) -> dict[str, Any]:
        level = self.config.thesis_risk_level.upper()
        if level not in {"LOW", "MODERATE", "HIGH"}:
            level = "MODERATE"
        return {
            "level": level,
            "flags": list(self.config.thesis_risk_flags),
            "source": "operator_input",
        }

    def _default_market_data_provider(self) -> BinanceKlineProvider | BybitKlineProvider:
        if self.asset_profile["market_data_source"] == "BYBIT":
            return BybitKlineProvider(timeout_seconds=self.config.request_timeout_seconds)
        return BinanceKlineProvider(
            base_url=self.config.binance_base_url,
            timeout_seconds=self.config.request_timeout_seconds,
        )

    def _rationale(
        self,
        investor_score: int,
        investor_band: str,
        market: CryptoMarketMetrics,
        thesis_risk: dict[str, Any],
        data_confidence: str,
    ) -> list[str]:
        rationale = [
            f"{self.config.symbol} investor band is {investor_band} with a score of {investor_score}.",
            "This is deterministic investor guidance based on trend discount, momentum, drawdown, liquidity, volatility, and thesis risk.",
        ]
        if market.distance_from_200d_ma_pct is not None:
            rationale.append(f"{self.config.symbol} is {market.distance_from_200d_ma_pct}% from its 200D moving average.")
        if market.drawdown_from_cycle_high_pct is not None:
            rationale.append(f"{self.config.symbol} is {market.drawdown_from_cycle_high_pct}% below its recent cycle high.")
        if market.weekly_rsi is not None:
            rationale.append(f"Weekly RSI is {market.weekly_rsi}, which informs accumulation patience rather than trade timing.")
        if thesis_risk["level"] == "HIGH":
            rationale.append("Thesis risk is HIGH, so larger accumulation guidance is capped.")
        if data_confidence != "HIGH":
            rationale.append(f"Data confidence is {data_confidence}; avoid aggressive sizing until missing context is resolved.")
        return rationale


def run_crypto_investor_agent(
    config: CryptoInvestorConfig,
    market_data_provider: BinanceKlineProvider | BybitKlineProvider | None = None,
) -> dict[str, Any]:
    agent = CryptoInvestorAgent(config, market_data_provider=market_data_provider)
    return agent.run()


def normalize_crypto_investor_symbol(symbol: str) -> str:
    normalized = symbol.upper().replace("-", "").replace("/", "")
    for base_symbol, profile in ASSET_PROFILES.items():
        if normalized in profile["supported_aliases"]:
            return base_symbol
    if normalized.endswith("USDT") and len(normalized) > 4:
        return normalized[:-4]
    return normalized


def _asset_profile_for_symbol(base_symbol: str, config: CryptoInvestorConfig) -> dict[str, Any]:
    profile = ASSET_PROFILES.get(base_symbol)
    if profile is not None:
        resolved = dict(profile)
        if config.asset_name:
            resolved["name"] = config.asset_name
        if config.sector:
            resolved["sector"] = config.sector
        if config.market_data_source:
            source = config.market_data_source.upper()
            if source not in SUPPORTED_MARKET_DATA_SOURCES:
                raise CryptoInvestorAgentError(f"Unsupported market data source {config.market_data_source!r}.")
            resolved["market_data_source"] = source
            resolved["derive_higher_timeframes_from_daily"] = source == "BYBIT"
        return resolved

    source = (config.market_data_source or "BINANCE").upper()
    if source not in SUPPORTED_MARKET_DATA_SOURCES:
        raise CryptoInvestorAgentError(f"Unsupported market data source {config.market_data_source!r}.")
    return {
        "name": config.asset_name or base_symbol,
        "base_asset": base_symbol,
        "market_symbol": f"{base_symbol}USDT",
        "sector": config.sector or "Crypto asset",
        "market_data_source": source,
        "derive_higher_timeframes_from_daily": source == "BYBIT",
        "supported_aliases": {base_symbol, f"{base_symbol}USDT"},
    }


def classify_monthly_trend(monthly: pd.DataFrame) -> str:
    close = monthly["close"].astype(float)
    ema10 = calculate_ema(close, 10)
    ema20 = calculate_ema(close, 20)
    latest_close = float(close.iloc[-1])
    latest_ema10 = float(ema10.iloc[-1])
    latest_ema20 = float(ema20.iloc[-1])
    if latest_ema20 <= 0:
        return ""
    distance_pct = ((latest_close - latest_ema20) / latest_ema20) * 100
    if latest_close < latest_ema20:
        return "BELOW_EMA20"
    if abs(distance_pct) <= 5:
        return "NEAR_EMA20"
    if latest_ema10 < latest_ema20:
        return "ABOVE_EMA20_EMA10_BELOW_EMA20"
    if distance_pct > 80:
        return "EXTREMELY_EXTENDED"
    return "BULLISH_ABOVE_EMA20"


def _safe_fetch(
    provider: BinanceKlineProvider | BybitKlineProvider,
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame | None:
    try:
        return provider.fetch_ohlcv(symbol, interval, limit)
    except (DataLoadError, OSError, TimeoutError):
        return None


def _resample_ohlcv(frame: pd.DataFrame | None, rule: str) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    resampled = (
        working.set_index("timestamp")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return resampled


def _has_manual_market_inputs(config: CryptoInvestorConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.current_price,
            config.ma200,
            config.weekly_rsi,
            config.monthly_ema20,
            config.monthly_trend,
            config.recent_cycle_high,
            config.quote_volume_usd,
            config.average_quote_volume_usd,
            config.atr_pct,
        )
    )


def _classify_manual_monthly_trend(price: float | None, monthly_ema20: float | None) -> str:
    if price is None or monthly_ema20 is None or monthly_ema20 <= 0:
        return ""
    distance_pct = ((price - monthly_ema20) / monthly_ema20) * 100
    if price < monthly_ema20:
        return "BELOW_EMA20"
    if abs(distance_pct) <= 5:
        return "NEAR_EMA20"
    if distance_pct > 80:
        return "EXTREMELY_EXTENDED"
    return "BULLISH_ABOVE_EMA20"


def _calculate_atr_pct(daily: pd.DataFrame, window: int = 14) -> float | None:
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)
    close = daily["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(window, min_periods=window).mean().iloc[-1])
    latest_close = float(close.iloc[-1])
    if latest_close <= 0:
        return None
    return _round((atr / latest_close) * 100)


def _market_missing_fields() -> list[str]:
    return [
        "price",
        "distance_from_200d_ma",
        "weekly_rsi",
        "monthly_trend",
        "drawdown_from_cycle_high",
        "volume_trend",
        "volatility_risk",
        "liquidity_proxy",
    ]


def _data_quality_confidence(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "HIGH"
    if len(missing_fields) <= 2:
        return "MEDIUM"
    return "LOW"


def _risk_level(
    thesis_risk_level: str,
    atr_pct: float | None,
    liquidity_score: int,
    data_confidence: str,
) -> str:
    if thesis_risk_level.upper() == "HIGH" or data_confidence == "LOW":
        return "HIGH"
    level = "MEDIUM" if thesis_risk_level.upper() == "MODERATE" else "LOW"
    if atr_pct is not None and atr_pct > 10:
        level = _increase_risk_level(level)
    if liquidity_score < 4:
        level = _increase_risk_level(level)
    return level


def _increase_risk_level(level: str) -> str:
    if level == "LOW":
        return "MEDIUM"
    return "HIGH"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
