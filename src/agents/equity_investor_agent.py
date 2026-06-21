"""Equity Investor Agent: long-term valuation/accumulation guidance for stocks and ETFs.

Mirrors the role agents/crypto_investor_agent.py plays for altcoins, but uses
equity-appropriate inputs (P/E, P/B, PEG, FCF yield) instead of MVRV/fear-greed.

This agent only ever proposes a PositionRecommendation. It does not size
positions; src/risk/live_risk_engine.py is the only thing allowed to do that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data.equity_data_adapter import EquityDataResult, load_equity_data
from data.equity_fundamentals_provider import EquityFundamentalsProvider
from decision.recommendation import PositionRecommendation
from scoring.equity_investor_score import (
    EQUITY_INVESTOR_WEIGHTS,
    band_for_equity_investor_score,
    calculate_equity_investor_score,
    factor_payload,
    score_distance_from_200d_ma,
    score_fcf_yield,
    score_growth_consistency,
    score_quality_roe,
    score_valuation_pb,
    score_valuation_pe,
    score_valuation_peg,
)
from trading_agent.indicators import calculate_ema, calculate_rsi


EQUITY_INVESTOR_REPORT_TEMPLATE = "equity_investor_{symbol}_report.json"
CORE_EXPENSE_RATIO_MAX_PCT = 0.20

CORE_ETF_PROFILES = {
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "expense_ratio_pct": 0.0945, "tracks": "S&P 500", "broad_market": True},
    "QQQ": {"name": "Invesco QQQ Trust", "expense_ratio_pct": 0.20, "tracks": "Nasdaq-100", "broad_market": True},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "expense_ratio_pct": 0.03, "tracks": "CRSP US Total Market", "broad_market": True},
    "IWM": {"name": "iShares Russell 2000 ETF", "expense_ratio_pct": 0.19, "tracks": "Russell 2000", "broad_market": True},
    "DIA": {"name": "SPDR Dow Jones Industrial Average ETF", "expense_ratio_pct": 0.16, "tracks": "Dow Jones Industrial Average", "broad_market": True},
}


@dataclass(frozen=True)
class EquityInvestorConfig:
    symbol: str = "SPY"
    bucket: str | None = None
    output_dir: Path = Path("outputs")
    offline: bool = False
    request_timeout_seconds: float = 10.0
    default_position_usd: float = 2_000.0
    current_price: float | None = None
    ma200: float | None = None
    weekly_rsi: float | None = None
    monthly_ema20: float | None = None
    monthly_trend: str | None = None


@dataclass(frozen=True)
class EquityMarketMetrics:
    price: float | None
    ma200: float | None
    distance_from_200d_ma_pct: float | None
    weekly_rsi: float | None
    monthly_ema20: float | None
    monthly_trend: str
    as_of: str | None
    missing_fields: list[str]


class EquityInvestorAgentError(ValueError):
    """Raised when the Equity Investor Agent cannot analyze a symbol."""


class EquityInvestorAgent:
    """Long-term accumulation analysis for stocks and core index ETFs.

    Deterministic and market-data only. Produces investor guidance and a
    PositionRecommendation proposal, never a sized order.
    """

    def __init__(
        self,
        config: EquityInvestorConfig,
        equity_data_loader: Callable[..., EquityDataResult] | None = None,
        fundamentals_provider: EquityFundamentalsProvider | None = None,
    ) -> None:
        symbol = config.symbol.upper()
        bucket = config.bucket.lower() if config.bucket else _default_bucket_for_symbol(symbol)
        if bucket not in {"core", "growth"}:
            raise EquityInvestorAgentError(f"Unsupported equity bucket {bucket!r}.")

        self.config = EquityInvestorConfig(
            symbol=symbol,
            bucket=bucket,
            output_dir=Path(config.output_dir),
            offline=config.offline,
            request_timeout_seconds=config.request_timeout_seconds,
            default_position_usd=config.default_position_usd,
            current_price=config.current_price,
            ma200=config.ma200,
            weekly_rsi=config.weekly_rsi,
            monthly_ema20=config.monthly_ema20,
            monthly_trend=config.monthly_trend,
        )
        self.bucket = bucket
        self._equity_data_loader = equity_data_loader or load_equity_data
        self._fundamentals_provider = fundamentals_provider or EquityFundamentalsProvider()

    def run(self) -> dict[str, Any]:
        if self.bucket == "core":
            payload = self._run_core()
        else:
            payload = self._run_growth()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.output_dir / EQUITY_INVESTOR_REPORT_TEMPLATE.format(symbol=self.config.symbol.lower())
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    def _run_core(self) -> dict[str, Any]:
        profile = CORE_ETF_PROFILES.get(self.config.symbol, {
            "name": self.config.symbol,
            "expense_ratio_pct": None,
            "tracks": "unknown",
            "broad_market": False,
        })
        expense_ratio = profile["expense_ratio_pct"]
        is_reasonable = bool(profile["broad_market"]) and expense_ratio is not None and expense_ratio <= CORE_EXPENSE_RATIO_MAX_PCT
        conviction_score = 1.0 if is_reasonable else 0.3
        action = "buy" if is_reasonable else "hold"
        rationale = (
            f"{self.config.symbol} tracks {profile['tracks']} with a {expense_ratio}% expense ratio; "
            f"{'within' if is_reasonable else 'outside'} the {CORE_EXPENSE_RATIO_MAX_PCT}% core diversification threshold."
            if expense_ratio is not None
            else f"{self.config.symbol} has no known expense ratio on file; treat as unverified for the core bucket."
        )

        recommendation = PositionRecommendation(
            symbol=self.config.symbol,
            asset_class="equity",
            bucket="core",
            action=action,
            conviction_score=conviction_score,
            suggested_size_usd=self.config.default_position_usd if action == "buy" else 0.0,
            rationale=rationale,
            source_agent="equity_investor_agent",
        )

        return {
            "agent": "EQUITY_INVESTOR",
            "symbol": self.config.symbol,
            "bucket": "core",
            "generated_at": datetime.now(UTC).isoformat(),
            "asset_profile": {
                "name": profile["name"],
                "tracks": profile["tracks"],
                "expense_ratio_pct": expense_ratio,
            },
            "core_vehicle_check": {
                "reasonable_broad_market_vehicle": is_reasonable,
                "expense_ratio_max_pct": CORE_EXPENSE_RATIO_MAX_PCT,
            },
            "final_investor_action": action.upper(),
            "rationale": [rationale],
            "position_recommendation": recommendation.to_dict(),
            "safety": {
                "live_trading_enabled": False,
                "broker_order_endpoints_enabled": False,
                "output_is_investor_guidance_only": True,
            },
        }

    def _run_growth(self) -> dict[str, Any]:
        market = self._market_metrics()
        fundamentals = self._fundamentals()
        factor_scores = self._factor_scores(market, fundamentals)
        investor_score = calculate_equity_investor_score(factor_scores)
        band = band_for_equity_investor_score(investor_score)
        missing_fields = sorted(set(market.missing_fields) | set(_fundamentals_missing_fields(fundamentals)))
        data_confidence = _data_quality_confidence(missing_fields)
        action = "hold" if band.final_investor_action == "DO_NOT_ACCUMULATE" else "buy"
        conviction_score = round(investor_score / 100, 4)

        recommendation = PositionRecommendation(
            symbol=self.config.symbol,
            asset_class="equity",
            bucket="growth",
            action=action,
            conviction_score=conviction_score,
            suggested_size_usd=round(self.config.default_position_usd * conviction_score, 2) if action == "buy" else 0.0,
            rationale=f"{self.config.symbol} equity investor score is {investor_score} ({band.investor_band}).",
            source_agent="equity_investor_agent",
        )

        return {
            "agent": "EQUITY_INVESTOR",
            "symbol": self.config.symbol,
            "bucket": "growth",
            "generated_at": datetime.now(UTC).isoformat(),
            "investor_score": investor_score,
            "investor_band": band.investor_band,
            "accumulation_bias": band.accumulation_bias,
            "suggested_dca_multiplier": band.suggested_dca_multiplier,
            "final_investor_action": band.final_investor_action,
            "market_snapshot": {
                "as_of": market.as_of,
                "price": market.price,
                "ma200": market.ma200,
                "distance_from_200d_ma_pct": market.distance_from_200d_ma_pct,
                "weekly_rsi": market.weekly_rsi,
                "monthly_ema20": market.monthly_ema20,
                "monthly_trend": market.monthly_trend,
            },
            "fundamentals": fundamentals,
            "factor_scores": factor_scores,
            "rationale": self._rationale(investor_score, band.investor_band, market, fundamentals, data_confidence),
            "data_quality": {
                "missing_fields": missing_fields,
                "confidence": data_confidence,
            },
            "position_recommendation": recommendation.to_dict(),
            "safety": {
                "live_trading_enabled": False,
                "broker_order_endpoints_enabled": False,
                "output_is_investor_guidance_only": True,
            },
        }

    def _market_metrics(self) -> EquityMarketMetrics:
        if self.config.current_price is not None:
            return self._manual_market_metrics()
        if self.config.offline:
            return EquityMarketMetrics(
                price=None, ma200=None, distance_from_200d_ma_pct=None, weekly_rsi=None,
                monthly_ema20=None, monthly_trend="", as_of=None, missing_fields=_market_missing_fields(),
            )

        try:
            result = self._equity_data_loader(self.config.symbol, timeframes=("1d",), refresh_cache=False)
        except Exception:
            return EquityMarketMetrics(
                price=None, ma200=None, distance_from_200d_ma_pct=None, weekly_rsi=None,
                monthly_ema20=None, monthly_trend="", as_of=None, missing_fields=_market_missing_fields(),
            )

        daily = result.frames.get("1d") if hasattr(result, "frames") else None
        missing_fields: list[str] = []
        price = ma200 = distance_pct = weekly_rsi = monthly_ema20 = None
        monthly_trend = ""
        as_of = None

        if daily is None or daily.empty:
            return EquityMarketMetrics(
                price=None, ma200=None, distance_from_200d_ma_pct=None, weekly_rsi=None,
                monthly_ema20=None, monthly_trend="", as_of=None, missing_fields=_market_missing_fields(),
            )

        daily = daily.copy()
        price = _round(float(daily.iloc[-1]["close"]))
        as_of = pd.Timestamp(daily.iloc[-1]["timestamp"]).isoformat()
        if len(daily) >= 200:
            ma200 = _round(float(daily["close"].astype(float).rolling(200, min_periods=200).mean().iloc[-1]))
            distance_pct = _round(((price - ma200) / ma200) * 100) if ma200 and ma200 > 0 else None
        else:
            missing_fields.append("distance_from_200d_ma")

        weekly = _resample(daily, "W")
        if weekly is None or len(weekly) < 20:
            missing_fields.append("weekly_rsi")
        else:
            weekly_rsi = _round(float(calculate_rsi(weekly["close"].astype(float), 14).iloc[-1]))

        monthly = _resample(daily, "ME")
        if monthly is None or len(monthly) < 25:
            missing_fields.append("monthly_trend")
        else:
            monthly_close = monthly["close"].astype(float)
            monthly_ema20 = _round(float(calculate_ema(monthly_close, 20).iloc[-1]))
            monthly_trend = _classify_monthly_trend(monthly_close, monthly_ema20)

        return EquityMarketMetrics(
            price=price, ma200=ma200, distance_from_200d_ma_pct=distance_pct, weekly_rsi=weekly_rsi,
            monthly_ema20=monthly_ema20, monthly_trend=monthly_trend, as_of=as_of, missing_fields=missing_fields,
        )

    def _manual_market_metrics(self) -> EquityMarketMetrics:
        price = self.config.current_price
        ma200 = self.config.ma200
        monthly_ema20 = self.config.monthly_ema20
        distance_pct = _round(((price - ma200) / ma200) * 100) if price is not None and ma200 and ma200 > 0 else None
        monthly_trend = self.config.monthly_trend or ""
        missing_fields = []
        if distance_pct is None:
            missing_fields.append("distance_from_200d_ma")
        if self.config.weekly_rsi is None:
            missing_fields.append("weekly_rsi")
        if not monthly_trend:
            missing_fields.append("monthly_trend")
        return EquityMarketMetrics(
            price=_round(price), ma200=_round(ma200), distance_from_200d_ma_pct=distance_pct,
            weekly_rsi=_round(self.config.weekly_rsi), monthly_ema20=_round(monthly_ema20),
            monthly_trend=monthly_trend, as_of=datetime.now(UTC).isoformat(), missing_fields=missing_fields,
        )

    def _fundamentals(self) -> dict[str, Any]:
        return self._fundamentals_provider.fetch(self.config.symbol, offline=self.config.offline)

    def _factor_scores(self, market: EquityMarketMetrics, fundamentals: dict[str, Any]) -> dict[str, dict[str, Any]]:
        revenue_growth = fundamentals.get("quarterly_revenue_growth_yoy")
        earnings_growth = fundamentals.get("quarterly_earnings_growth_yoy")
        return {
            "valuation_pe": factor_payload(
                fundamentals.get("pe_forward") or fundamentals.get("pe_trailing"),
                score_valuation_pe(fundamentals.get("pe_forward") or fundamentals.get("pe_trailing")),
                EQUITY_INVESTOR_WEIGHTS["valuation_pe"],
            ),
            "valuation_peg": factor_payload(
                fundamentals.get("peg_ratio"),
                score_valuation_peg(fundamentals.get("peg_ratio")),
                EQUITY_INVESTOR_WEIGHTS["valuation_peg"],
            ),
            "valuation_pb": factor_payload(
                fundamentals.get("price_to_book"),
                score_valuation_pb(fundamentals.get("price_to_book")),
                EQUITY_INVESTOR_WEIGHTS["valuation_pb"],
            ),
            "fcf_yield": factor_payload(
                fundamentals.get("fcf_yield_pct"),
                score_fcf_yield(fundamentals.get("fcf_yield_pct")),
                EQUITY_INVESTOR_WEIGHTS["fcf_yield"],
                "value_pct",
            ),
            "quality_roe": factor_payload(
                fundamentals.get("return_on_equity_ttm"),
                score_quality_roe(fundamentals.get("return_on_equity_ttm")),
                EQUITY_INVESTOR_WEIGHTS["quality_roe"],
            ),
            "growth_consistency": factor_payload(
                {"revenue_growth_yoy": revenue_growth, "earnings_growth_yoy": earnings_growth},
                score_growth_consistency(revenue_growth, earnings_growth),
                EQUITY_INVESTOR_WEIGHTS["growth_consistency"],
            ),
            "distance_from_200d_ma": factor_payload(
                market.distance_from_200d_ma_pct,
                score_distance_from_200d_ma(market.distance_from_200d_ma_pct),
                EQUITY_INVESTOR_WEIGHTS["distance_from_200d_ma"],
                "value_pct",
            ),
        }

    def _rationale(
        self,
        investor_score: int,
        investor_band: str,
        market: EquityMarketMetrics,
        fundamentals: dict[str, Any],
        data_confidence: str,
    ) -> list[str]:
        rationale = [
            f"{self.config.symbol} growth-bucket investor band is {investor_band} with a score of {investor_score}.",
            "This is deterministic investor guidance based on valuation (P/E, PEG, P/B), FCF yield, quality, growth consistency, and trend discount.",
        ]
        if fundamentals.get("missing"):
            rationale.append("Fundamentals data is unavailable; set ALPHA_VANTAGE_API_KEY or a data/cache/fundamentals_<symbol>.json override to improve confidence.")
        if market.distance_from_200d_ma_pct is not None:
            rationale.append(f"{self.config.symbol} is {market.distance_from_200d_ma_pct}% from its 200D moving average.")
        if data_confidence != "HIGH":
            rationale.append(f"Data confidence is {data_confidence}; avoid aggressive sizing until missing context is resolved.")
        return rationale


def run_equity_investor_agent(
    config: EquityInvestorConfig,
    equity_data_loader: Callable[..., EquityDataResult] | None = None,
    fundamentals_provider: EquityFundamentalsProvider | None = None,
) -> dict[str, Any]:
    agent = EquityInvestorAgent(config, equity_data_loader=equity_data_loader, fundamentals_provider=fundamentals_provider)
    return agent.run()


def is_core_etf_symbol(symbol: str) -> bool:
    return symbol.upper() in CORE_ETF_PROFILES


def _default_bucket_for_symbol(symbol: str) -> str:
    return "core" if is_core_etf_symbol(symbol) else "growth"


def _resample(daily: pd.DataFrame, rule: str) -> pd.DataFrame | None:
    working = daily.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    resampled = (
        working.set_index("timestamp")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return resampled


def _classify_monthly_trend(monthly_close: pd.Series, monthly_ema20: float | None) -> str:
    if monthly_ema20 is None or monthly_ema20 <= 0:
        return ""
    latest_close = float(monthly_close.iloc[-1])
    distance_pct = ((latest_close - monthly_ema20) / monthly_ema20) * 100
    if latest_close < monthly_ema20:
        return "BELOW_EMA20"
    if abs(distance_pct) <= 5:
        return "NEAR_EMA20"
    if distance_pct > 80:
        return "EXTREMELY_EXTENDED"
    return "BULLISH_ABOVE_EMA20"


def _market_missing_fields() -> list[str]:
    return ["price", "distance_from_200d_ma", "weekly_rsi", "monthly_trend"]


def _fundamentals_missing_fields(fundamentals: dict[str, Any]) -> list[str]:
    if fundamentals.get("missing"):
        return ["pe", "peg", "price_to_book", "fcf_yield", "return_on_equity", "growth_consistency"]
    return []


def _data_quality_confidence(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "HIGH"
    if len(missing_fields) <= 2:
        return "MEDIUM"
    return "LOW"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
