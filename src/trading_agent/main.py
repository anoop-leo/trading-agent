"""Command-line orchestration for the Phase 1 signal engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from decision.decision_engine import (
    DecisionInput,
    DecisionResult,
    apply_multi_timeframe_alignment,
    make_decision,
)
from scoring.market_regime_skill import MarketRegimeResult, calculate_market_regime
from scoring.multi_timeframe_skill import (
    TimeframeSignal,
    analyze_multi_timeframe,
    chart_annotation,
)
from scoring.risk_reward_skill import RiskRewardResult, calculate_risk_reward
from scoring.setup_detection_skill import SetupInput, SetupResult, detect_setup
from scoring.support_resistance_skill import SupportResistanceResult, calculate_support_resistance
from trading_agent.config import AgentConfig
from trading_agent.data import BinanceKlineProvider, BybitKlineProvider
from trading_agent.indicators import add_indicators
from trading_agent.journal import update_signal_journal
from trading_agent.models import MarketDataProvider, SignalScores
from trading_agent.output import build_output_payload, macd_direction, write_chart, write_json
from trading_agent.scoring import calculate_recent_swing_high, calculate_recent_swing_low, calculate_scores, calculate_volume_ratio


ChartWriter = Callable[[pd.DataFrame, Path, str, str], Path]


@dataclass(frozen=True)
class TimeframeAnalysis:
    timeframe: str
    indicators: pd.DataFrame
    scores: SignalScores
    decision: DecisionResult
    support_resistance: SupportResistanceResult
    risk_reward: RiskRewardResult
    market_regime: MarketRegimeResult
    setup: SetupResult


def calculate_trade_quality(
    indicator_frame: pd.DataFrame,
    support_resistance: SupportResistanceResult | None = None,
) -> tuple[SupportResistanceResult, RiskRewardResult, MarketRegimeResult]:
    """Calculate Phase 1.1 trade-quality skills from the latest indicator frame."""

    latest = indicator_frame.iloc[-1]
    current_price = float(latest["close"])
    if support_resistance is None:
        support_resistance = calculate_support_resistance(indicator_frame, current_price)
    risk_reward = calculate_risk_reward(
        current_price=current_price,
        support=support_resistance.support,
        resistance=support_resistance.resistance,
    )
    market_regime = calculate_market_regime(
        current_price=current_price,
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
    )
    return support_resistance, risk_reward, market_regime


def build_decision_input(
    symbol: str,
    position_mode: str,
    indicator_frame: pd.DataFrame,
    scores: SignalScores,
    support_resistance: SupportResistanceResult,
    risk_reward: RiskRewardResult,
    market_regime: MarketRegimeResult,
    setup: SetupResult,
) -> DecisionInput:
    """Build Decision Engine v2 input from the latest indicator frame."""

    latest = indicator_frame.iloc[-1]
    return DecisionInput(
        symbol=symbol,
        position_mode=position_mode,
        trend_score=scores.trend_score,
        momentum_score=scores.momentum_score,
        volume_score=scores.volume_score,
        bottom_score=scores.bottom_score,
        sr_score=scores.sr_score,
        rr_score=scores.rr_score,
        regime_score=scores.regime_score,
        current_price=float(latest["close"]),
        ema20=float(latest["ema_20"]),
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
        recent_swing_high=calculate_recent_swing_high(indicator_frame),
        recent_swing_low=calculate_recent_swing_low(indicator_frame),
        rr_ratio=risk_reward.rr_ratio,
        market_regime=market_regime.market_regime.value,
        support=support_resistance.support,
        setup=setup.setup.value,
        setup_score=setup.setup_score,
        setup_confidence=setup.setup_confidence,
        volume_ratio=calculate_volume_ratio(latest),
        rsi=float(latest["rsi_14"]),
        macd=macd_direction(latest),
    )


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    position_mode: str,
    ohlcv: pd.DataFrame,
) -> TimeframeAnalysis:
    """Run the Phase 1 deterministic signal pipeline for one timeframe."""

    return analyze_indicator_frame(symbol, timeframe, position_mode, add_indicators(ohlcv))


def analyze_indicator_frame(
    symbol: str,
    timeframe: str,
    position_mode: str,
    indicators: pd.DataFrame,
    support_resistance: SupportResistanceResult | None = None,
) -> TimeframeAnalysis:
    """Run the Phase 1 deterministic signal pipeline from precomputed indicators."""

    support_resistance, risk_reward, market_regime = calculate_trade_quality(indicators, support_resistance)
    base_scores = calculate_scores(indicators)
    scores = replace(
        base_scores,
        sr_score=support_resistance.sr_score,
        rr_score=risk_reward.rr_score,
        regime_score=market_regime.regime_score,
    )
    latest = indicators.iloc[-1]
    setup = detect_setup(
        SetupInput(
            price=float(latest["close"]),
            ema20=float(latest["ema_20"]),
            ema50=float(latest["ema_50"]),
            ema200=float(latest["ema_200"]),
            support=support_resistance.support,
            resistance=support_resistance.resistance,
            volume_ratio=calculate_volume_ratio(latest),
            trend_score=scores.trend_score,
            momentum_score=scores.momentum_score,
            bottom_score=scores.bottom_score,
            market_regime=market_regime.market_regime.value,
            rsi=float(latest["rsi_14"]),
        )
    )
    decision_input = build_decision_input(
        symbol,
        position_mode,
        indicators,
        scores,
        support_resistance,
        risk_reward,
        market_regime,
        setup,
    )
    decision = make_decision(decision_input)
    return TimeframeAnalysis(
        timeframe=timeframe,
        indicators=indicators,
        scores=scores,
        decision=decision,
        support_resistance=support_resistance,
        risk_reward=risk_reward,
        market_regime=market_regime,
        setup=setup,
    )


def build_timeframe_signal(analysis: TimeframeAnalysis) -> TimeframeSignal:
    """Build a compact signal snapshot for multi-timeframe alignment."""

    latest = analysis.indicators.iloc[-1]
    return TimeframeSignal(
        timeframe=analysis.timeframe,
        trend_score=analysis.scores.trend_score,
        momentum_score=analysis.scores.momentum_score,
        volume_score=analysis.scores.volume_score,
        bottom_score=analysis.scores.bottom_score,
        sr_score=analysis.scores.sr_score,
        rr_score=analysis.scores.rr_score,
        regime_score=analysis.scores.regime_score,
        setup=analysis.setup.setup.value,
        setup_confidence=analysis.setup.setup_confidence,
        decision=analysis.decision.decision.value,
        price=float(latest["close"]),
        rsi=float(latest["rsi_14"]),
        macd=macd_direction(latest),
        ema20=float(latest["ema_20"]),
        ema50=float(latest["ema_50"]),
        ema200=float(latest["ema_200"]),
        market_regime=analysis.market_regime.market_regime.value,
    )


def _fetch_timeframes(config: AgentConfig) -> tuple[str, ...]:
    timeframes = [config.interval]
    for timeframe in config.timeframes:
        if timeframe not in timeframes:
            timeframes.append(timeframe)
    return tuple(timeframes)


def build_market_data_provider(config: AgentConfig) -> MarketDataProvider:
    """Build the configured public market-data provider."""

    if config.resolved_market_data_source == "BYBIT":
        return BybitKlineProvider(
            base_url=config.bybit_base_url,
            timeout_seconds=config.request_timeout_seconds,
        )
    return BinanceKlineProvider(
        base_url=config.binance_base_url,
        timeout_seconds=config.request_timeout_seconds,
    )


def run(
    config: AgentConfig,
    provider: MarketDataProvider | None = None,
    chart_writer: ChartWriter = write_chart,
) -> dict[str, Any]:
    """Run one local Phase 1 signal-generation cycle."""

    market_data_provider = provider or build_market_data_provider(config)

    analyses: dict[str, TimeframeAnalysis] = {}
    for timeframe in _fetch_timeframes(config):
        ohlcv = market_data_provider.fetch_ohlcv(
            symbol=config.symbol,
            interval=timeframe,
            limit=config.history_limit,
        )
        analyses[timeframe] = analyze_timeframe(config.symbol, timeframe, config.position_mode, ohlcv)

    primary_analysis = analyses[config.interval]
    multi_timeframe = analyze_multi_timeframe(
        {timeframe: build_timeframe_signal(analyses[timeframe]) for timeframe in config.timeframes}
    )
    final_decision = apply_multi_timeframe_alignment(
        primary_analysis.decision.decision,
        multi_timeframe.alignment.value,
        config.position_mode,
    )
    payload = build_output_payload(
        config,
        primary_analysis.indicators,
        primary_analysis.scores,
        primary_analysis.decision,
        primary_analysis.support_resistance,
        primary_analysis.risk_reward,
        primary_analysis.market_regime,
        primary_analysis.setup,
        multi_timeframe,
        final_decision,
    )
    journal_frame = analyses["1d"].indicators if "1d" in analyses else primary_analysis.indicators
    journal_path, journal_status = update_signal_journal(payload, journal_frame, config.output_dir)
    payload["signal_journal"] = {
        "path": str(journal_path),
        "inserted": journal_status["inserted"],
        "evaluated_count": journal_status["evaluated_count"],
    }

    write_json(payload, config.output_dir)
    chart_writer(
        primary_analysis.indicators,
        config.output_dir,
        config.symbol,
        chart_annotation(multi_timeframe),
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 technical signal engine.")
    _add_signal_arguments(parser)
    return parser


def _add_signal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTCUSDT", help="Supported symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Kline interval.")
    parser.add_argument(
        "--market-data-source",
        default="AUTO",
        choices=["AUTO", "BINANCE", "BYBIT"],
        help="Public market data source. AUTO uses Bybit for HYPEUSDT and Binance otherwise.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Multi-timeframe intervals, e.g. --timeframes 1h 4h 1d.",
    )
    parser.add_argument("--history-limit", type=int, default=500, help="Number of candles to fetch.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Artifact output directory.")
    parser.add_argument(
        "--position-mode",
        default="NO_POSITION",
        choices=["NO_POSITION", "HOLDING"],
        help="Decision terminology mode.",
    )


def build_backtest_parser() -> argparse.ArgumentParser:
    from backtesting.profiles import PROFILE_NAMES

    parser = argparse.ArgumentParser(description="Run the Phase 1.5 historical backtesting engine.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Backtest symbol. BTCUSDT is the Phase 1.5 default.")
    parser.add_argument("--start", default="2017-01-01", help="Start date, e.g. 2017-01-01.")
    parser.add_argument("--end", default="latest", help="End date or latest.")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=[*PROFILE_NAMES, "all"],
        help="Strategy profile to backtest. Use all to write profile_comparison.json.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "aggressive",
            "broker_cost_validation",
            "cross_asset_validation",
            "trend_holding",
            "regime_gated_trend_holding",
            "regime_gated_portfolio_governor",
            "hybrid_trend_rider",
            "hybrid_optimization",
            "hybrid_conservative",
            "hybrid_balanced",
            "hybrid_aggressive",
            "exit_optimization",
        ],
        help="Run a named strategy research workflow.",
    )
    parser.add_argument(
        "--stop-type",
        default="fixed",
        choices=["fixed", "atr", "swing_low", "support_zone"],
        help="Stop placement model for aggressive backtests.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Assets for cross-asset validation. Default: BTCUSDT ETHUSDT SOLUSDT SPY QQQ.",
    )
    parser.add_argument(
        "--include-optional-assets",
        action="store_true",
        help="Include optional cross-asset validation assets TQQQ and NVDA.",
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial virtual capital.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Historical candle cache directory.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing candle cache files and rebuild the requested range.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Backtest artifact output directory.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Backtest timeframes. Default: 1h 4h 1d.",
    )
    parser.add_argument("--primary-timeframe", default="1h", help="Primary replay timeframe.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print replay progress every N primary candles. Use 0 to disable interval progress.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable backtest progress logging.")
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run Phase 1.6 benchmark strategy comparison.",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run Phase 1.6 benchmark, regime, and filter-attribution research.",
    )
    parser.add_argument(
        "--trend-participation",
        action="store_true",
        help="Run Phase 1.7 trend participation research.",
    )
    parser.add_argument(
        "--profit-capture",
        action="store_true",
        help="Run Phase 1.8 trade duration and profit-capture analysis.",
    )
    parser.add_argument(
        "--trend-rider",
        action="store_true",
        help="Run Phase 1.9 Trend Rider analysis.",
    )
    return parser


def build_audit_fees_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit backtest fee, slippage, and net PnL reporting.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Backtest symbol. BTCUSDT is the default.")
    parser.add_argument("--start", default="2017-01-01", help="Start date, e.g. 2017-01-01.")
    parser.add_argument("--end", default="latest", help="End date or latest.")
    parser.add_argument(
        "--strategy",
        default="aggressive",
        choices=["aggressive"],
        help="Strategy to audit. Phase 1.17B supports aggressive.",
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial virtual capital.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Historical candle cache directory.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing candle cache files and rebuild the requested range.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Audit artifact output directory.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Backtest timeframes. Default: 1h 4h 1d.",
    )
    parser.add_argument("--primary-timeframe", default="1h", help="Primary replay timeframe.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print replay progress every N primary candles. Use 0 to disable interval progress.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable audit progress logging.")
    return parser


def build_validate_equities_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate equity data adapters and rerun Phase 1.16 equity checks.")
    parser.add_argument("--start", default="2018-01-01", help="Start date, e.g. 2018-01-01.")
    parser.add_argument("--end", default="latest", help="End date or latest.")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["SPY", "QQQ"],
        help="Equity assets to validate. Default: SPY QQQ.",
    )
    parser.add_argument(
        "--include-optional-assets",
        action="store_true",
        help="Include optional equity assets IWM, DIA, TQQQ, and NVDA.",
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial virtual capital.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Historical candle cache directory.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing equity cache files and rebuild the requested range.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Validation artifact output directory.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Validation timeframes. Default: 1h 4h 1d.",
    )
    parser.add_argument("--primary-timeframe", default="1h", help="Primary replay timeframe.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print replay progress every N primary candles. Use 0 to disable interval progress.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable validation progress logging.")
    return parser


def build_coinbase_execution_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a read-only Coinbase Advanced execution-cost audit.")
    parser.add_argument(
        "--product",
        "--product-id",
        dest="product_id",
        default="BTC-USD",
        help="Coinbase product id or common symbol. Examples: BTC-USD, BTC-USDT, BTC/USD.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Coinbase fee rate as a ratio. Default: 0.001 for 0.10%% per side.",
    )
    parser.add_argument(
        "--order-sizes",
        nargs="+",
        type=float,
        default=[100.0, 500.0, 1000.0, 2500.0, 5000.0],
        help="USD market-order sizes to estimate. Default: 100 500 1000 2500 5000.",
    )
    parser.add_argument(
        "--intended-order-size",
        type=float,
        default=2500.0,
        help="USD order size used for top-level acceptance criteria. Default: 2500.",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=24.0,
        help="Audit duration. Default: 24 hours.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Seconds between order-book samples. Default: 300.",
    )
    parser.add_argument(
        "--order-book-limit",
        type=int,
        default=50,
        help="Coinbase product-book depth limit. Default: 50.",
    )
    parser.add_argument(
        "--coinbase-base-url",
        default="https://api.coinbase.com/api/v3/brokerage",
        help="Coinbase Advanced brokerage API base URL.",
    )
    parser.add_argument(
        "--bearer-token-env",
        default="COINBASE_ADVANCED_READ_ONLY_TOKEN",
        help="Environment variable containing an optional read-only Coinbase bearer token.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Coinbase request timeout.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Audit artifact output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement if a timestamped audit output file already exists.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable Coinbase audit progress logging.")
    return parser


def build_merge_coinbase_execution_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Coinbase execution audit CSV samples.")
    parser.add_argument(
        "--input-csv",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/coinbase_orderbook_samples.csv"),
            Path("outputs/coinbase_orderbook_samples_24h.csv"),
        ],
        help="Coinbase order-book sample CSV files to merge.",
    )
    parser.add_argument(
        "--failed-samples",
        type=int,
        default=0,
        help="Total failed sample count across the merged collections. Default: 0.",
    )
    parser.add_argument(
        "--intended-order-size",
        type=float,
        default=2500.0,
        help="USD order size column used for final cost criteria. Default: 2500.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Final audit output directory.")
    return parser


def build_shadow_coinbase_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Coinbase Advanced read-only shadow trading.")
    parser.add_argument("--product", "--product-id", dest="product_id", default="BTC-USD", help="Coinbase product id.")
    parser.add_argument("--duration-days", type=float, default=30.0, help="Shadow trading duration. Default: 30 days.")
    parser.add_argument(
        "--cycle-interval-seconds",
        type=float,
        default=3600.0,
        help="Seconds between shadow strategy cycles. Default: 3600.",
    )
    parser.add_argument("--cycle-limit", type=int, default=None, help="Optional cycle count cap for smoke tests.")
    parser.add_argument(
        "--target-signal-count",
        type=int,
        default=None,
        help="Stop once the cumulative enriched signal journal reaches this many signals.",
    )
    parser.add_argument(
        "--no-resume-signal-collection",
        action="store_true",
        help="Start a fresh enriched signal collection instead of loading signal_journal_v2.json.",
    )
    parser.add_argument("--history-limit", type=int, default=220, help="Candles per timeframe to fetch.")
    parser.add_argument("--initial-shadow-capital", type=float, default=10000.0, help="Initial virtual capital.")
    parser.add_argument("--intended-order-size", type=float, default=2500.0, help="Maximum shadow position size.")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Coinbase fee rate assumption.")
    parser.add_argument(
        "--max-all-in-cost-per-side",
        type=float,
        default=0.0015,
        help="Maximum acceptable measured all-in execution cost per side.",
    )
    parser.add_argument(
        "--coinbase-base-url",
        default="https://api.coinbase.com/api/v3/brokerage",
        help="Coinbase Advanced brokerage API base URL.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Coinbase request timeout.")
    parser.add_argument("--order-book-limit", type=int, default=50, help="Coinbase product-book depth limit.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Shadow artifact output directory.")
    parser.add_argument("--quiet", action="store_true", help="Disable shadow cycle progress logging.")
    parser.add_argument(
        "--enable-risk-engine",
        action="store_true",
        help="Gate shadow trading entries through the live risk engine (bucket caps, drawdown circuit breaker).",
    )
    parser.add_argument(
        "--risk-config-path",
        type=Path,
        default=None,
        help="Path to risk_config.json. Defaults to config/risk_config.json. Only used with --enable-risk-engine.",
    )
    parser.add_argument(
        "--portfolio-state-path",
        type=Path,
        default=None,
        help="Path to portfolio_state.json. Defaults to data/portfolio_state.json. Only used with --enable-risk-engine.",
    )
    return parser


def build_false_avoid_analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze AVOID LONG signals that later moved higher.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Research artifact output directory.")
    return parser


def build_collect_shadow_signals_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect enriched Coinbase shadow signal snapshots.")
    parser.add_argument("--product", "--product-id", dest="product_id", default="BTC-USD", help="Coinbase product id.")
    parser.add_argument("--target-signals", type=int, default=50, help="Cumulative enriched signal target.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=3600.0,
        help="Seconds between enriched signal snapshots.",
    )
    parser.add_argument("--history-limit", type=int, default=220, help="Candles per timeframe to fetch.")
    parser.add_argument("--intended-order-size", type=float, default=2500.0, help="Reference shadow order size.")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Coinbase fee rate assumption.")
    parser.add_argument(
        "--max-all-in-cost-per-side",
        type=float,
        default=0.0015,
        help="Maximum acceptable measured all-in execution cost per side.",
    )
    parser.add_argument(
        "--coinbase-base-url",
        default="https://api.coinbase.com/api/v3/brokerage",
        help="Coinbase Advanced brokerage API base URL.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Coinbase request timeout.")
    parser.add_argument("--order-book-limit", type=int, default=50, help="Coinbase product-book depth limit.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Collection artifact output directory.")
    parser.add_argument("--reset", action="store_true", help="Reset v2 signal collection files before collecting.")
    parser.add_argument("--quiet", action="store_true", help="Disable collection progress logging.")
    return parser


def build_screen_airdrop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only airdrop token due-diligence screener. Reads public data ABOUT a "
        "token; never connects a wallet, signs, or claims.",
    )
    parser.add_argument("--address", "--contract", dest="address", default=None, help="Token contract address (EVM 0x...).")
    parser.add_argument(
        "--chain",
        default="ethereum",
        help="Chain name: ethereum, bsc, polygon, arbitrum, optimism, base, avalanche, and more.",
    )
    parser.add_argument("--symbol", default=None, help="Optional token symbol label for the report.")
    parser.add_argument("--timeout-seconds", type=float, default=15.0, help="Security-data request timeout.")
    parser.add_argument("--json", action="store_true", help="Emit the structured report as JSON instead of text.")
    return parser


def _run_screen_airdrop_command(argv: Sequence[str] | None = None) -> int:
    from airdrop.screener import format_risk_report, screen_token
    from airdrop.token_security_provider import TokenSecurityProvider

    args = build_screen_airdrop_parser().parse_args(argv)
    if not args.address:
        print(
            "screen-airdrop requires --address (a token contract address). A symbol alone is "
            "ambiguous and unsafe to screen.",
            file=sys.stderr,
        )
        return 2

    provider = TokenSecurityProvider(timeout_seconds=args.timeout_seconds)
    envelope = provider.fetch(args.address, args.chain)
    report = screen_token(envelope, symbol=args.symbol)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_risk_report(report))
    return 0


def build_investor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic investor accumulation scoring.")
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Investor asset. Supports BTC, ETH, SOL, XRP, AVAX/AVX, LINK/CHAINLINK, ONDO, and HYPE/HYPER.",
    )
    parser.add_argument("--offline", action="store_true", help="Run without network calls and mark unavailable data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Investor report output directory.")
    parser.add_argument(
        "--binance-base-url",
        default="https://api.binance.com",
        help="Binance public market data base URL.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Public data request timeout.")
    parser.add_argument(
        "--portfolio-risk-profile",
        default="BALANCED",
        choices=["CONSERVATIVE", "BALANCED", "AGGRESSIVE", "HIGH_CONVICTION_CRYPTO"],
        help="Portfolio risk profile used for institutional BTC allocation guidance.",
    )
    parser.add_argument(
        "--current-btc-allocation-pct",
        type=float,
        default=None,
        help="Current BTC allocation as a percentage of the portfolio.",
    )
    parser.add_argument(
        "--target-btc-allocation-pct",
        type=float,
        default=None,
        help="Optional BTC target allocation override.",
    )
    parser.add_argument(
        "--max-btc-allocation-pct",
        type=float,
        default=None,
        help="Optional BTC maximum allocation override.",
    )
    parser.add_argument("--current-btc", type=float, default=1.13059494, help="Current BTC held for goal planning.")
    parser.add_argument("--target-btc", type=float, default=2.0, help="Target BTC position for goal planning.")
    parser.add_argument(
        "--target-sell-price",
        type=float,
        default=500000.0,
        help="Future BTC price used for the staged planning framework.",
    )
    parser.add_argument(
        "--planned-sell-btc",
        type=float,
        default=1.0,
        help="Future BTC amount included in the staged planning framework.",
    )
    parser.add_argument(
        "--retain-btc",
        type=float,
        default=1.0,
        help="BTC intended to be retained long term after the staged planning framework.",
    )
    parser.add_argument("--monthly-dca-usd", type=float, default=None, help="Optional monthly DCA amount in USD.")
    parser.add_argument(
        "--lump-sum-available-usd",
        type=float,
        default=None,
        help="Optional available lump-sum capital for planning context.",
    )
    parser.add_argument(
        "--reference-price-for-dip",
        type=float,
        default=None,
        help="Optional reference BTC price used for dip reserve drawdown triggers.",
    )
    parser.add_argument(
        "--thesis-risk-level",
        default="MODERATE",
        choices=["LOW", "MODERATE", "HIGH"],
        help="Non-BTC crypto thesis risk level used by the generic crypto investor agent.",
    )
    parser.add_argument(
        "--thesis-risk-flags",
        nargs="*",
        default=[],
        help="Optional non-BTC crypto thesis risk notes, for example oracle_competition token_unlocks.",
    )
    parser.add_argument("--asset-name", default=None, help="Optional non-BTC crypto asset name override.")
    parser.add_argument("--sector", default=None, help="Optional non-BTC crypto sector override.")
    parser.add_argument(
        "--market-data-source",
        default=None,
        choices=["BINANCE", "BYBIT"],
        help="Optional non-BTC crypto market data source override.",
    )
    parser.add_argument("--current-price", "--price", dest="current_price", type=float, default=None)
    parser.add_argument("--ma200", type=float, default=None, help="Manual 200-day moving average.")
    parser.add_argument("--weekly-rsi", type=float, default=None, help="Manual weekly RSI.")
    parser.add_argument("--monthly-ema20", type=float, default=None, help="Manual monthly EMA20.")
    parser.add_argument("--monthly-trend", default=None, help="Manual monthly trend label.")
    parser.add_argument("--recent-cycle-high", type=float, default=None, help="Manual recent cycle high.")
    parser.add_argument(
        "--quote-volume-usd",
        "--quote-volume",
        dest="quote_volume_usd",
        type=float,
        default=None,
        help="Manual current quote volume in USD.",
    )
    parser.add_argument(
        "--average-quote-volume-usd",
        "--average-quote-volume",
        dest="average_quote_volume_usd",
        type=float,
        default=None,
        help="Manual average quote volume in USD.",
    )
    parser.add_argument("--atr-pct", type=float, default=None, help="Manual ATR percentage.")
    parser.add_argument(
        "--asset-class",
        default="AUTO",
        choices=["AUTO", "CRYPTO", "EQUITY"],
        help="Force crypto or equity dispatch. AUTO recognizes known core ETFs (SPY, QQQ, VTI, IWM, DIA) as equity.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        choices=["core", "growth", "speculative"],
        help="Override the portfolio bucket used for equity risk-engine evaluation.",
    )
    parser.add_argument(
        "--default-position-usd",
        type=float,
        default=2_000.0,
        help="Baseline position size before conviction scaling and risk-engine caps.",
    )
    parser.add_argument(
        "--risk-config-path",
        type=Path,
        default=None,
        help="Path to risk_config.json. Defaults to config/risk_config.json.",
    )
    parser.add_argument(
        "--portfolio-state-path",
        type=Path,
        default=None,
        help="Path to portfolio_state.json. Defaults to data/portfolio_state.json.",
    )
    parser.add_argument(
        "--skip-risk-engine",
        action="store_true",
        help="Skip the live risk engine gate and return the raw investor agent payload.",
    )
    return parser


def _run_signal_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = AgentConfig(
        symbol=args.symbol,
        interval=args.interval,
        history_limit=args.history_limit,
        output_dir=args.output_dir,
        position_mode=args.position_mode,
        timeframes=args.timeframes,
        market_data_source=args.market_data_source,
    )
    return run(config)


def _run_validate_equities_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from backtesting.backtest_engine import BacktestConfig
    from backtesting.cross_asset_validation import run_equity_validation

    args = build_validate_equities_parser().parse_args(argv)
    config = BacktestConfig(
        symbol="SPY",
        start=args.start,
        end=args.end,
        primary_timeframe=args.primary_timeframe,
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
        profile="aggressive",
        refresh_cache=args.refresh_cache,
    )
    progress_callback = None if args.quiet else _print_backtest_progress
    payload = run_equity_validation(
        config,
        equity_assets=tuple(asset.upper() for asset in args.assets),
        include_optional=args.include_optional_assets,
        progress_callback=progress_callback,
    )
    report = payload["equity_validation_report"]
    return {
        "phase": "1.16B",
        "strategy": report["strategy"],
        "data_validation": report["data_validation"],
        "equity_metrics": report["equity_metrics"],
        "equity_rankings": report["equity_rankings"],
        "updated_cross_asset_rankings": report["updated_cross_asset_rankings"],
        "equity_vs_crypto_analysis": report["equity_vs_crypto_analysis"],
        "success_criteria": report["success_criteria"],
        "trend_following_assessment": report["trend_following_assessment"],
        "recommended_production_assets": report["recommended_production_assets"],
        "artifacts": payload["artifacts"],
    }


def _run_audit_fees_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from backtesting.backtest_engine import BacktestConfig
    from backtesting.fee_slippage_audit import run_fee_slippage_audit

    args = build_audit_fees_parser().parse_args(argv)
    config = BacktestConfig(
        symbol=args.symbol.upper(),
        start=args.start,
        end=args.end,
        primary_timeframe=args.primary_timeframe,
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
        profile="aggressive",
        refresh_cache=args.refresh_cache,
        stop_type="fixed",
    )
    progress_callback = None if args.quiet else _print_backtest_progress
    payload = run_fee_slippage_audit(
        config,
        strategy=args.strategy,
        progress_callback=progress_callback,
    )
    report = payload["fee_slippage_audit"]
    return {
        "symbol": report["symbol"],
        "phase": report["phase"],
        "strategy": report["strategy"],
        "trade_count": report["trade_count"],
        "fees_modeled": report["fees_modeled"],
        "slippage_modeled": report["slippage_modeled"],
        "fee_rate": report["fee_rate"],
        "slippage_rate": report["slippage_rate"],
        "net_pnl_equals_pnl": report["net_pnl_equals_pnl"],
        "final_equity_reconciles_to_net_pnl": report["final_equity_reconciles_to_net_pnl"],
        "total_gross_pnl_before_costs": report["total_gross_pnl_before_costs"],
        "total_gross_pnl_after_slippage": report["total_gross_pnl_after_slippage"],
        "total_fees": report["total_fees"],
        "total_slippage_cost": report["total_slippage_cost"],
        "total_net_pnl": report["total_net_pnl"],
        "final_equity_reconciliation_delta": report["final_equity_reconciliation_delta"],
        "artifacts": payload["artifacts"],
    }


def _run_coinbase_execution_audit_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from data.coinbase_execution_cost_audit import (
        CoinbaseExecutionAuditConfig,
        run_coinbase_execution_cost_audit,
    )

    args = build_coinbase_execution_audit_parser().parse_args(argv)
    bearer_token = os.environ.get(args.bearer_token_env) if args.bearer_token_env else None
    config = CoinbaseExecutionAuditConfig(
        product_id=args.product_id,
        fee_rate=args.fee_rate,
        order_sizes_usd=tuple(args.order_sizes),
        intended_order_size_usd=args.intended_order_size,
        duration_hours=args.duration_hours,
        interval_seconds=args.interval_seconds,
        output_dir=args.output_dir,
        base_url=args.coinbase_base_url,
        timeout_seconds=args.timeout_seconds,
        order_book_limit=args.order_book_limit,
        bearer_token=bearer_token,
        overwrite=args.overwrite,
    )
    progress_callback = None if args.quiet else _print_coinbase_execution_audit_progress
    payload = run_coinbase_execution_cost_audit(config, progress_callback=progress_callback)
    report = payload["coinbase_execution_cost_audit"]
    return {
        "phase": report["phase"],
        "mode": report["mode"],
        "collection_mode": report["collection_mode"],
        "overwrite_protection_enabled": report["overwrite_protection_enabled"],
        "product_id": report["product_id"],
        "strategy_trading_enabled": report["strategy_trading_enabled"],
        "automated_live_trading_enabled": report["automated_live_trading_enabled"],
        "fee_rate": report["fee_rate"],
        "intended_order_size_usd": report["intended_order_size_usd"],
        "duration_hours": report["duration_hours"],
        "interval_seconds": report["interval_seconds"],
        "expected_samples": report["expected_samples"],
        "minimum_successful_samples_for_full_verdict": report["minimum_successful_samples_for_full_verdict"],
        "current_run_samples": report["current_run_samples"],
        "successful_samples": report["successful_samples"],
        "failed_samples": report["failed_samples"],
        "total_attempted_samples": report["total_attempted_samples"],
        "cumulative_successful_samples": report["cumulative_successful_samples"],
        "cumulative_failed_samples": report["cumulative_failed_samples"],
        "cumulative_attempted_samples": report["cumulative_attempted_samples"],
        "duplicate_samples_removed": report["duplicate_samples_removed"],
        "master_file_path": report["master_file_path"],
        "current_run_file_path": report["current_run_file_path"],
        "collection_complete": report["collection_complete"],
        "full_24h_complete": report["full_24h_complete"],
        "audit_status": report["audit_status"],
        "error_rate": report["error_rate"],
        "average_spread_pct": report["average_spread_pct"],
        "median_spread_pct": report["median_spread_pct"],
        "p95_spread_pct": report["p95_spread_pct"],
        "average_estimated_all_in_cost_per_side": report["average_estimated_all_in_cost_per_side"],
        "median_estimated_all_in_cost_per_side": report["median_estimated_all_in_cost_per_side"],
        "p95_estimated_all_in_cost_per_side": report["p95_estimated_all_in_cost_per_side"],
        "depth_support": report["depth_support"],
        "criteria_met": report["criteria_met"],
        "verdict": report["verdict"],
        "verdict_basis": report["verdict_basis"],
        "artifacts": payload["artifacts"],
    }


def _run_merge_coinbase_execution_audit_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from data.coinbase_execution_cost_audit import (
        CoinbaseExecutionAuditMergeConfig,
        merge_coinbase_execution_audit_samples,
    )

    args = build_merge_coinbase_execution_audit_parser().parse_args(argv)
    config = CoinbaseExecutionAuditMergeConfig(
        input_csv_paths=tuple(args.input_csv),
        output_dir=args.output_dir,
        intended_order_size_usd=args.intended_order_size,
        failed_samples=args.failed_samples,
    )
    payload = merge_coinbase_execution_audit_samples(config)
    report = payload["coinbase_execution_cost_audit_final"]
    return {
        "phase": report["phase"],
        "mode": report["mode"],
        "strategy_trading_enabled": report["strategy_trading_enabled"],
        "automated_live_trading_enabled": report["automated_live_trading_enabled"],
        "total_successful_samples": report["total_successful_samples"],
        "total_failed_samples": report["total_failed_samples"],
        "sample_requirement_met": report["sample_requirement_met"],
        "error_rate": report["error_rate"],
        "average_all_in_cost_per_side": report["average_all_in_cost_per_side"],
        "median_all_in_cost_per_side": report["median_all_in_cost_per_side"],
        "p95_all_in_cost_per_side": report["p95_all_in_cost_per_side"],
        "worst_all_in_cost_per_side": report["worst_all_in_cost_per_side"],
        "average_spread_pct": report["average_spread_pct"],
        "median_spread_pct": report["median_spread_pct"],
        "p95_spread_pct": report["p95_spread_pct"],
        "depth_support_ratio": report["depth_support_ratio"],
        "duplicate_timestamps_removed": report["duplicate_timestamps_removed"],
        "verdict": report["verdict"],
        "verdict_basis": report["verdict_basis"],
        "shadow_trading_allowed": report["shadow_trading_allowed"],
        "live_trading_allowed": report["live_trading_allowed"],
        "artifacts": payload["artifacts"],
    }


def _run_shadow_coinbase_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH
    from risk.risk_config import DEFAULT_RISK_CONFIG_PATH
    from shadow_trading.coinbase_shadow import ShadowTradingConfig, run_coinbase_shadow_trading

    args = build_shadow_coinbase_parser().parse_args(argv)
    config = ShadowTradingConfig(
        product_id=args.product_id,
        initial_shadow_capital=args.initial_shadow_capital,
        intended_order_size_usd=args.intended_order_size,
        fee_rate=args.fee_rate,
        max_all_in_cost_per_side=args.max_all_in_cost_per_side,
        duration_days=args.duration_days,
        cycle_interval_seconds=args.cycle_interval_seconds,
        cycle_limit=args.cycle_limit,
        target_signal_count=args.target_signal_count,
        resume_signal_collection=not args.no_resume_signal_collection,
        history_limit=args.history_limit,
        output_dir=args.output_dir,
        base_url=args.coinbase_base_url,
        timeout_seconds=args.timeout_seconds,
        order_book_limit=args.order_book_limit,
        risk_engine_enabled=args.enable_risk_engine,
        risk_config_path=args.risk_config_path or DEFAULT_RISK_CONFIG_PATH,
        portfolio_state_path=args.portfolio_state_path or DEFAULT_PORTFOLIO_STATE_PATH,
    )
    progress_callback = None if args.quiet else _print_shadow_progress
    payload = run_coinbase_shadow_trading(config, progress_callback=progress_callback)
    summary = payload["shadow_summary_30d"]
    return {
        "phase": summary["phase"],
        "mode": summary["mode"],
        "product_id": summary["product_id"],
        "live_trading_enabled": summary["live_trading_enabled"],
        "order_endpoint_calls_allowed": summary["order_endpoint_calls_allowed"],
        "total_return_pct": summary["total_return_pct"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "profit_factor": summary["profit_factor"],
        "win_rate": summary["win_rate"],
        "total_trades": summary["total_trades"],
        "average_net_pnl_per_trade": summary["average_net_pnl_per_trade"],
        "average_holding_time": summary["average_holding_time"],
        "average_all_in_cost_per_side": summary["average_all_in_cost_per_side"],
        "signal_count": summary["signal_count"],
        "rejected_signal_count": summary["rejected_signal_count"],
        "data_quality_score": summary["data_quality_score"],
        "system_uptime_pct": summary["system_uptime_pct"],
        "api_error_rate": summary["api_error_rate"],
        "final_verdict": summary["final_verdict"],
        "risk_engine": summary["risk_engine"],
        "artifacts": payload["artifacts"],
    }


def _run_false_avoid_analysis_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from research.false_avoid_analysis import FalseAvoidAnalysisConfig, run_false_avoid_analysis

    args = build_false_avoid_analysis_parser().parse_args(argv)
    payload = run_false_avoid_analysis(FalseAvoidAnalysisConfig(output_dir=args.output_dir))
    report = payload["false_avoid_analysis"]
    candidate = payload["watch_long_candidate_backtest"]
    return {
        "phase": report["phase"],
        "mode": report["mode"],
        "live_trading_enabled": report["live_trading_enabled"],
        "order_endpoint_calls_allowed": report["order_endpoint_calls_allowed"],
        "metrics": report["metrics"],
        "false_avoid_rejection_reason_counts": report["false_avoid_rejection_reason_counts"],
        "watch_long_candidate_status": candidate["status"],
        "recommendation": report["recommendation"],
        "artifacts": payload["artifacts"],
    }


def _run_collect_shadow_signals_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from shadow_trading.coinbase_shadow import (
        ShadowSignalCollectionConfig,
        collect_enriched_shadow_signals,
    )

    args = build_collect_shadow_signals_parser().parse_args(argv)
    config = ShadowSignalCollectionConfig(
        product_id=args.product_id,
        target_signals=args.target_signals,
        interval_seconds=args.interval_seconds,
        history_limit=args.history_limit,
        output_dir=args.output_dir,
        base_url=args.coinbase_base_url,
        timeout_seconds=args.timeout_seconds,
        order_book_limit=args.order_book_limit,
        intended_order_size_usd=args.intended_order_size,
        fee_rate=args.fee_rate,
        max_all_in_cost_per_side=args.max_all_in_cost_per_side,
        reset=args.reset,
    )
    progress_callback = None if args.quiet else _print_collect_shadow_signal_progress
    payload = collect_enriched_shadow_signals(config, progress_callback=progress_callback)
    report = payload["enriched_false_avoid_analysis"]
    quality = payload["signal_journal_quality_report"]
    health = payload["shadow_system_health"]
    return {
        "phase": report["phase"],
        "mode": report["mode"],
        "product_id": args.product_id,
        "live_trading_enabled": report["live_trading_enabled"],
        "order_endpoint_calls_allowed": report["order_endpoint_calls_allowed"],
        "total_signals": report["total_signals"],
        "total_avoid_long_signals": report["total_avoid_long_signals"],
        "false_avoid_count": report["false_avoid_count"],
        "correct_avoid_count": report["correct_avoid_count"],
        "inconclusive_count": report["inconclusive_count"],
        "false_avoid_rate": report["false_avoid_rate"],
        "most_common_false_avoid_rejection_reasons": report["most_common_false_avoid_rejection_reasons"],
        "average_missed_gain_1d": report["average_missed_gain_1d"],
        "average_missed_gain_3d": report["average_missed_gain_3d"],
        "max_missed_gain": report["max_missed_gain"],
        "target_reached": report["target_reached"],
        "whether_watch_long_should_be_tested_again": report["whether_watch_long_should_be_tested_again"],
        "signal_journal_quality_score": quality["signal_journal_quality_score"],
        "incomplete_signal_rows": quality["incomplete_signal_rows"],
        "avoid_long_rows_missing_rejection_reasons": quality["avoid_long_rows_missing_rejection_reasons"],
        "api_error_rate": health["api_error_rate"],
        "unauthorized_order_endpoint_calls": health["unauthorized_order_endpoint_calls"],
        "artifacts": payload["artifacts"],
    }


def _run_investor_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_investor_parser().parse_args(argv)
    symbol = args.symbol.upper()

    from agents.equity_investor_agent import is_core_etf_symbol

    is_equity = args.asset_class == "EQUITY" or (args.asset_class == "AUTO" and is_core_etf_symbol(symbol))

    if is_equity:
        payload = _run_equity_investor(args, symbol)
        recommendation_dict = payload["position_recommendation"]
        from decision.recommendation import PositionRecommendation

        recommendation = PositionRecommendation.from_dict(recommendation_dict)
    elif symbol in {"BTC", "BTCUSDT"}:
        payload = _run_btc_investor(args)
        from decision.investor_recommendation import build_crypto_position_recommendation

        recommendation = build_crypto_position_recommendation(payload, args.default_position_usd)
    else:
        payload = _run_crypto_investor(args)
        from decision.investor_recommendation import build_crypto_position_recommendation

        recommendation = build_crypto_position_recommendation(payload, args.default_position_usd)

    if args.skip_risk_engine:
        return payload

    return _attach_risk_decision(payload, recommendation, args)


def _run_btc_investor(args: argparse.Namespace) -> dict[str, Any]:
    from agents.investor_agent import InvestorAgentConfig, run_investor_agent

    return run_investor_agent(
        InvestorAgentConfig(
            symbol="BTC",
            output_dir=args.output_dir,
            offline=args.offline,
            request_timeout_seconds=args.timeout_seconds,
            binance_base_url=args.binance_base_url,
            portfolio_risk_profile=args.portfolio_risk_profile,
            current_btc_allocation_pct=args.current_btc_allocation_pct,
            target_btc_allocation_pct=args.target_btc_allocation_pct,
            max_btc_allocation_pct=args.max_btc_allocation_pct,
            current_btc=args.current_btc,
            target_btc=args.target_btc,
            target_sell_price=args.target_sell_price,
            planned_sell_btc=args.planned_sell_btc,
            retain_btc=args.retain_btc,
            monthly_dca_usd=args.monthly_dca_usd,
            lump_sum_available_usd=args.lump_sum_available_usd,
            reference_price_for_dip=args.reference_price_for_dip,
        )
    )


def _run_crypto_investor(args: argparse.Namespace) -> dict[str, Any]:
    from agents.crypto_investor_agent import CryptoInvestorConfig, run_crypto_investor_agent

    return run_crypto_investor_agent(
        CryptoInvestorConfig(
            symbol=args.symbol,
            output_dir=args.output_dir,
            offline=args.offline,
            request_timeout_seconds=args.timeout_seconds,
            binance_base_url=args.binance_base_url,
            thesis_risk_level=args.thesis_risk_level,
            thesis_risk_flags=tuple(args.thesis_risk_flags),
            asset_name=args.asset_name,
            sector=args.sector,
            market_data_source=args.market_data_source,
            current_price=args.current_price,
            ma200=args.ma200,
            weekly_rsi=args.weekly_rsi,
            monthly_ema20=args.monthly_ema20,
            monthly_trend=args.monthly_trend,
            recent_cycle_high=args.recent_cycle_high,
            quote_volume_usd=args.quote_volume_usd,
            average_quote_volume_usd=args.average_quote_volume_usd,
            atr_pct=args.atr_pct,
        )
    )


def _run_equity_investor(args: argparse.Namespace, symbol: str) -> dict[str, Any]:
    from agents.equity_investor_agent import EquityInvestorConfig, run_equity_investor_agent

    return run_equity_investor_agent(
        EquityInvestorConfig(
            symbol=symbol,
            bucket=args.bucket,
            output_dir=args.output_dir,
            offline=args.offline,
            request_timeout_seconds=args.timeout_seconds,
            default_position_usd=args.default_position_usd,
            current_price=args.current_price,
            ma200=args.ma200,
            weekly_rsi=args.weekly_rsi,
            monthly_ema20=args.monthly_ema20,
            monthly_trend=args.monthly_trend,
        )
    )


def _attach_risk_decision(
    payload: dict[str, Any],
    recommendation: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from risk.live_risk_engine import LiveRiskEngine
    from risk.portfolio_state import DEFAULT_PORTFOLIO_STATE_PATH, load_portfolio_state
    from risk.risk_config import DEFAULT_RISK_CONFIG_PATH, load_risk_config
    from risk.risk_decision_log import append_risk_decision

    risk_config_path = args.risk_config_path or DEFAULT_RISK_CONFIG_PATH
    portfolio_state_path = args.portfolio_state_path or DEFAULT_PORTFOLIO_STATE_PATH
    config = load_risk_config(risk_config_path)
    state = load_portfolio_state(portfolio_state_path, config)

    engine = LiveRiskEngine(config)
    decision = engine.evaluate(recommendation, state)
    append_risk_decision(decision)

    payload["position_recommendation"] = recommendation.to_dict()
    payload["risk_decision"] = decision.to_dict()
    return payload


def _run_backtest_command(argv: Sequence[str] | None = None) -> dict[str, Any]:
    from backtesting.backtest_engine import (
        BacktestConfig,
        load_or_download_timeframes,
        run_backtest,
    )
    from backtesting.benchmarks.research import (
        run_hybrid_runner_optimization,
        run_hybrid_trend_rider_analysis,
        run_profit_capture_analysis,
        run_benchmark_suite,
        run_phase16_research,
        run_portfolio_risk_governor_analysis,
        run_regime_gated_trend_holding_analysis,
        run_trend_holding_analysis,
        run_trend_rider_analysis,
        run_trend_participation_research,
    )
    from backtesting.broker_cost_validation import run_broker_cost_validation
    from research.exit_optimization_engine import run_exit_optimization
    from backtesting.cross_asset_validation import DEFAULT_CROSS_ASSETS, run_cross_asset_validation
    from backtesting.backtest_report import (
        write_backtest_report,
        write_benchmark_comparison,
        write_profile_comparison,
    )
    from backtesting.profiles import PROFILE_NAMES

    args = build_backtest_parser().parse_args(argv)
    config = BacktestConfig(
        symbol=args.symbol.upper(),
        start=args.start,
        end=args.end,
        primary_timeframe=args.primary_timeframe,
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        progress_interval=args.progress_interval,
        profile="balanced" if args.profile == "all" else args.profile,
        refresh_cache=args.refresh_cache,
        stop_type=args.stop_type,
    )
    progress_callback = None if args.quiet else _print_backtest_progress
    if args.strategy == "aggressive":
        from backtesting.benchmarks.research import run_market_structure_stop_analysis

        payload = run_market_structure_stop_analysis(
            config,
            focus_stop_type=args.stop_type,
            progress_callback=progress_callback,
        )
        report = payload["market_structure_stop_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.12",
            "focus_stop_type": args.stop_type,
            "targets": report["targets"],
            "baseline_stop_out_count": report["baseline_stop_out_count"],
            "stop_out_survival_analysis": report["stop_out_survival_analysis"],
            "closest_strategy": report["closest_strategy"],
            "best_stop_type": report["best_stop_type"],
            "best_profit_capture": report["best_profit_capture"],
            "best_risk_adjusted_return": report["best_risk_adjusted_return"],
            "recommended_production_configuration": report["recommended_production_configuration"],
            "recommendation": report["recommendation"],
            "rankings": report["rankings"],
            "strategies_meeting_targets": report["strategies_meeting_targets"],
            "selected_strategy": report["strategies"].get(f"aggressive_{args.stop_type}_stop")
            if args.stop_type != "fixed"
            else report["strategies"].get("aggressive_current"),
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "total_trades": metrics["total_trades"],
                    "stop_out_count": metrics["stop_out_count"],
                    "stop_type_usage": metrics["stop_type_usage"],
                    "average_stop_distance_pct": metrics["average_stop_distance_pct"],
                    "average_stop_distance_atr": metrics["average_stop_distance_atr"],
                    "survived_stopouts_count": metrics["survived_stopouts_count"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "broker_cost_validation":
        cost_config = replace(
            config,
            symbol=args.symbol.upper(),
            profile="aggressive",
            stop_type="fixed",
        )
        payload = run_broker_cost_validation(cost_config, progress_callback=progress_callback)
        report = payload["broker_cost_validation_report"]
        return {
            "symbol": report["symbol"],
            "phase": report["phase"],
            "strategy": report["strategy"],
            "success_criteria": report["success_criteria"],
            "rankings": report["rankings"],
            "recommendation": report["recommendation"],
            "profiles": {
                profile: {
                    "fee_rate": metrics["fee_rate"],
                    "slippage_rate": metrics["slippage_rate"],
                    "all_in_cost_per_side_pct": metrics["all_in_cost_per_side_pct"],
                    "final_equity": metrics["final_equity"],
                    "total_return_pct": metrics["total_return_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "total_trades": metrics["total_trades"],
                    "cost_drag_pct": metrics["cost_drag_pct"],
                    "live_ready": metrics["live_readiness"]["live_ready"],
                }
                for profile, metrics in report["profiles"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "exit_optimization":
        exit_config = replace(
            config,
            symbol=args.symbol.upper(),
            start="2020-01-01" if args.start == "2017-01-01" else args.start,
            profile="aggressive",
            stop_type="fixed",
        )
        payload = run_exit_optimization(exit_config, progress_callback=progress_callback)
        report = payload["exit_optimization_report"]
        return {
            "symbol": exit_config.symbol,
            "phase": "1.17",
            "targets": report["targets"],
            "ranking_table": report["ranking_table"],
            "best_profit_capture_model": report["best_profit_capture_model"],
            "best_sharpe_model": report["best_sharpe_model"],
            "best_hybrid_model": report["best_hybrid_model"],
            "recommended_production_exit": report["recommended_production_exit"],
            "comparison_vs_current_production": report["comparison_vs_current_production"],
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "cross_asset_validation":
        assets = tuple(asset.upper() for asset in args.assets) if args.assets else DEFAULT_CROSS_ASSETS
        payload = run_cross_asset_validation(
            config,
            assets=assets,
            include_optional=args.include_optional_assets,
            progress_callback=progress_callback,
        )
        report = payload["cross_asset_validation"]
        return {
            "symbol": "MULTI_ASSET",
            "phase": "1.16",
            "strategy": report["strategy"],
            "common_start": report["common_start"],
            "common_end": report["common_end"],
            "failed_assets": report["failed_assets"],
            "asset_class_analysis": report["asset_class_analysis"],
            "success_criteria": report["success_criteria"],
            "rankings": report["rankings"],
            "recommended_production_assets": report["recommended_production_assets"],
            "trend_following_assessment": report["trend_following_assessment"],
            "assets": {
                asset: {
                    "status": row["status"],
                    "total_return_pct": row.get("total_return_pct"),
                    "cagr": row.get("cagr"),
                    "sharpe_ratio": row.get("sharpe_ratio"),
                    "max_drawdown_pct": row.get("max_drawdown_pct"),
                    "profit_factor": row.get("profit_factor"),
                    "win_rate": row.get("win_rate"),
                    "total_trades": row.get("total_trades"),
                    "profit_capture_ratio": row.get("profit_capture_ratio"),
                    "robustness_score": row.get("robustness_score"),
                    "failure_analysis": row.get("failure_analysis"),
                    "error": row.get("error"),
                }
                for asset, row in report["assets"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "trend_holding":
        payload = run_trend_holding_analysis(config, progress_callback=progress_callback)
        report = payload["trend_holding_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.13",
            "targets": report["targets"],
            "comparison": report["comparison"],
            "recommended_configuration": report["recommended_configuration"],
            "missed_opportunity_recheck": {
                "sample_size": report["missed_opportunity_recheck"]["sample_size"],
                "runner_survived_count": report["missed_opportunity_recheck"]["runner_survived_count"],
                "runner_captured_count": report["missed_opportunity_recheck"]["runner_captured_count"],
                "additional_profit_captured_pct": report["missed_opportunity_recheck"]["additional_profit_captured_pct"],
            },
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "max_runner_return_pct": metrics["max_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "median_runner_holding_hours": metrics["median_runner_holding_hours"],
                    "tp1_hits": metrics["tp1_hits"],
                    "tp2_hits": metrics["tp2_hits"],
                    "runner_activations": metrics["runner_activations"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "regime_gated_trend_holding":
        payload = run_regime_gated_trend_holding_analysis(config, progress_callback=progress_callback)
        report = payload["regime_gated_trend_holding_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.14",
            "targets": report["targets"],
            "comparison": report["comparison"],
            "recommendation": report["recommendation"],
            "missed_opportunity_recheck": {
                "sample_size": report["missed_opportunity_recheck"]["sample_size"],
                "survived_count": report["missed_opportunity_recheck"]["survived_count"],
                "additional_profit_captured_pct": report["missed_opportunity_recheck"]["additional_profit_captured_pct"],
            },
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "runner_activation_count": metrics["runner_activation_count"],
                    "runner_disabled_count": metrics["runner_disabled_count"],
                    "strong_bull_periods": metrics["strong_bull_periods"],
                    "bull_periods": metrics["bull_periods"],
                    "range_periods": metrics["range_periods"],
                    "bear_periods": metrics["bear_periods"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "regime_gated_portfolio_governor":
        payload = run_portfolio_risk_governor_analysis(config, progress_callback=progress_callback)
        report = payload["portfolio_risk_governor_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.15",
            "targets": report["targets"],
            "rankings": report["rankings"],
            "comparison": report["comparison"],
            "recommendation": report["recommendation"],
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "risk_state_counts": metrics["risk_state_counts"],
                    "average_position_size": metrics["average_position_size"],
                    "average_runner_size": metrics["average_runner_size"],
                    "portfolio_stop_count": metrics["portfolio_stop_count"],
                    "defensive_mode_hours": metrics["defensive_mode_hours"],
                    "runner_activation_count": metrics["runner_activation_count"],
                    "runner_disabled_count": metrics["runner_disabled_count"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy in {"hybrid_optimization", "hybrid_conservative", "hybrid_balanced", "hybrid_aggressive"}:
        focus_strategy = None if args.strategy == "hybrid_optimization" else args.strategy
        payload = run_hybrid_runner_optimization(
            config,
            focus_strategy=focus_strategy,
            progress_callback=progress_callback,
        )
        report = payload["hybrid_runner_optimization"]
        return {
            "symbol": config.symbol,
            "phase": "1.11",
            "focus_strategy": focus_strategy,
            "targets": report["targets"],
            "closest_profile": report["closest_profile"],
            "recommendation": report["recommendation"],
            "rankings": report["rankings"],
            "selected_strategy": report["strategies"].get(args.strategy) if focus_strategy else None,
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "median_runner_holding_hours": metrics["median_runner_holding_hours"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "max_runner_return_pct": metrics["max_runner_return_pct"],
                    "average_runner_drawdown_pct": metrics["average_runner_drawdown_pct"],
                    "max_runner_drawdown_pct": metrics["max_runner_drawdown_pct"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "target_assessment": metrics["target_assessment"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.strategy == "hybrid_trend_rider":
        payload = run_hybrid_trend_rider_analysis(config, progress_callback=progress_callback)
        report = payload["hybrid_trend_rider_report"]
        return {
            "symbol": config.symbol,
            "phase": "1.10",
            "targets": report["targets"],
            "target_assessment": report["target_assessment"],
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr": metrics["cagr"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                    "tp1_hit_count": metrics["tp1_hit_count"],
                    "tp2_hit_count": metrics["tp2_hit_count"],
                    "runner_activation_count": metrics["runner_activation_count"],
                    "runner_exit_reasons": metrics["runner_exit_reasons"],
                    "average_runner_drawdown_pct": metrics["average_runner_drawdown_pct"],
                    "max_runner_drawdown_pct": metrics["max_runner_drawdown_pct"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.trend_rider:
        payload = run_trend_rider_analysis(config, progress_callback=progress_callback)
        report = payload["trend_rider_analysis"]
        return {
            "symbol": config.symbol,
            "phase": "1.9",
            "target_profit_capture_ratio": report["target_profit_capture_ratio"],
            "strategies": {
                strategy: {
                    "total_return_pct": metrics["total_return_pct"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "total_trades": metrics["total_trades"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                    "profit_capture_target_met": metrics["profit_capture_target_met"],
                    "average_runner_return_pct": metrics["average_runner_return_pct"],
                    "average_runner_holding_hours": metrics["average_runner_holding_hours"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.profit_capture:
        payload = run_profit_capture_analysis(config, progress_callback=progress_callback)
        report = payload["profit_capture_analysis"]
        return {
            "symbol": config.symbol,
            "phase": "1.8",
            "strategies": {
                strategy: {
                    "total_trades": metrics["total_trades"],
                    "average_holding_hours": metrics["average_holding_hours"],
                    "median_holding_hours": metrics["median_holding_hours"],
                    "profit_capture_ratio": metrics["profit_capture_ratio"],
                }
                for strategy, metrics in report["strategies"].items()
            },
            "artifacts": payload["artifacts"],
        }

    if args.trend_participation:
        trend_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, trend_timeframes)
        payload = run_trend_participation_research(config, cached_data=cached_data, progress_callback=progress_callback)
        report = payload["trend_participation"]
        return {
            "symbol": config.symbol,
            "phase": "1.7",
            "best_strategy_by_risk_adjusted_return": report["best_strategy_by_risk_adjusted_return"],
            "best_strategy_under_drawdown_limit": report["best_strategy_under_drawdown_limit"],
            "strategies": report["strategies"],
            "artifacts": payload["artifacts"],
        }

    if args.research:
        research_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, research_timeframes)
        payload = run_phase16_research(config, cached_data=cached_data, progress_callback=progress_callback)
        return {
            "symbol": config.symbol,
            "phase": "1.6",
            "best_benchmark": payload["strategy_research_report"]["best_benchmark"],
            "best_regime": payload["strategy_research_report"]["best_regime"],
            "worst_regime": payload["strategy_research_report"]["worst_regime"],
            "most_expensive_filter": payload["strategy_research_report"]["most_expensive_filter"],
            "artifacts": payload["artifacts"],
        }

    if args.benchmarks:
        benchmark_timeframes = _ordered_backtest_timeframes(
            config.primary_timeframe,
            tuple([*args.timeframes, "1d"]),
        )
        cached_data = load_or_download_timeframes(config, benchmark_timeframes)
        results = run_benchmark_suite(config, cached_data=cached_data, progress_callback=progress_callback)
        comparison_path = write_benchmark_comparison(results, args.output_dir)
        return {
            "symbol": config.symbol,
            "benchmark_comparison": {
                strategy: _benchmark_summary(result)
                for strategy, result in results.items()
            },
            "best_strategy_by_risk_adjusted_return": _best_strategy_from_results(results),
            "artifacts": {
                "benchmark_comparison": str(comparison_path),
            },
        }

    if args.profile == "all":
        comparison_timeframes = _ordered_backtest_timeframes(config.primary_timeframe, tuple(args.timeframes))
        cached_data = load_or_download_timeframes(config, comparison_timeframes)
        results = {}
        artifacts = {}
        for profile_name in PROFILE_NAMES:
            if progress_callback is not None:
                print(f"[backtest] profile {profile_name}", file=sys.stderr, flush=True)
            profile_config = replace(config, profile=profile_name)
            result = run_backtest(profile_config, cached_data=cached_data, progress_callback=progress_callback)
            results[profile_name] = result
            paths = write_backtest_report(
                result,
                args.output_dir / "profiles" / profile_name,
                write_chart=False,
            )
            artifacts[profile_name] = {name: str(path) for name, path in paths.items()}
        comparison_path = write_profile_comparison(results, args.output_dir)
        return {
            "symbol": config.symbol,
            "profile": "all",
            "best_profile_by_risk_adjusted_return": _best_profile_from_results(results),
            "profiles": {
                profile: _profile_summary(result)
                for profile, result in results.items()
            },
            "artifacts": {
                "profile_comparison": str(comparison_path),
                "profiles": artifacts,
            },
        }

    result = run_backtest(config, progress_callback=progress_callback)
    paths = write_backtest_report(result, args.output_dir)
    return {
        "symbol": result.symbol,
        "profile": result.profile,
        "strategy_profile": result.strategy_profile,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        **result.metrics,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _ordered_backtest_timeframes(primary_timeframe: str, timeframes: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [primary_timeframe]
    for timeframe in timeframes:
        if timeframe not in ordered:
            ordered.append(timeframe)
    return tuple(ordered)


def _profile_summary(result: Any) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr",
        "max_drawdown_pct",
        "profit_factor",
        "sharpe_ratio",
        "win_rate",
        "expectancy",
        "total_trades",
    )
    return {key: result.metrics.get(key) for key in keys}


def _benchmark_summary(result: Any) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr",
        "max_drawdown_pct",
        "sharpe_ratio",
        "total_trades",
    )
    return {key: result.metrics.get(key) for key in keys}


def _best_profile_from_results(results: dict[str, Any]) -> str | None:
    if not results:
        return None
    return max(
        results,
        key=lambda profile: (
            float(results[profile].metrics.get("sharpe_ratio", float("-inf"))),
            -float(results[profile].metrics.get("max_drawdown_pct", float("inf"))),
            float(results[profile].metrics.get("cagr", float("-inf"))),
        ),
    )


def _best_strategy_from_results(results: dict[str, Any]) -> str | None:
    if not results:
        return None
    return max(
        results,
        key=lambda strategy: (
            float(results[strategy].metrics.get("sharpe_ratio", float("-inf"))),
            -float(results[strategy].metrics.get("max_drawdown_pct", float("inf"))),
            float(results[strategy].metrics.get("cagr", float("-inf"))),
        ),
    )


def _print_backtest_progress(event: dict[str, Any]) -> None:
    if event.get("phase") == "benchmark":
        print(
            f"[backtest] benchmark {event['strategy']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "broker_cost_profile":
        print(
            f"[backtest] broker cost profile {event['profile']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "filter_attribution":
        print(
            f"[backtest] filter attribution {event['experiment']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "exit_model":
        print(
            f"[backtest] exit model {event['model']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "cross_asset_load":
        print(
            f"[backtest] cross-asset load {event['symbol']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "cross_asset_backtest":
        print(
            f"[backtest] cross-asset backtest {event['symbol']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "equity_validation_load":
        print(
            f"[backtest] equity data validation load {event['symbol']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "prepared":
        print(
            "[backtest] prepared "
            f"{event['symbol']} with {event['primary_rows']} primary candles "
            f"across {', '.join(event['timeframes'])}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "replay":
        timestamp = event.get("timestamp") or "warming up"
        latest_decision = event.get("latest_decision") or "none"
        print(
            "[backtest] replay "
            f"{event['processed_rows']}/{event['total_rows']} "
            f"({event['pct_complete']}%) "
            f"decisions={event['decisions']} trades={event['trades']} "
            f"latest={latest_decision} timestamp={timestamp}",
            file=sys.stderr,
            flush=True,
        )


def _print_coinbase_execution_audit_progress(event: dict[str, Any]) -> None:
    if event.get("phase") == "coinbase_execution_sample":
        print(
            "[coinbase-audit] sample "
            f"{event['sample']}/{event['expected_samples']} "
            f"{event['product_id']} bid={event['best_bid']} ask={event['best_ask']} "
            f"spread={event['spread_pct']}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event.get("phase") == "coinbase_execution_error":
        print(
            "[coinbase-audit] error "
            f"sample={event['sample']} type={event['error_type']} message={event['error']}",
            file=sys.stderr,
            flush=True,
        )


def _print_shadow_progress(event: dict[str, Any]) -> None:
    if event.get("phase") == "coinbase_shadow_cycle":
        print(
            "[shadow] cycle "
            f"{event['cycle']} action={event['action']} decision={event['decision']} "
            f"price={event['price']} trades={event['cumulative_trades']}",
            file=sys.stderr,
            flush=True,
        )


def _print_collect_shadow_signal_progress(event: dict[str, Any]) -> None:
    if event.get("phase") == "enriched_shadow_signal_collection":
        print(
            "[shadow-collect] cycle "
            f"{event['cycle']} signals={event['total_signals']}/{event['target_signals']} "
            f"decision={event['final_decision']} price={event['price']}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "backtest":
        payload = _run_backtest_command(args[1:])
    elif args and args[0] == "audit-fees":
        payload = _run_audit_fees_command(args[1:])
    elif args and args[0] in {"audit-coinbase-execution", "coinbase-execution-audit"}:
        payload = _run_coinbase_execution_audit_command(args[1:])
    elif args and args[0] in {"merge-coinbase-execution-audit", "merge-coinbase-execution-samples"}:
        payload = _run_merge_coinbase_execution_audit_command(args[1:])
    elif args and args[0] in {"shadow-coinbase", "coinbase-shadow-trading"}:
        payload = _run_shadow_coinbase_command(args[1:])
    elif args and args[0] in {"collect-shadow-signals", "collect-enriched-shadow-signals"}:
        payload = _run_collect_shadow_signals_command(args[1:])
    elif args and args[0] in {"screen-airdrop", "airdrop-screen"}:
        return _run_screen_airdrop_command(args[1:])
    elif args and args[0] == "investor":
        payload = _run_investor_command(args[1:])
    elif args and args[0] in {"analyze-false-avoids", "false-avoid-analysis"}:
        payload = _run_false_avoid_analysis_command(args[1:])
    elif args and args[0] == "validate-equities":
        payload = _run_validate_equities_command(args[1:])
    elif args and args[0] == "signal":
        payload = _run_signal_command(args[1:])
    else:
        payload = _run_signal_command(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
