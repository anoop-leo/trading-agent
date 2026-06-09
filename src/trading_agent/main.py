"""Command-line orchestration for the Phase 1 signal engine."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from decision.decision_engine import DecisionInput, make_decision
from scoring.market_regime_skill import MarketRegimeResult, calculate_market_regime
from scoring.risk_reward_skill import RiskRewardResult, calculate_risk_reward
from scoring.setup_detection_skill import SetupInput, SetupResult, detect_setup
from scoring.support_resistance_skill import SupportResistanceResult, calculate_support_resistance
from trading_agent.config import AgentConfig
from trading_agent.data import BinanceKlineProvider
from trading_agent.indicators import add_indicators
from trading_agent.models import MarketDataProvider, SignalScores
from trading_agent.output import build_output_payload, macd_direction, write_chart, write_json
from trading_agent.scoring import calculate_recent_swing_high, calculate_recent_swing_low, calculate_scores, calculate_volume_ratio


ChartWriter = Callable[[pd.DataFrame, Path, str, str], Path]


def calculate_trade_quality(
    indicator_frame: pd.DataFrame,
) -> tuple[SupportResistanceResult, RiskRewardResult, MarketRegimeResult]:
    """Calculate Phase 1.1 trade-quality skills from the latest indicator frame."""

    latest = indicator_frame.iloc[-1]
    current_price = float(latest["close"])
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


def run(
    config: AgentConfig,
    provider: MarketDataProvider | None = None,
    chart_writer: ChartWriter = write_chart,
) -> dict[str, Any]:
    """Run one local Phase 1 signal-generation cycle."""

    market_data_provider = provider or BinanceKlineProvider(
        base_url=config.binance_base_url,
        timeout_seconds=config.request_timeout_seconds,
    )

    ohlcv = market_data_provider.fetch_ohlcv(
        symbol=config.symbol,
        interval=config.interval,
        limit=config.history_limit,
    )
    indicators = add_indicators(ohlcv)
    support_resistance, risk_reward, market_regime = calculate_trade_quality(indicators)
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
        config.symbol,
        config.position_mode,
        indicators,
        scores,
        support_resistance,
        risk_reward,
        market_regime,
        setup,
    )
    decision = make_decision(decision_input)
    payload = build_output_payload(config, indicators, scores, decision, support_resistance, risk_reward, market_regime, setup)

    write_json(payload, config.output_dir)
    chart_writer(indicators, config.output_dir, config.symbol, setup.setup.value)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 technical signal engine.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Supported symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Binance kline interval.")
    parser.add_argument("--history-limit", type=int, default=500, help="Number of candles to fetch.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Artifact output directory.")
    parser.add_argument(
        "--position-mode",
        default="NO_POSITION",
        choices=["NO_POSITION", "HOLDING"],
        help="Decision terminology mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig(
        symbol=args.symbol,
        interval=args.interval,
        history_limit=args.history_limit,
        output_dir=args.output_dir,
        position_mode=args.position_mode,
    )
    payload = run(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
