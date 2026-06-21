# AI Trading Agent

## Overview

AI Trading Agent is a local-first trading decision platform designed to run on a MacBook.

The system combines technical analysis, news analysis, market regime detection, and risk management to generate high-quality trading alerts.

The goal is not to create a fully autonomous trading bot.

The goal is to build an AI-assisted decision system that identifies high-probability trading opportunities and provides explainable recommendations.

---

# Project Goals

## Phase 1

Build a technical signal engine.

Features:

- Binance market data
- Bybit market data for HYPEUSDT
- RSI
- EMA20
- EMA50
- EMA200
- MACD
- Bollinger Bands
- Volume Spike Detection
- Bottom detection score
- Support/resistance score
- Risk/reward score
- BTC market regime score
- Setup detection
- Multi-timeframe analysis
- Signal journal with +1/+3/+7/+30 day evaluation
- Historical backtesting engine
- Decision Engine v2

Outputs:

No-position mode:

- BUY
- BUY WATCH
- WAIT
- AVOID LONG
- WATCH FOR REVERSAL

Holding-position mode:

- ADD
- HOLD
- REDUCE
- EXIT

Artifacts:

- output.json
- chart.png
- signal_journal.json
- backtest_report.json
- backtest_trades.csv
- equity_curve.csv
- terminal summary

No broker integration.

No live trading.

No AI reasoning.

---

# Local Usage

Install:

```bash
venv_trading/bin/python -m pip install -r requirements.txt
```

Run:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main --symbol BTCUSDT
```

For Hyperliquid HYPE, the agent uses Bybit public market data automatically:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main --symbol HYPEUSDT
```

Decision terminology defaults to no-position mode. Use holding mode when evaluating an existing position:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main --symbol BTCUSDT --position-mode HOLDING
```

Multi-timeframe analysis defaults to 1h, 4h, and 1d. Override it with:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main --symbol BTCUSDT --timeframes 1h 4h 1d
```

Test:

```bash
PYTHONPATH=src venv_trading/bin/python -m unittest discover -s tests
```

Backtest:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main backtest --symbol BTCUSDT --start 2017-01-01 --end latest
```

Backtest a strategy profile:

```bash
python src/main.py backtest --symbol BTCUSDT --profile conservative
python src/main.py backtest --symbol BTCUSDT --profile balanced
python src/main.py backtest --symbol BTCUSDT --profile aggressive
```

Compare all profiles and write `outputs/profile_comparison.json`:

```bash
python src/main.py backtest --symbol BTCUSDT --profile all
```

Compare benchmark strategies and write `outputs/benchmark_comparison.json`:

```bash
python src/main.py backtest --symbol BTCUSDT --benchmarks
```

Run Phase 1.6 research and write benchmark, regime, attribution, and final research reports:

```bash
python src/main.py backtest --symbol BTCUSDT --research
```

Run Phase 1.7 trend participation research:

```bash
python src/main.py backtest --symbol BTCUSDT --trend-participation
```

Run Phase 1.8 trade duration and profit-capture analysis:

```bash
python src/main.py backtest --symbol BTCUSDT --profit-capture
```

Run Phase 1.9 Trend Rider analysis:

```bash
python src/main.py backtest --symbol BTCUSDT --trend-rider
```

Run Phase 1.10 Hybrid Trend Rider analysis:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_trend_rider
```

Run Phase 1.11 Hybrid Runner Optimization profiles:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_conservative
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_balanced
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_aggressive
```

Run Phase 1.12 Market Structure Stop analysis:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type swing_low
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type support_zone
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type atr
```

Run Phase 1.13 Trend Holding Engine comparison:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy trend_holding
```

Run Phase 1.14 Regime-Gated Trend Holding comparison:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy regime_gated_trend_holding
```

Backtest progress is printed to stderr while the final JSON summary remains on stdout:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main backtest --symbol BTCUSDT --start 2024-01-01 --progress-interval 5000
```

Disable progress output with:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main backtest --symbol BTCUSDT --start 2024-01-01 --quiet
```

Force a full cache rebuild for the requested range:

```bash
python src/main.py backtest --symbol BTCUSDT --start 2015-01-01 --refresh-cache
```

Compatibility entrypoint:

```bash
python src/main.py backtest --symbol BTCUSDT --start 2017-01-01 --end latest
```

Artifacts are written to `outputs/output.json`, `outputs/chart.png`, and `outputs/signal_journal.json`.

Backtest artifacts are written to:

- `outputs/backtest_report.json`
- `outputs/backtest_trades.csv`
- `outputs/equity_curve.csv`
- `outputs/equity_curve.png`
- `outputs/profile_comparison.json` when running `--profile all`
- `outputs/benchmark_comparison.json` when running `--benchmarks`
- `outputs/regime_analysis.json` when running `--research`
- `outputs/filter_attribution.json` when running `--research`
- `outputs/strategy_research_report.json` when running `--research`
- `outputs/trend_participation_report.json` when running `--trend-participation`
- `outputs/profit_capture_analysis.json` when running `--profit-capture`
- `outputs/trend_rider_analysis.json` when running `--trend-rider`
- `outputs/hybrid_trend_rider_report.json` when running `--strategy hybrid_trend_rider`
- `outputs/hybrid_runner_optimization.json` when running `--strategy hybrid_conservative`, `--strategy hybrid_balanced`, or `--strategy hybrid_aggressive`
- `outputs/market_structure_stop_report.json` when running `--strategy aggressive --stop-type swing_low`, `support_zone`, or `atr`

Historical candles are cached under `data/cache/`, for example `data/cache/BTCUSDT_1h.csv`.
The cache is range-aware: if a requested backtest starts before the cached range or ends after it, the backtester downloads only the missing candles and merges them into the cache.
Use `--refresh-cache` to ignore existing cache files and rebuild the requested range.

Backtest execution is intentionally stricter than signal generation. A simulated long entry opens only when:

- Final decision is `BUY`, `BUY WATCH`, or `STRONG BUY`
- Multi-timeframe alignment is `BULLISH_ALIGNMENT`
- RR ratio meets the selected profile threshold
- Volume ratio meets the selected profile threshold
- Primary market regime is not `BEAR`
- Daily setup is not `BEAR_TREND`
- 4h price is above 4h EMA20
- Optional 4h MACD bullish confirmation passes when the profile requires it
- Optional 1h price above EMA20 confirmation passes when the profile requires it
- Simulator is in `NO_POSITION`
- Cooldown is not active

Rejected entry signals are counted by reason in `backtest_report.json`.

Profile parameters:

- `conservative`: `min_rr_ratio=2.5`, `min_volume_ratio=1.2`, 4h MACD bullish required, 1h price above EMA20 required, `allocation_per_trade=0.20`
- `balanced`: `min_rr_ratio=2.0`, `min_volume_ratio=1.0`, 4h MACD bullish required, 1h price above EMA20 required, `allocation_per_trade=0.25`
- `aggressive`: `min_rr_ratio=1.5`, `min_volume_ratio=0.8`, 4h MACD bullish not required, 1h price above EMA20 not required, `allocation_per_trade=0.30`

Backtest exits are managed by Phase 1.5.1 rules:

- `STOP_LOSS` is always active and can exit immediately
- `TAKE_PROFIT` waits for the 48-hour minimum hold and exits at `target_1`, or `entry_price + 2R` when `target_1` is unavailable
- `TRAILING_STOP` waits for the 48-hour minimum hold, activates after +2R, and trails by 1R
- `MOMENTUM_EXIT` waits for the 48-hour minimum hold and requires profitable trade, 4h bearish MACD, and 4h close below EMA20
- `TIME_EXIT` waits for the 48-hour minimum hold and exits after 7 days if the trade is not profitable
- `BEAR_TREND` waits for the 48-hour minimum hold unless daily setup is `BEAR_TREND` and daily price is below daily EMA20

The backtest report also includes blocked early exits, blocked momentum exits, average R multiple, average R multiple by exit reason, best trade, and worst trade.

Signal JSON shape:

```json
{
  "timestamp": "2026-06-09T00:00:00+00:00",
  "symbol": "BTCUSDT",
  "market_data_source": "BINANCE",
  "position_mode": "NO_POSITION",
  "price": 108250,
  "ema20": 107900,
  "ema50": 106700,
  "ema200": 101500,
  "rsi": 61.4,
  "macd": "bullish",
  "volume_ratio": 1.8,
  "trend_score": 10,
  "momentum_score": 8,
  "volume_score": 7,
  "bottom_score": 7,
  "support": 104500,
  "resistance": 112000,
  "distance_to_support": 3750,
  "distance_to_resistance": 3750,
  "sr_score": 5,
  "risk": 3750,
  "reward": 3750,
  "rr_ratio": 1,
  "rr_score": 4,
  "market_regime": "BULL",
  "regime_score": 10,
  "setup": "TREND_FOLLOWING",
  "setup_score": 10,
  "setup_confidence": 90,
  "setup_reason": [
    "Price is above EMA20, EMA50, and EMA200",
    "Trend score confirms strong uptrend"
  ],
  "recent_swing_high": 112000,
  "recent_swing_low": 104500,
  "decision": "BUY",
  "decision_meaning": "Potential long entry setup detected.",
  "confidence": 73,
  "entry_zone": {
    "low": 107900,
    "high": 108250
  },
  "stop_loss": 104500,
  "target_1": null,
  "target_2": 112000,
  "rationale": [
    "BTCUSDT is above EMA20, EMA50, and EMA200",
    "RSI is healthy and MACD is bullish",
    "Bottom detection suggests early reversal formation",
    "Trade quality scores are SR 5/10, RR 4/10 at 1.00R, and market regime BULL",
    "Setup detected: TREND_FOLLOWING with 90% setup confidence",
    "Decision is BUY with 73% confidence from deterministic Phase 1 rules"
  ],
  "multi_timeframe": {
    "alignment": "BULLISH_ALIGNMENT",
    "alignment_score": 95,
    "summary": "2 of 3 timeframes are bullish. Long setups are allowed.",
    "timeframes": {
      "1h": {
        "setup": "TREND_FOLLOWING",
        "decision": "BUY",
        "trend_score": 10,
        "momentum_score": 8,
        "volume_score": 7,
        "bottom_score": 7,
        "sr_score": 5,
        "rr_score": 4,
        "regime_score": 10,
        "setup_confidence": 90,
        "price": 108250,
        "rsi": 61.4,
        "macd": "bullish",
        "ema20": 107900,
        "ema50": 106700,
        "ema200": 101500,
        "market_regime": "BULL"
      },
      "4h": {
        "setup": "TREND_FOLLOWING",
        "decision": "BUY",
        "market_regime": "BULL"
      },
      "1d": {
        "setup": "RANGE_BOUND",
        "decision": "WAIT",
        "market_regime": "NEUTRAL"
      }
    }
  },
  "final_decision": "BUY",
  "final_decision_reason": "Multi-timeframe alignment supports long-side setups.",
  "signal_journal": {
    "path": "outputs/signal_journal.json",
    "inserted": true,
    "evaluated_count": 0
  }
}
```

---

## Phase 2

Add notification and monitoring.

Features:

- Telegram Bot
- Watchlists
- Scheduled execution
- Signal history database

Outputs:

- Telegram alerts
- Historical signal tracking

---

## Phase 3

Add AI News Agent.

Features:

- News ingestion
- Sentiment analysis
- SEC filing analysis
- Earnings analysis
- AI-generated trade explanations

Outputs:

- News score
- Event impact score
- AI trade rationale

---

## Phase 4

Add Market Regime Agent.

Features:

- SPY trend analysis
- QQQ trend analysis
- BTC trend analysis
- Volatility monitoring

Outputs:

- Bull Market
- Bear Market
- Sideways Market

---

## Phase 5

Add Risk Engine.

Features:

- Position sizing
- Stop loss recommendations
- Portfolio exposure controls
- Maximum daily loss protection

Outputs:

- Risk score
- Recommended position size

---

## Phase 6

Paper Trading.

Features:

- Simulated trades
- Trade journal
- Performance analytics

Metrics:

- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Profit Factor

---

## Phase 7

Broker Integration.

Potential Brokers:

- Robinhood
- Alpaca
- Binance

Rules:

- Human confirmation required
- No autonomous trading

---

# Architecture

Market Data Sources
↓
OHLCV Data Loader
↓
Indicator Engine
↓
Skill Scoring Engine
↓
Decision Engine
↓
Output Layer

Future:

News Agent
↓
Market Regime Agent
↓
Risk Engine
↓
Decision Engine

---

# Technical Stack

Language:

- Python 3.11+

Libraries:

- pandas
- numpy
- matplotlib

Future:

- FastAPI
- PostgreSQL
- SQLAlchemy
- Kafka
- Apache Flink
- OpenAI API

---

# Folder Structure

trading-agent/

├── README.md

├── PROJECT.md

├── AGENTS.md

├── pyproject.toml

├── requirements.txt

├── architecture/

├── src/

│ ├── decision/

│ │   ├── __init__.py

│ │   └── decision_engine.py

│ ├── scoring/

│ │   ├── __init__.py

│ │   ├── support_resistance_skill.py

│ │   ├── risk_reward_skill.py

│ │   ├── market_regime_skill.py

│ │   ├── setup_detection_skill.py

│ │   └── multi_timeframe_skill.py

│ └── trading_agent/

│     ├── config.py

│     ├── models.py

│     ├── data.py

│     ├── indicators.py

│     ├── scoring.py

│     ├── decision.py

│     ├── output.py

│     └── main.py

├── tests/

├── outputs/

│ ├── output.json

│ └── chart.png

---

# Decision Framework

Each skill contributes a score.

Trend Skill

- EMA20
- EMA50
- EMA200

Score: 0-10

Momentum Skill

- RSI
- MACD

Score: 0-10

Volume Skill

- Relative Volume
- Volume Spike

Score: 0-10

Bottom Skill

- Price near recent swing low
- RSI reversal zone
- Improving MACD histogram

Score: 0-10

Support / Resistance Skill

- Support is the most recent meaningful swing low
- Resistance is the most recent meaningful swing high
- Buying near support scores higher than buying near resistance

Score: 0-10

Risk / Reward Skill

- Risk = current price - support
- Reward = resistance - current price
- RR ratio = reward / risk

Score: 0-10

Market Regime Skill

- BULL: price > EMA200
- NEUTRAL: price between EMA50 and EMA200
- BEAR: price < EMA50

Score: 0, 5, or 10

Setup Detection Skill

- BOTTOMING
- BREAKOUT
- TREND_FOLLOWING
- PULLBACK
- RANGE_BOUND
- BEAR_TREND

Multi-Timeframe Skill

- Default timeframes: 1h, 4h, 1d
- Summarizes setup, scores, regime, indicators, and decision for each timeframe
- Classifies alignment as BULLISH_ALIGNMENT, BEARISH_ALIGNMENT, REVERSAL_FORMING, PULLBACK_IN_UPTREND, PULLBACK_IN_DOWNTREND, MIXED_ALIGNMENT, or RANGE_ALIGNMENT
- `PULLBACK_IN_UPTREND` requires daily bullish trend above EMA200, a 4h pullback, and 1h weakness

Signal Journal

- Writes `outputs/signal_journal.json`
- Stores timestamp, symbol, setup, final_decision, and price
- Evaluates +1 day, +3 days, +7 days, and +30 days when enough future candle data exists
- Records future price, percent change, and whether the signal was favorable, unfavorable, neutral, or observed

Backtesting Engine

- Replays historical candles locally from cached OHLCV files
- Uses the existing multi-timeframe signal pipeline with 1h, 4h, and 1d context
- Avoids lookahead by slicing each timeframe to candles available at the replay timestamp
- Writes report, trades, equity curve, and optional equity chart artifacts
- Compares strategy equity with buy-and-hold equity
- Compares Buy & Hold, EMA200, Golden Cross, RSI Trend, and Agent Aggressive benchmarks with return, CAGR, max drawdown, Sharpe, and total trades
- Splits benchmark results into 2018 Bear, 2019 Recovery, 2020 Bull, 2021 Bull, 2022 Bear, 2023 Recovery, 2024 Bull, and 2025-2026 Current regimes
- Runs aggressive-profile attribution experiments by disabling MACD, volume, RR, and alignment filters one at a time
- Compares Agent Aggressive, Bull Mode Agent, and RSI Trend for trend participation research
- Bull Mode activates when daily RSI is above 55 and daily close is above daily EMA200, then uses `min_rr_ratio=1.2`, `min_volume_ratio=0.5`, and allows `PULLBACK_IN_UPTREND`
- Analyzes trade duration, top winning trades, missed upside after exit, and profit capture ratio for Agent Aggressive, Bull Mode Agent, and RSI Trend
- Compares Current Aggressive with Trend Rider Aggressive, which takes 50% profit at +2R, 25% at +4R, and lets the remaining 25% run until daily RSI < 50, daily EMA20 < daily EMA50, or a 10% trailing stop
- Marks any still-open Phase 1.9 comparison position at the backtest boundary with `END_OF_BACKTEST` so profit capture includes open runners
- Compares Agent Aggressive, Trend Rider Aggressive, and Hybrid Trend Rider for Phase 1.10
- Hybrid Trend Rider uses aggressive entries with 25% allocation, sells 50% at +2R, sells 25% at +4R, moves the stop to breakeven after TP1, and exits the runner on daily RSI < 50, daily EMA20 < daily EMA50, daily price < daily EMA50, 15% runner trailing stop, or runner drawdown above 25%
- Compares Agent Aggressive, Trend Rider Aggressive, Hybrid Conservative, Hybrid Balanced, and Hybrid Aggressive for Phase 1.11 runner optimization
- Hybrid Conservative keeps a 25% runner and exits on daily RSI < 50, daily EMA20 < daily EMA50, or a 15% trailing stop
- Hybrid Balanced keeps a 40% runner and exits on daily RSI < 45, daily EMA50 < daily EMA100, or a 20% trailing stop
- Hybrid Aggressive keeps a 50% runner and exits on weekly RSI < 45, weekly EMA20 < weekly EMA50, or a 25% trailing stop
- Compares aggressive current stops, ATR stops, swing-low stops, and support-zone stops for Phase 1.12
- Market Structure Stops keep aggressive entries unchanged and only replace stop placement
- `src/risk/structure_stop_engine.py` owns ATR, swing-low, support-zone detection, stop priority, and stop distance reporting
- Swing-low stops use the most recent swing low over 20 candles minus 0.5 ATR, with a 1.5 ATR minimum distance
- Support-zone stops cluster nearby swing lows over 50 candles, use the strongest support zone low minus 0.5 ATR, and fall back to swing low then ATR when structure is unavailable
- ATR stops use entry price minus 1.5 ATR
- Compares Agent Aggressive with Trend Holding Engine for Phase 1.13
- Trend Holding sells 50% at +2R, sells another 25% at +4R, moves the stop to breakeven after TP1, and lets the remaining 25% runner exit only on daily close below EMA50, daily MACD bearish, or a 20% trailing stop
- Trend Holding reports return, drawdown, Sharpe, profit capture, runner metrics, and a top-50 missed-opportunity recheck
- Compares Agent Aggressive, Trend Holding, and Regime-Gated Trend Holding for Phase 1.14
- Regime-Gated Trend Holding activates runners only in `STRONG_BULL`, defined by daily EMA20 > EMA50 > EMA200, daily RSI > 55, daily MACD bullish, and weekly close > weekly EMA20
- Regime gate reduces runner size by 50% above 15% portfolio drawdown and disables new runners above 20% portfolio drawdown
- Compares Agent Aggressive, Trend Holding, Regime-Gated Trend Holding, and Regime-Gated + Portfolio Governor for Phase 1.15
- Portfolio Governor tracks current equity, peak equity, drawdown, risk state, portfolio stops, volatility-adjusted position size, average position size, and average runner size
- Risk states are `NORMAL`, `CAUTION`, `DEFENSIVE`, and `CAPITAL_PRESERVATION`
- Position sizing uses 1% risk per trade, then adjusts allocation by drawdown state and ATR volatility
- Portfolio stop triggers above 25% drawdown, closes active runners, disables new runners, and recovers only after drawdown falls below 15%
- Run with `python src/main.py backtest --strategy regime_gated_portfolio_governor`
- Writes `outputs/portfolio_risk_governor_report.json`
- Cross-Asset Validation tests Agent Aggressive across BTCUSDT, ETHUSDT, SOLUSDT, SPY, and QQQ without per-asset tuning
- Optional validation assets are TQQQ and NVDA
- Cross-Asset Validation ranks assets by return, Sharpe, drawdown, and robustness score
- Robustness score is 100 points across positive return, Sharpe > 0.8, drawdown < 25%, profit factor > 1, and win rate > 40%
- Run with `python src/main.py backtest --strategy cross_asset_validation`
- Restrict assets with `python src/main.py backtest --strategy cross_asset_validation --assets BTCUSDT ETHUSDT SOLUSDT`
- Writes `outputs/cross_asset_validation.json`
- Equity Data Adapter validates SPY and QQQ through Yahoo Finance, Stooq, optional Alpha Vantage, and optional Twelve Data fallbacks
- Alpha Vantage requires `ALPHA_VANTAGE_API_KEY`; Twelve Data requires `TWELVE_DATA_API_KEY`
- 4h equity candles are generated from 1h aggregation when providers do not provide 4h directly
- Run equity validation with `python src/main.py validate-equities`
- Writes `outputs/equity_validation_report.json` and `outputs/data_provider_diagnostics.json`
- Exit Optimization tests production Agent Aggressive entries with alternate exit-only models for Phase 1.17
- Exit models include baseline, EMA20 trend rider, EMA20/EMA50 cross, ATR trailing stops, chandelier exit, partial-profit trend ride, multi-target exits, ADX trend strength, volatility adaptive exit, and hybrid partial/EMA20/ATR variants
- Run with `python src/main.py backtest --strategy exit_optimization`
- Writes `outputs/exit_optimization_report.json` and `outputs/exit_model_rankings.json`

Decision Engine v2 considers setup and position mode together. In no-position mode, `BREAKOUT` and `TREND_FOLLOWING` map to `BUY`, `PULLBACK` maps to `BUY WATCH`, `RANGE_BOUND` maps to `WAIT`, and `BEAR_TREND` maps to `AVOID LONG`. In holding-position mode, bullish continuation can map to `ADD` or `HOLD`, mixed setups map to `HOLD`, and bearish setups map to `REDUCE` or `EXIT`.

Decision Engine v2

No-position labels:

- BUY
- BUY WATCH
- WAIT
- AVOID LONG
- WATCH FOR REVERSAL

Holding-position labels:

- ADD
- HOLD
- REDUCE
- EXIT

Confidence is calculated from trend, momentum, volume, bottom, support/resistance, risk/reward, and regime scores.

Decision priority prevents ambiguous bearish signals. In no-position mode, bearish setups emit `AVOID LONG` instead of `SELL`, so the output cannot be mistaken for a short instruction. When `bottom_score >= 8`, the engine does not emit an exit-style bearish label; no-position mode emits `WATCH FOR REVERSAL` with a cautious confirmation rationale.

Multi-timeframe alignment is applied as a final decision layer. Bearish alignment blocks primary-timeframe BUY decisions, reversal-forming alignment emits WATCH FOR REVERSAL, pullback-in-uptrend alignment emits BUY WATCH, pullback-in-downtrend alignment emits AVOID LONG, bullish alignment allows BUY or BUY WATCH, and mixed/range alignment waits.

Entry zones, stop loss, targets, and rationale are deterministic rule outputs. The system does not place trades.

---

# Long-Term Vision

Create a modular AI trading platform using:

- Multi-agent architecture
- Real-time market streams
- Kafka event processing - optional
- Apache Flink analytics - optional
- AI-powered reasoning
- Explainable trading decisions

The system should be capable of monitoring multiple assets simultaneously while maintaining strict risk management and human oversight.
