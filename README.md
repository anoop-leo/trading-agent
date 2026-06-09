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

Decision terminology defaults to no-position mode. Use holding mode when evaluating an existing position:

```bash
PYTHONPATH=src venv_trading/bin/python -m trading_agent.main --symbol BTCUSDT --position-mode HOLDING
```

Test:

```bash
PYTHONPATH=src venv_trading/bin/python -m unittest discover -s tests
```

Artifacts are written to `outputs/output.json` and `outputs/chart.png`.

Signal JSON shape:

```json
{
  "symbol": "BTCUSDT",
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
  ]
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

│ │   └── setup_detection_skill.py

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
