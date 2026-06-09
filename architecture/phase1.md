# Phase 1 - Technical Signal Agent

## Objective

Build a local-first trading signal engine that runs entirely on a MacBook.

This phase focuses only on technical analysis.

No broker integration.

No live trading.

No news analysis.

No AI reasoning.

No Telegram alerts.

The goal is to establish a reliable signal generation framework.

---

# Success Criteria

Given a symbol such as BTCUSDT:

The system should:

1. Download market data
2. Calculate indicators
3. Calculate skill scores
4. Generate a deterministic trading decision
5. Save outputs

Example:

WATCH FOR REVERSAL

Symbol: BTCUSDT

Trend Score: 5/10
Momentum Score: 4/10
Volume Score: 7/10
Bottom Score: 8/10

Confidence: 60%

Reason:
Bottom detection suggests early reversal formation while BTCUSDT is still below longer trend averages.

---

# Data Source

Primary:

Binance REST API

Endpoint:

/api/v3/klines

Supported Symbols:

- BTCUSDT
- ETHUSDT
- SOLUSDT

Default Timeframe:

1h

Default History:

500 candles

---

# Architecture

Binance API
↓
Data Loader
↓
Indicator Engine
↓
Skill Scoring Engine
↓
Decision Engine
↓
Output Layer

---

# Module 1 - Data Loader

Location:

src/data/

Responsibilities:

- Fetch OHLCV candles
- Normalize data
- Convert to pandas DataFrame
- Handle API failures

Output:

DataFrame

Columns:

timestamp
open
high
low
close
volume

---

# Module 2 - Indicator Engine

Location:

src/indicators/

Responsibilities:

Calculate:

EMA20
EMA50
EMA200

RSI14

MACD
MACD Signal
MACD Histogram

Bollinger Bands

Volume Moving Average

Output:

DataFrame with indicator columns

---

# Module 3 - Skill Scoring Engine

Location:

src/scoring/

Purpose:

Convert indicators into scores.

---

## Trend Skill

Maximum Score:

10

Rules:

Price > EMA20 = +2

Price > EMA50 = +3

Price > EMA200 = +5

---

## Momentum Skill

Maximum Score:

10

Rules:

RSI between 50 and 70 = +4

MACD Bullish = +6

---

## Volume Skill

Maximum Score:

10

Rules:

Volume > 150% Average = +5

Volume > 200% Average = +10

---

## Bottom Skill

Maximum Score:

10

Rules:

- Price near recent swing low
- RSI in reversal zone
- MACD histogram improving
- Latest candle closes green
- If volume_ratio < 0.8, bottom_score cannot exceed 6
- If volume_ratio < 0.5, bottom_score cannot exceed 4
- Rationale includes "Bottom signal lacks volume confirmation." when volume_ratio < 0.8

---

## Support / Resistance Skill

Maximum Score:

10

Rules:

- Support is the most recent meaningful swing low
- Resistance is the most recent meaningful swing high
- Score 10 when price is near support and far from resistance
- Score 5 when price is between support and resistance
- Score 0 when price is near resistance

---

## Risk / Reward Skill

Maximum Score:

10

Rules:

- Risk = current price - support
- Reward = resistance - current price
- RR ratio = reward / risk
- RR < 1.0 scores 0
- RR 1.0-2.0 scores 4
- RR 2.0-3.0 scores 7
- RR > 3.0 scores 10

---

## Market Regime Skill

Maximum Score:

10

Rules:

- BULL when price > EMA200; score 10
- NEUTRAL when price is between EMA50 and EMA200; score 5
- BEAR when price < EMA50; score 0

---

## Setup Detection Skill

Purpose:

Classify current market structure into one deterministic setup.

Supported Setups:

- BOTTOMING
- BREAKOUT
- TREND_FOLLOWING
- PULLBACK
- RANGE_BOUND
- BEAR_TREND

Priority:

1. BREAKOUT
2. TREND_FOLLOWING
3. BOTTOMING
4. PULLBACK
5. RANGE_BOUND
6. BEAR_TREND

---

# Module 4 - Decision Engine

Location:

src/decision/decision_engine.py

Input:

Trend Score
Momentum Score
Volume Score
Bottom Score
Support / Resistance Score
Risk / Reward Score
Market Regime Score
Current Price
EMA20
EMA50
EMA200
Recent Swing High
Recent Swing Low
RR Ratio
Market Regime
Support
Setup
Setup Score
Setup Confidence
Position Mode

Maximum:

70

Normalize:

Confidence = Total × 100 / 70

---

Decision Rules

Position modes:

- NO_POSITION, the default mode for users who do not currently hold the asset
- HOLDING, for evaluating an existing position

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

No-position behavior:

BUY

- confidence >= 70
- setup = BREAKOUT or TREND_FOLLOWING

BUY WATCH

- setup = PULLBACK

WATCH FOR REVERSAL

- bottom_score >= 8 unless BUY conditions are met
- setup = BOTTOMING
- If trend_score <= 3, momentum_score <= 3, and market_regime = BEAR, confidence is clamped to 55-65
- Rationale must state: "Market is bearish, but bottom detection is strong. Wait for confirmation before buying."

WAIT

- confidence between 40 and 70
- setup = RANGE_BOUND

AVOID LONG

- market_regime = BEAR
- confidence below 40
- setup = BEAR_TREND
- bottom_score < 8

Holding-position behavior:

ADD

- confidence >= 70 and market_regime = BULL
- setup = BREAKOUT or strong TREND_FOLLOWING

HOLD

- mixed setup
- setup = RANGE_BOUND, PULLBACK, or BOTTOMING

REDUCE

- market_regime = BEAR
- bottom_score < 8
- strong exit conditions are not met

EXIT

- market_regime = BEAR
- trend_score <= 3
- momentum_score <= 3
- bottom_score <= 4
- volume_score >= 5 OR price breaks below support

Decision meanings:

- AVOID LONG: Do not open a new long position here.
- REDUCE: Bearish risk is rising; consider reducing exposure.
- EXIT: Bearish conditions are strong; consider exiting the current position.

---

# Module 5 - Output Layer

Location:

src/output/

Responsibilities:

Create:

output.json

chart.png

terminal summary

---

# JSON Output

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
"rr_ratio": 1.0,
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

---

# Chart Output

Create PNG chart with:

- Panel 1: Candlesticks, EMA20, EMA50, EMA200
- Panel 2: RSI
- Panel 3: MACD
- Panel 4: Volume bars and volume moving average
- Support, resistance, and entry price lines on the price panel
- Setup label in the chart title

Save:

outputs/chart.png

---

# Project Structure

src/

├── decision/

│   ├── __init__.py

│   └── decision_engine.py

├── scoring/

│   ├── __init__.py

│   ├── support_resistance_skill.py

│   ├── risk_reward_skill.py

│   ├── market_regime_skill.py

│   └── setup_detection_skill.py

└── trading_agent/

    ├── config.py

    ├── models.py

    ├── data.py

    ├── indicators.py

    ├── scoring.py

    ├── decision.py

    ├── output.py

    └── main.py

---

# Unit Tests

Required:

tests/

├── test_config.py

├── test_models.py

├── test_data.py

├── test_indicators.py

├── test_scoring.py

├── test_support_resistance_skill.py

├── test_risk_reward_skill.py

├── test_market_regime_skill.py

├── test_setup_detection.py

├── test_decision.py

├── test_decision_engine_v2.py

├── test_output.py

└── test_main.py

Coverage Goal:

80%+

---

# Future Compatibility

This phase must be designed to support:

Phase 2:

Telegram Alerts

Phase 3:

News Agent

Phase 4:

Market Regime Agent

Phase 5:

Risk Engine

Phase 6:

Paper Trading

Phase 7:

Broker Integration

No module should directly depend on future phases.
