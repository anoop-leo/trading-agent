# Trading Agent Project

Goal:
Build an AI-assisted trading decision system running on a MacBook.

Phase 1:

- Python only
- No broker integration
- No live trading
- Crypto first (BTCUSDT, ETHUSDT, SOLUSDT)
- Binance API market data
- pandas DataFrames

Indicators:

- RSI
- EMA20
- EMA50
- EMA200
- MACD
- Bollinger Bands
- Volume Spike
- Bottom Detection
- Support and Resistance
- Risk / Reward
- Market Regime
- Setup Detection

Output:

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

Architecture:

Binance API
↓
OHLCV Loader
↓
Indicator Engine
↓
Scoring Engine
↓
Decision Engine
↓
JSON + PNG

Future phases:

- News Agent
- Market Regime Agent
- Risk Engine
- Paper Trading
- Broker Integration
