# Trading Agent Project

Goal:
Build an AI-assisted trading decision system running on a MacBook.

Phase 1:

- Python only
- No broker integration
- No live trading
- Crypto first (BTCUSDT, ETHUSDT, SOLUSDT)
- Binance API market data by default
- Bybit API market data for HYPEUSDT
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
- Multi-Timeframe Analysis
- Signal Journal
- Backtesting Engine

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
- signal_journal.json
- backtest_report.json
- backtest_trades.csv
- equity_curve.csv
- benchmark_comparison.json
- regime_analysis.json
- filter_attribution.json
- strategy_research_report.json
- trend_participation_report.json
- profit_capture_analysis.json
- trend_rider_analysis.json
- hybrid_trend_rider_report.json
- hybrid_runner_optimization.json
- market_structure_stop_report.json
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
Multi-Timeframe Alignment
↓
Decision Engine
↓
JSON + PNG + Journal + Backtest Reports

Phase 1.6 Research:

- Benchmark comparison
- Regime analysis
- Filter attribution
- Strategy research report

Phase 1.7 Research:

- Trend participation comparison
- Bull Mode Agent
- Trade regime tagging

Phase 1.8 Research:

- Trade duration analysis
- Missed upside after exit
- Profit capture ratio

Phase 1.9 Research:

- Trend Rider Aggressive comparison
- Partial profit taking at +2R and +4R
- 25% trend runner with daily RSI, daily EMA, and 10% trailing exits

Phase 1.10 Research:

- Hybrid Trend Rider comparison
- 25% aggressive allocation with +2R and +4R partial exits
- Breakeven stop after TP1
- Runner exits for daily RSI, daily EMA, daily EMA50 break, 15% trailing stop, and 25% runner drawdown

Phase 1.11 Research:

- Hybrid Runner Optimization
- Compare Agent Aggressive, Trend Rider, Hybrid Conservative, Hybrid Balanced, and Hybrid Aggressive
- Conservative runner: 25% runner, daily RSI < 50, daily EMA20 < daily EMA50, 15% trailing stop
- Balanced runner: 40% runner, daily RSI < 45, daily EMA50 < daily EMA100, 20% trailing stop
- Aggressive runner: 50% runner, weekly RSI < 45, weekly EMA20 < weekly EMA50, 25% trailing stop
- Rank profiles by Sharpe, return, profit capture, and drawdown

Phase 1.12 Research:

- Market Structure Stop Engine
- Compare aggressive current stops, ATR stops, swing-low stops, and support-zone stops
- Risk module: src/risk/structure_stop_engine.py
- ATR stop: entry price - 1.5 ATR
- Swing low stop: most recent swing low - 0.5 ATR, with 1.5 ATR minimum distance
- Support zone stop: strongest clustered swing-low support zone - 0.5 ATR, with 1.5 ATR minimum distance
- Track stop type usage, stop distance, stop-out count, profit capture, and survived stop-outs

Phase 1.13 Research:

- Trend Holding Engine
- Compare Agent Aggressive versus Trend Holding Engine
- TP1: +2R, sell 50%, move stop to breakeven
- TP2: +4R, sell 25%, activate 25% runner
- Runner exits on daily close below EMA50, daily MACD bearish, or 20% trailing stop
- Track runner returns, runner holding time, TP hits, runner activations, and top-50 missed-opportunity recheck

Phase 1.14 Research:

- Regime-Gated Trend Holding
- Compare Agent Aggressive, Trend Holding, and Regime-Gated Trend Holding
- Activate runners only in STRONG_BULL macro regime
- STRONG_BULL requires daily EMA20 > EMA50 > EMA200, daily RSI > 55, daily MACD bullish, and weekly close > weekly EMA20
- Use standard Agent Aggressive exits outside STRONG_BULL
- Reduce runner size above 15% portfolio drawdown and disable new runners above 20% drawdown
- Track regime periods, disabled runners, profit capture, and top-50 missed-opportunity recheck

Future phases:

- News Agent
- Market Regime Agent
- Risk Engine
- Paper Trading
- Broker Integration
