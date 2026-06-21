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
- portfolio_risk_governor_report.json
- cross_asset_validation.json
- equity_validation_report.json
- data_provider_diagnostics.json
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

Phase 1.15 Research:

- Portfolio Risk Governor
- Compare Agent Aggressive, Trend Holding, Regime-Gated Trend Holding, and Regime-Gated + Portfolio Governor
- Track account equity, peak equity, drawdown, and risk state
- NORMAL uses 100% risk allocation and enables runners
- CAUTION uses 75% risk allocation and enables runners
- DEFENSIVE uses 50% risk allocation and disables runners
- CAPITAL_PRESERVATION uses 25% risk allocation and disables new Trend Holding runners
- Use 1% risk per trade with ATR volatility adjustment
- Trigger portfolio stop above 25% drawdown, close active runners, and recover after drawdown falls below 15%
- Track risk state counts, average position size, average runner size, portfolio stop count, and defensive mode hours

Phase 1.16 Research:

- Cross-Asset Validation
- Use Agent Aggressive without modifying strategy parameters
- Default assets: BTCUSDT, ETHUSDT, SOLUSDT, SPY, QQQ
- Optional assets: TQQQ, NVDA
- Preferred common history starts in 2018; minimum target is 2020
- Track return, CAGR, Sharpe, max drawdown, profit factor, win rate, total trades, and profit capture ratio
- Compute crypto and equity class averages
- Compute robustness score from positive return, Sharpe > 0.8, drawdown < 25%, profit factor > 1, and win rate > 40%
- Report rejected-entry filters, losing exit reasons, production asset recommendations, and trend-following assessment
- Output: outputs/cross_asset_validation.json

Phase 1.16B Research:

- Equity Data Adapter Fix & Validation
- Adapter: src/data/equity_data_adapter.py
- Command: python src/main.py validate-equities
- Required equities: SPY, QQQ
- Optional equities: IWM, DIA, TQQQ, NVDA
- Provider fallback chain: Yahoo Finance, Stooq, Alpha Vantage, Twelve Data
- Alpha Vantage is optional and requires ALPHA_VANTAGE_API_KEY
- Twelve Data is optional and requires TWELVE_DATA_API_KEY
- Validate OHLCV exists, duplicate timestamps, missing timestamps, sort order, and minimum 3 years history
- Generate 4h candles from 1h aggregation when provider lacks 4h
- Rerun Agent Aggressive unchanged for equity validation
- Output: outputs/equity_validation_report.json
- Diagnostics: outputs/data_provider_diagnostics.json

Phase 1.17 Research:

- Exit Optimization Engine
- Module: src/research/exit_optimization_engine.py
- Command: python src/main.py backtest --strategy exit_optimization
- Uses Agent Aggressive baseline entries from 2020-present BTCUSDT
- Keeps entry logic, stops, sizing, risk filters, market regime, support/resistance, and signal generation unchanged
- Replays exit-only models: baseline, EMA20 trend rider, EMA20/EMA50 cross, ATR trailing exits, chandelier exit, partial-profit trend ride, multi-target exits, trend-strength ADX exit, volatility adaptive exit, and hybrid partial/EMA20/ATR exits
- Ranks by profit capture ratio, Sharpe, total return, drawdown, and profit factor
- Output: outputs/exit_optimization_report.json
- Rankings: outputs/exit_model_rankings.json

Future phases:

- News Agent
- Market Regime Agent
- Risk Engine
- Paper Trading
- Broker Integration
