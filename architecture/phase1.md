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

Secondary:

Bybit REST API for HYPEUSDT because Binance spot does not support HYPEUSDT

Endpoint:

/api/v3/klines

/v5/market/kline

Supported Symbols:

- BTCUSDT
- ETHUSDT
- SOLUSDT
- HYPEUSDT

Default Timeframe:

1h

Default Multi-Timeframes:

1h, 4h, 1d

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

## Multi-Timeframe Skill

Location:

src/scoring/multi_timeframe_skill.py

Purpose:

Compare short-term, medium-term, and long-term technical context.

Default Timeframes:

- 1h
- 4h
- 1d

For each timeframe, summarize:

- trend_score
- momentum_score
- volume_score
- bottom_score
- sr_score
- rr_score
- regime_score
- setup
- setup_confidence
- decision
- price
- rsi
- macd
- ema20
- ema50
- ema200
- market_regime

Alignment Types:

BULLISH_ALIGNMENT

- At least 2 of 3 timeframes are BREAKOUT or TREND_FOLLOWING
- Daily timeframe is not bearish

BEARISH_ALIGNMENT

- At least 2 of 3 timeframes are BEAR_TREND
- Daily timeframe is BEAR_TREND or market_regime is BEAR

REVERSAL_FORMING

- 1h is BOTTOMING
- 4h or 1d is still BEAR_TREND

PULLBACK_IN_UPTREND

- Daily timeframe is bullish and price is above EMA200
- 4h timeframe is PULLBACK
- 1h timeframe shows weakness
- Meaning: daily uptrend remains intact while 4h and 1h reset for potential continuation

PULLBACK_IN_DOWNTREND

- Daily timeframe is bearish
- 1h is BREAKOUT, TREND_FOLLOWING, PULLBACK, or RANGE_BOUND
- Meaning: short-term bounce is occurring inside a higher-timeframe downtrend

MIXED_ALIGNMENT

- Timeframes conflict
- No clear majority

RANGE_ALIGNMENT

- At least 2 of 3 timeframes are RANGE_BOUND

Alignment Score:

0-100

- BULLISH_ALIGNMENT: 80-100
- BEARISH_ALIGNMENT: 0-30
- REVERSAL_FORMING: 40-60
- PULLBACK_IN_UPTREND: 60-80
- PULLBACK_IN_DOWNTREND: 25-45
- MIXED_ALIGNMENT: 40-60
- RANGE_ALIGNMENT: 35-55

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

Multi-Timeframe Final Decision Rules:

BEARISH_ALIGNMENT

- NO_POSITION final_decision = AVOID LONG
- HOLDING final_decision = REDUCE or EXIT
- BUY is blocked

REVERSAL_FORMING

- final_decision = WATCH FOR REVERSAL
- BUY is blocked until 4h confirms

PULLBACK_IN_UPTREND

- NO_POSITION final_decision = BUY WATCH
- HOLDING final_decision = HOLD

PULLBACK_IN_DOWNTREND

- NO_POSITION final_decision = AVOID LONG
- HOLDING final_decision = REDUCE

BULLISH_ALIGNMENT

- BUY or BUY WATCH is allowed in NO_POSITION mode
- HOLD or ADD is allowed in HOLDING mode

MIXED_ALIGNMENT

- NO_POSITION final_decision = WAIT
- HOLDING final_decision = HOLD

RANGE_ALIGNMENT

- NO_POSITION final_decision = WAIT
- HOLDING final_decision = HOLD

---

# Module 5 - Output Layer

Location:

src/output/

Responsibilities:

Create:

output.json

chart.png

signal_journal.json

backtest_report.json

backtest_trades.csv

equity_curve.csv

terminal summary

---

# JSON Output

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

---

# Signal Journal

Location:

src/trading_agent/journal.py

Purpose:

Persist compact signal records and evaluate them after future candles are available.

Stored Fields:

- timestamp
- symbol
- setup
- final_decision
- price

Evaluation Horizons:

- +1 day
- +3 days
- +7 days
- +30 days

Evaluation Output:

- target_timestamp
- evaluated_timestamp
- price
- price_change_pct
- outcome

Save:

outputs/signal_journal.json

---

# Backtesting Engine

Location:

src/backtesting/

Purpose:

Replay historical candles through the existing deterministic trading agent and answer whether past decisions were useful.

Modules:

- backtest_engine.py
- benchmark_strategies.py
- benchmarks/
  - strategies.py
  - research.py
- trade_simulator.py
- performance_metrics.py
- backtest_report.py

Default Backtest:

- Symbol: BTCUSDT
- Primary timeframe: 1h
- Context timeframe: 4h
- Trend timeframe: 1d
- Start: 2017-01-01 when available
- End: latest available candle

Data Handling:

- Download required candles upfront
- Cache candles under data/cache/
- Reuse cache during replay
- Do not call Binance at every candle
- Precompute indicator columns once per timeframe before replay
- Precompute replay-safe support/resistance levels per candle
- Cache files are range-aware; missing older/newer requested candles are downloaded and merged
- --refresh-cache ignores existing cache files and rebuilds the requested range

Lookahead Rules:

- At timestamp T, only candles available at or before T are used
- Higher-timeframe candles are included only after their interval has completed
- Primary timeframe candles are evaluated only after the candle has completed
- Support, resistance, indicators, swing high, and swing low are calculated from past data only

Progress Logging:

- CLI progress is written to stderr
- Final backtest summary JSON is written to stdout
- --progress-interval controls replay progress frequency
- --quiet disables progress logging

Trade Simulation:

- Long-only
- Initial capital: 10000
- Fee: 0.1%
- Slippage: 0.05%
- BUY, BUY WATCH, or STRONG BUY can open a long only when every execution gate passes
- ADD is disabled in the backtester
- Strategy profile controls RR threshold, volume threshold, optional confirmations, and allocation per trade
- Exits are managed by deterministic Phase 1.5.1 exit rules

Strategy Profiles:

Conservative:

- min_rr_ratio = 2.5
- min_volume_ratio = 1.2
- require_4h_macd_bullish = true
- require_price_above_1h_ema20 = true
- allocation_per_trade = 0.20
- require_alignment = true
- require_rr_ratio = true
- require_volume_ratio = true
- enable_bull_market_mode = false
- bull_min_rr_ratio = 1.2
- bull_min_volume_ratio = 0.5
- bull_allow_pullback_alignment = true

Balanced:

- min_rr_ratio = 2.0
- min_volume_ratio = 1.0
- require_4h_macd_bullish = true
- require_price_above_1h_ema20 = true
- allocation_per_trade = 0.25
- require_alignment = true
- require_rr_ratio = true
- require_volume_ratio = true
- enable_bull_market_mode = false
- bull_min_rr_ratio = 1.2
- bull_min_volume_ratio = 0.5
- bull_allow_pullback_alignment = true

Aggressive:

- min_rr_ratio = 1.5
- min_volume_ratio = 0.8
- require_4h_macd_bullish = false
- require_price_above_1h_ema20 = false
- allocation_per_trade = 0.30
- require_alignment = true
- require_rr_ratio = true
- require_volume_ratio = true
- enable_bull_market_mode = false
- bull_min_rr_ratio = 1.2
- bull_min_volume_ratio = 0.5
- bull_allow_pullback_alignment = true

Entry Execution Gates:

- final_decision is BUY, BUY WATCH, or STRONG BUY
- multi_timeframe.alignment is BULLISH_ALIGNMENT
- rr_ratio >= selected profile min_rr_ratio
- volume_ratio >= selected profile min_volume_ratio
- market_regime is not BEAR
- daily timeframe setup is not BEAR_TREND
- 4h price > 4h EMA20
- 4h MACD is bullish when selected profile requires it
- 1h price is above EMA20 when selected profile requires it
- simulator position mode is NO_POSITION
- cooldown is not active
- Attribution experiments can disable alignment, RR, or volume gates through strategy profile switches
- Bull Mode can relax RR and volume gates and allow PULLBACK_IN_UPTREND entries when daily RSI > 55 and daily close > daily EMA200

Rejected Entry Reasons:

- not_buy_decision
- bearish_alignment
- low_rr_ratio
- low_volume_ratio
- bear_market_regime
- daily_bear_trend
- below_4h_ema20
- four_hour_macd_not_bullish
- below_1h_ema20
- already_holding
- cooldown_active

Exit Management:

- minimum_hold_hours defaults to 48
- STOP_LOSS is always active and can exit immediately
- TAKE_PROFIT waits for the minimum hold and exits at target_1, or entry_price + 2R when target_1 is unavailable
- TRAILING_STOP waits for the minimum hold, activates after +2R, and trails by 1R
- MOMENTUM_EXIT waits for the minimum hold and requires profitable trade, 4h bearish MACD, and 4h close below EMA20
- TIME_EXIT waits for the minimum hold and exits after 7 days if the trade is not profitable
- BEAR_TREND waits for the minimum hold unless daily setup is BEAR_TREND and daily price is below daily EMA20

Trade Audit Fields:

- open_reason
- close_reason
- entry_decision
- exit_decision
- entry_alignment
- exit_alignment
- entry_rr_ratio
- entry_volume_ratio
- entry_market_regime
- entry_daily_setup
- entry_4h_price
- entry_4h_ema20
- entry_stop_loss
- entry_target_1
- initial_risk
- r_multiple
- max_price
- trailing_stop
- rejected_entry_reasons
- entry_regime_tag
- exit_regime_tag
- entry_bull_mode_active

Metrics:

- total_return_pct
- CAGR
- win_rate
- loss_rate
- total_trades
- winning_trades
- losing_trades
- average_win_pct
- average_loss_pct
- profit_factor
- max_drawdown_pct
- sharpe_ratio
- expectancy
- buy_and_hold_return_pct
- total_decisions
- entry_attempts
- accepted_entries
- rejected_entries
- rejected_entry_reasons
- exit_reasons
- exits_before_minimum_hold
- momentum_exits_blocked
- average_holding_hours
- median_holding_hours
- average_r_multiple
- average_r_multiple_by_exit_reason
- best_trade
- worst_trade

Outputs:

- outputs/backtest_report.json
- outputs/backtest_trades.csv
- outputs/equity_curve.csv
- outputs/equity_curve.png
- outputs/profile_comparison.json when running all profiles
- outputs/benchmark_comparison.json when running benchmarks
- outputs/regime_analysis.json when running Phase 1.6 research
- outputs/filter_attribution.json when running Phase 1.6 research
- outputs/strategy_research_report.json when running Phase 1.6 research
- outputs/trend_participation_report.json when running Phase 1.7 research
- outputs/profit_capture_analysis.json when running Phase 1.8 analysis
- outputs/trend_rider_analysis.json when running Phase 1.9 analysis
- outputs/hybrid_trend_rider_report.json when running Phase 1.10 analysis
- outputs/hybrid_runner_optimization.json when running Phase 1.11 profile optimization
- outputs/market_structure_stop_report.json when running Phase 1.12 stop analysis

Run:

python src/main.py backtest --symbol BTCUSDT --start 2017-01-01 --end latest

python src/main.py backtest --symbol BTCUSDT --profile conservative

python src/main.py backtest --symbol BTCUSDT --profile balanced

python src/main.py backtest --symbol BTCUSDT --profile aggressive

python src/main.py backtest --symbol BTCUSDT --profile all

python src/main.py backtest --symbol BTCUSDT --benchmarks

python src/main.py backtest --symbol BTCUSDT --research

python src/main.py backtest --symbol BTCUSDT --trend-participation

python src/main.py backtest --symbol BTCUSDT --profit-capture

python src/main.py backtest --symbol BTCUSDT --trend-rider

python src/main.py backtest --symbol BTCUSDT --strategy hybrid_trend_rider

python src/main.py backtest --symbol BTCUSDT --start 2024-01-01 --progress-interval 5000

python src/main.py backtest --symbol BTCUSDT --start 2024-01-01 --quiet

---

# Phase 1.6 - Benchmark And Regime Analysis

Purpose:

Determine whether the deterministic trading agent adds value beyond simple benchmark strategies.

Benchmark Strategies:

- BuyAndHoldStrategy
- EMA200Strategy
- GoldenCrossStrategy
- RSITrendStrategy
- AgentAggressiveStrategy

Benchmark Metrics:

- total_return_pct
- CAGR
- max_drawdown_pct
- sharpe_ratio
- total_trades

Regime Analysis:

- 2018 Bear
- 2019 Recovery
- 2020 Bull
- 2021 Bull
- 2022 Bear
- 2023 Recovery
- 2024 Bull
- 2025-2026 Current

Filter Attribution:

- Base aggressive profile
- No MACD filter
- No Volume filter
- No RR filter
- No Alignment filter

Research Outputs:

- outputs/benchmark_comparison.json
- outputs/regime_analysis.json
- outputs/filter_attribution.json
- outputs/strategy_research_report.json

---

# Phase 1.7 - Trend Participation Research

Purpose:

Understand why RSI Trend can outperform Agent Aggressive and test whether Bull Mode increases trend participation while keeping drawdown under 20%.

Bull Mode:

- Active when daily RSI > 55
- Active when daily close > daily EMA200
- min_rr_ratio = 1.2
- min_volume_ratio = 0.5
- Allows PULLBACK_IN_UPTREND entries
- Falls back to current aggressive profile when inactive

Trade Regime Tags:

- bull
- neutral
- bear

Comparison:

- Agent Aggressive
- Bull Mode Agent
- RSI Trend

Metrics:

- total_return_pct
- CAGR
- max_drawdown_pct
- sharpe_ratio
- total_trades

Output:

- outputs/trend_participation_report.json

---

# Phase 1.8 - Trade Duration And Profit Capture Analysis

Purpose:

Determine whether existing strategy exits are cutting winning trades too early.

Strategies:

- Agent Aggressive
- Bull Mode Agent
- RSI Trend

Duration Metrics:

- average_holding_hours
- median_holding_hours
- longest_winning_trade
- longest_losing_trade
- top_10_winning_trades

Missed Opportunity Windows:

- 7 days after exit
- 30 days after exit
- 90 days after exit

Missed Opportunity Metrics:

- average_missed_gain_pct
- max_missed_gain_pct
- top_10_missed_opportunities

Profit Capture Ratio:

- Sum of positive captured trade gains divided by maximum positive trend gains available from entry through 90 days after exit

Output:

- outputs/profit_capture_analysis.json

Rules:

- Do not add indicators
- Do not change trading rules
- Analyze existing closed trades only

---

# Phase 1.9 - Trend Rider Engine

Purpose:

Increase profit capture ratio by keeping a partial runner alive after fixed R-based profit taking.

Comparison:

- Current Aggressive
- Trend Rider Aggressive

Trend Rider Exit Logic:

- Enter with 100% of the configured strategy allocation
- TP1 at +2R closes 50% of the original position
- TP2 at +4R closes 25% of the original position
- Remaining 25% becomes the trend runner

Runner Exit Conditions:

- Daily RSI < 50
- Daily EMA20 < Daily EMA50
- 10% trailing stop from the runner high

Tracked Trade Fields:

- runner_return_pct
- runner_holding_hours
- profit_capture_ratio
- partial_exits

Backtest Boundary:

- Phase 1.9 research marks still-open comparison positions with END_OF_BACKTEST so profit capture includes open trend exposure
- END_OF_BACKTEST is a research accounting event, not a live trading exit signal

Output:

- outputs/trend_rider_analysis.json

Goal:

- Profit Capture Ratio > 10%

---

# Phase 1.10 - Hybrid Position Management

Purpose:

Improve Trend Rider returns while reducing maximum drawdown.

Comparison:

- Agent Aggressive
- Trend Rider Aggressive
- Hybrid Trend Rider

Hybrid Entry:

- Uses existing aggressive entry gates
- Uses 25% allocation per trade
- Does not add indicators or new signal rules

Hybrid Exit Structure:

- TP1 at +2R closes 50% of the original position
- Stop moves to breakeven after TP1
- TP2 at +4R closes 25% of the original position
- Remaining 25% becomes the long-term runner
- No new entry is allowed while the runner is active

Runner Exit Conditions:

- Daily RSI < 50
- Daily EMA20 < Daily EMA50
- Daily price < Daily EMA50
- 15% trailing stop from highest close after runner activation
- Runner drawdown > 25%

Report Metrics:

- total_return_pct
- CAGR
- max_drawdown_pct
- sharpe_ratio
- profit_factor
- total_trades
- win_rate
- profit_capture_ratio
- average_runner_holding_hours
- average_runner_return_pct
- TP1 hit count
- TP2 hit count
- runner activation count
- runner exit reasons
- average runner drawdown
- max runner drawdown

Targets:

- Return >= 120%
- Max Drawdown < 25%
- Profit Capture Ratio > 15%

Output:

- outputs/hybrid_trend_rider_report.json

---

# Phase 1.11 - Hybrid Runner Optimization

Purpose:

Find the best balance between trend capture, drawdown control, and risk-adjusted return by profiling Hybrid Trend Rider runner exits.

Comparison:

- Agent Aggressive
- Trend Rider Aggressive
- Hybrid Conservative
- Hybrid Balanced
- Hybrid Aggressive

Hybrid Conservative:

- Runner size: 25%
- TP1 at +2R
- TP2 at +4R
- Runner exits on daily RSI < 50, daily EMA20 < daily EMA50, or 15% trailing stop

Hybrid Balanced:

- Runner size: 40%
- TP1 at +2R
- TP2 at +4R
- Runner exits on daily RSI < 45, daily EMA50 < daily EMA100, or 20% trailing stop

Hybrid Aggressive:

- Runner size: 50%
- TP1 at +2R
- TP2 at +4R
- Runner exits on weekly RSI < 45, weekly EMA20 < weekly EMA50, or 25% trailing stop

Tracked Metrics:

- average_runner_holding_hours
- median_runner_holding_hours
- average_runner_return_pct
- max_runner_return_pct
- average_runner_drawdown_pct
- max_runner_drawdown_pct
- profit_capture_ratio
- total_return_pct
- CAGR
- sharpe_ratio
- profit_factor
- max_drawdown_pct

Targets:

- Return > 120%
- Max Drawdown < 25%
- Profit Capture Ratio > 15%
- Sharpe > 0.90

Commands:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_conservative
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_balanced
python src/main.py backtest --symbol BTCUSDT --strategy hybrid_aggressive
```

Output:

- outputs/hybrid_runner_optimization.json

---

# Phase 1.12 - Market Structure Stop Engine

Purpose:

Reduce premature stop-outs during major BTC trends by widening stops using market structure while keeping aggressive entry rules unchanged.

Comparison:

- Aggressive Current
- Aggressive + ATR Stop
- Aggressive + Swing Low Stop
- Aggressive + Support Zone Stop

Stop Types:

Implemented in:

- src/risk/structure_stop_engine.py

FIXED

- Existing decision stop
- Kept as the benchmark

ATR

- stop_price = entry_price - 1.5 ATR

SWING_LOW

- Find the most recent swing low over the previous 20 candles
- stop_price = swing_low - 0.5 ATR
- If the stop is closer than 1.5 ATR, use entry_price - 1.5 ATR

SUPPORT_ZONE

- Detect swing lows over the previous 50 candles
- Cluster nearby lows
- Use the strongest support zone low
- stop_price = support_low - 0.5 ATR
- If the stop is closer than 1.5 ATR, use entry_price - 1.5 ATR
- If no support zone exists, fall back to SWING_LOW, then ATR

Tracked Trade Fields:

- entry_stop_type
- entry_stop_distance_pct
- entry_stop_distance_atr
- entry_atr
- entry_stop_candidates

Report Metrics:

- total_return_pct
- CAGR
- sharpe_ratio
- max_drawdown_pct
- profit_factor
- win_rate
- total_trades
- profit_capture_ratio
- stop_out_count
- stop_type_usage
- average_stop_distance_pct
- average_stop_distance_atr
- survived_stopouts_count

Targets:

- Return > 120%
- Profit Capture Ratio > 10%
- Max Drawdown < 20%
- Sharpe > 0.80

Commands:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type swing_low
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type support_zone
python src/main.py backtest --symbol BTCUSDT --strategy aggressive --stop-type atr
```

Output:

- outputs/market_structure_stop_report.json

---

# Phase 1.13 - Trend Holding Engine

Purpose:

Increase profit capture by changing exit quality while keeping aggressive entries deterministic.

Implemented in:

- src/strategy/trend_holding_engine.py
- src/backtesting/trend_holding_simulator.py

Position Management:

- TP1 at +2R sells 50%
- Stop moves to breakeven after TP1
- TP2 at +4R sells another 25%
- Remaining 25% becomes the runner

Trend State:

- STRONG_BULL
- BULL
- RANGE
- BEAR

Runner Exit Rules:

- Daily close below Daily EMA50
- Daily MACD turns bearish
- 20% trailing stop from highest close after runner activation

Report Metrics:

- total_return_pct
- CAGR
- max_drawdown_pct
- sharpe_ratio
- profit_factor
- profit_capture_ratio
- average_runner_return_pct
- max_runner_return_pct
- average_runner_holding_hours
- median_runner_holding_hours
- tp1_hits
- tp2_hits
- runner_activations
- missed_opportunity_recheck

Targets:

- Return > 120%
- Profit Capture Ratio > 10%
- Max Drawdown < 20%
- Sharpe >= 0.80

Command:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy trend_holding
```

Output:

- outputs/trend_holding_report.json

---

# Phase 1.14 - Regime-Gated Trend Holding

Purpose:

Activate Trend Holding only during strong bull regimes to preserve trend capture while reducing drawdown.

Implemented in:

- src/strategy/trend_holding_engine.py
- src/backtesting/regime_gated_trend_holding_simulator.py

Macro Regimes:

- STRONG_BULL
- BULL
- RANGE
- BEAR

STRONG_BULL:

- Daily EMA20 > EMA50 > EMA200
- Daily RSI > 55
- Daily MACD bullish
- Weekly close > Weekly EMA20

Exit Routing:

- STRONG_BULL uses Trend Holding exits
- BULL, RANGE, and BEAR use standard Agent Aggressive exits
- Runners are not activated outside STRONG_BULL

Risk Controls:

- Portfolio drawdown > 15% reduces runner allocation by 50%
- Portfolio drawdown > 20% disables new runners

Report Metrics:

- total_return_pct
- CAGR
- sharpe_ratio
- profit_factor
- max_drawdown_pct
- win_rate
- profit_capture_ratio
- runner_activation_count
- runner_disabled_count
- strong_bull_periods
- bull_periods
- range_periods
- bear_periods
- missed_opportunity_recheck

Targets:

- Return > 120%
- Profit Capture Ratio > 10%
- Max Drawdown < 25%
- Sharpe >= 0.80

Command:

```bash
python src/main.py backtest --symbol BTCUSDT --strategy regime_gated_trend_holding
```

Output:

- outputs/regime_gated_trend_holding_report.json

---

# Chart Output

Create PNG chart with:

- Panel 1: Candlesticks, EMA20, EMA50, EMA200
- Panel 2: RSI
- Panel 3: MACD
- Panel 4: Volume bars and volume moving average
- Support, resistance, and entry price lines on the price panel
- Multi-timeframe setup labels and alignment in the chart title

Save:

outputs/chart.png

---

# Project Structure

Phase 1.15 Portfolio Risk Governor

- Module: `src/risk/portfolio_risk_governor.py`
- Simulator: `src/backtesting/portfolio_governor_simulator.py`
- CLI: `python src/main.py backtest --strategy regime_gated_portfolio_governor`
- Report: `outputs/portfolio_risk_governor_report.json`

Risk states:

- `NORMAL`: drawdown < 10%, allocation 100%, runners enabled
- `CAUTION`: drawdown >= 10%, allocation 75%, runners enabled
- `DEFENSIVE`: drawdown >= 15%, allocation 50%, runners disabled
- `CAPITAL_PRESERVATION`: drawdown >= 20%, allocation 25%, runners disabled, no new Trend Holding runners

Sizing:

- Risk per trade is 1% of current equity
- Position size = risk amount / (entry price - stop price)
- If ATR > ATR_MA, size is multiplied by ATR_MA / ATR and clamped between 0.25 and 1.0

Portfolio stop:

- Triggers when drawdown > 25%
- Closes active runners
- Disables new runners
- Recovers only after drawdown falls below 15%

Report metrics:

- total_return_pct
- CAGR
- Sharpe
- Profit Factor
- Max Drawdown
- Win Rate
- Profit Capture Ratio
- risk_state_counts
- average_position_size
- average_runner_size
- portfolio_stop_count
- defensive_mode_hours

---

Phase 1.16 Cross-Asset Validation

- Module: `src/backtesting/cross_asset_validation.py`
- CLI: `python src/main.py backtest --strategy cross_asset_validation`
- Default assets: BTCUSDT, ETHUSDT, SOLUSDT, SPY, QQQ
- Optional assets: TQQQ, NVDA
- Report: `outputs/cross_asset_validation.json`

Rules:

- Use Agent Aggressive
- Do not tune per asset
- Do not change indicators, entry logic, exit logic, or position sizing
- Use identical multi-timeframe logic when all required OHLCV timeframes are available
- Equity support is market-data-only validation, not broker integration

Robustness Score:

- Positive return: 20 points
- Sharpe > 0.8: 20 points
- Max drawdown < 25%: 20 points
- Profit factor > 1: 20 points
- Win rate > 40%: 20 points

Report metrics:

- total_return_pct
- CAGR
- Sharpe Ratio
- Max Drawdown
- Profit Factor
- Win Rate
- Total Trades
- Profit Capture Ratio
- Asset class averages
- Failure analysis
- Recommended production assets

---

Phase 1.16B Equity Data Adapter Fix & Validation

- Adapter: `src/data/equity_data_adapter.py`
- CLI: `python src/main.py validate-equities`
- Reports: `outputs/equity_validation_report.json` and `outputs/data_provider_diagnostics.json`

Provider fallback chain:

- Yahoo Finance
- Stooq
- Alpha Vantage, optional with `ALPHA_VANTAGE_API_KEY`
- Twelve Data, optional with `TWELVE_DATA_API_KEY`

Validation rules:

- OHLCV exists
- No duplicate timestamps
- No missing timestamps
- Sorted ascending
- Minimum 3 years history
- 1h, 4h, and 1d timeframes available
- 4h is generated from 1h aggregation when a provider lacks native 4h

Strategy rules:

- Agent Aggressive only
- No strategy optimization
- No asset-specific tuning
- No indicator or exit changes

Success criteria:

- SPY data loads
- QQQ data loads
- SPY and QQQ backtests complete
- At least one equity has Sharpe > 0.8
- Both required equities have positive return

---

Phase 1.17 Exit Optimization Engine

- Module: `src/research/exit_optimization_engine.py`
- CLI: `python src/main.py backtest --strategy exit_optimization`
- Reports: `outputs/exit_optimization_report.json` and `outputs/exit_model_rankings.json`

Research scope:

- Use BTCUSDT from 2020-present
- Use Agent Aggressive baseline entries
- Keep entry logic unchanged
- Keep fixed stop loss logic unchanged
- Keep position sizing, risk filters, market regime, support/resistance, and signal generation unchanged
- Replay exit-only models against the same accepted entry schedule

Exit models:

- Baseline production exit
- EMA20 trend rider
- EMA20 / EMA50 cross exit
- ATR trailing exits at 2.0x, 2.5x, 3.0x, and 4.0x
- Chandelier exit using highest high(22) minus 3 ATR
- Partial profit plus EMA20 trend ride
- Multi-target exits
- Trend-strength exit using ADX and EMA alignment
- Volatility-adaptive ATR trailing exit
- Hybrid partial-profit, EMA20, and ATR trailing exits

Ranking:

- Profit capture ratio
- Sharpe ratio
- Total return
- Max drawdown
- Profit factor

Success criteria:

- Profit capture ratio > 30%
- Sharpe >= 0.80
- Profit factor > 1.40
- Max drawdown < 10%

---

src/

├── data/

│   ├── __init__.py

│   └── equity_data_adapter.py

├── backtesting/

│   ├── __init__.py

│   ├── backtest_engine.py

│   ├── benchmark_strategies.py

│   ├── benchmarks/

│   │   ├── __init__.py

│   │   ├── strategies.py

│   │   └── research.py

│   ├── trade_simulator.py

│   ├── trend_rider_simulator.py

│   ├── hybrid_trend_rider_simulator.py

│   ├── trend_holding_simulator.py

│   ├── regime_gated_trend_holding_simulator.py

│   ├── portfolio_governor_simulator.py

│   ├── cross_asset_validation.py

│   ├── performance_metrics.py

│   └── backtest_report.py

├── risk/

│   ├── __init__.py

│   ├── structure_stop_engine.py

│   └── portfolio_risk_governor.py

├── research/

│   ├── __init__.py

│   └── exit_optimization_engine.py

├── decision/

│   ├── __init__.py

│   └── decision_engine.py

├── scoring/

│   ├── __init__.py

│   ├── support_resistance_skill.py

│   ├── risk_reward_skill.py

│   ├── market_regime_skill.py

│   ├── setup_detection_skill.py

│   └── multi_timeframe_skill.py

└── trading_agent/

    ├── config.py

    ├── models.py

    ├── data.py

    ├── indicators.py

    ├── journal.py

    ├── scoring.py

    ├── decision.py

    ├── output.py

    └── main.py

---

# Unit Tests

Required:

tests/

├── test_config.py

├── test_backtest_engine.py

├── test_trade_simulator.py

├── test_performance_metrics.py

├── test_models.py

├── test_data.py

├── test_indicators.py

├── test_scoring.py

├── test_support_resistance_skill.py

├── test_risk_reward_skill.py

├── test_market_regime_skill.py

├── test_setup_detection.py

├── test_multi_timeframe_skill.py

├── test_journal.py

├── test_decision.py

├── test_decision_engine_v2.py

├── test_output.py

├── test_portfolio_risk_governor.py

├── test_cross_asset_validation.py

├── test_equity_data_adapter.py

├── test_exit_optimization_engine.py

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
