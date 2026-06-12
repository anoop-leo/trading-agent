"""JSON and chart output helpers for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from decision.decision_engine import DecisionResult, FinalDecisionResult
from scoring.market_regime_skill import MarketRegimeResult
from scoring.multi_timeframe_skill import MultiTimeframeResult, TimeframeSignal
from scoring.risk_reward_skill import RiskRewardResult
from scoring.setup_detection_skill import SetupResult
from scoring.support_resistance_skill import SupportResistanceResult, calculate_support_resistance
from trading_agent.config import AgentConfig
from trading_agent.models import SignalScores
from trading_agent.scoring import calculate_recent_swing_high, calculate_recent_swing_low, calculate_volume_ratio


class OutputError(RuntimeError):
    """Raised when Phase 1 artifacts cannot be written."""


REQUIRED_SIGNAL_COLUMNS = (
    "timestamp",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "high",
    "low",
    "volume",
    "volume_ma_20",
)


def _format_number(value: object, decimals: int = 2) -> int | float:
    if value is None or pd.isna(value):
        raise OutputError("Cannot build signal output from missing indicator values.")
    rounded = round(float(value), decimals)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _latest_signal_row(indicator_frame: pd.DataFrame) -> pd.Series:
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in indicator_frame.columns]
    if missing:
        raise OutputError(f"Cannot build signal output missing columns: {', '.join(missing)}.")
    if indicator_frame.empty:
        raise OutputError("Cannot build signal output from an empty frame.")
    return indicator_frame.iloc[-1]


def macd_direction(row: pd.Series) -> str:
    """Convert latest MACD values into a compact JSON label."""

    if row["macd"] > row["macd_signal"]:
        return "bullish"
    if row["macd"] < row["macd_signal"]:
        return "bearish"
    return "neutral"


def _format_optional_number(value: float | None, decimals: int = 2) -> int | float | None:
    if value is None:
        return None
    return _format_number(value, decimals)


def _entry_zone_to_json(decision: DecisionResult) -> dict[str, int | float] | None:
    if decision.entry_zone is None:
        return None
    return {
        "low": _format_number(decision.entry_zone.low),
        "high": _format_number(decision.entry_zone.high),
    }


def _timeframe_signal_to_json(signal: TimeframeSignal) -> dict[str, Any]:
    return {
        "setup": signal.setup,
        "decision": signal.decision,
        "trend_score": signal.trend_score,
        "momentum_score": signal.momentum_score,
        "volume_score": signal.volume_score,
        "bottom_score": signal.bottom_score,
        "sr_score": signal.sr_score,
        "rr_score": signal.rr_score,
        "regime_score": signal.regime_score,
        "setup_confidence": signal.setup_confidence,
        "price": _format_number(signal.price),
        "rsi": _format_number(signal.rsi),
        "macd": signal.macd,
        "ema20": _format_number(signal.ema20),
        "ema50": _format_number(signal.ema50),
        "ema200": _format_number(signal.ema200),
        "market_regime": signal.market_regime,
    }


def _multi_timeframe_to_json(multi_timeframe: MultiTimeframeResult) -> dict[str, Any]:
    return {
        "alignment": multi_timeframe.alignment.value,
        "alignment_score": multi_timeframe.alignment_score,
        "summary": multi_timeframe.summary,
        "timeframes": {
            timeframe: _timeframe_signal_to_json(signal)
            for timeframe, signal in multi_timeframe.timeframes.items()
        },
    }


def build_output_payload(
    config: AgentConfig,
    indicator_frame: pd.DataFrame,
    scores: SignalScores,
    decision: DecisionResult,
    support_resistance: SupportResistanceResult,
    risk_reward: RiskRewardResult,
    market_regime: MarketRegimeResult,
    setup: SetupResult,
    multi_timeframe: MultiTimeframeResult | None = None,
    final_decision: FinalDecisionResult | None = None,
) -> dict[str, Any]:
    """Build the JSON-compatible Phase 1 signal payload."""

    row = _latest_signal_row(indicator_frame)
    payload = {
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "symbol": config.symbol,
        "market_data_source": config.resolved_market_data_source,
        "position_mode": config.position_mode,
        "price": _format_number(row["close"]),
        "ema20": _format_number(row["ema_20"]),
        "ema50": _format_number(row["ema_50"]),
        "ema200": _format_number(row["ema_200"]),
        "rsi": _format_number(row["rsi_14"]),
        "macd": macd_direction(row),
        "volume_ratio": _format_number(calculate_volume_ratio(row)),
        "trend_score": scores.trend_score,
        "momentum_score": scores.momentum_score,
        "volume_score": scores.volume_score,
        "bottom_score": scores.bottom_score,
        "support": _format_number(support_resistance.support),
        "resistance": _format_number(support_resistance.resistance),
        "distance_to_support": _format_number(support_resistance.distance_to_support),
        "distance_to_resistance": _format_number(support_resistance.distance_to_resistance),
        "sr_score": scores.sr_score,
        "risk": _format_number(risk_reward.risk),
        "reward": _format_number(risk_reward.reward),
        "rr_ratio": _format_number(risk_reward.rr_ratio),
        "rr_score": scores.rr_score,
        "market_regime": market_regime.market_regime.value,
        "regime_score": scores.regime_score,
        "setup": setup.setup.value,
        "setup_score": setup.setup_score,
        "setup_confidence": setup.setup_confidence,
        "setup_reason": setup.setup_reason,
        "recent_swing_high": _format_number(calculate_recent_swing_high(indicator_frame)),
        "recent_swing_low": _format_number(calculate_recent_swing_low(indicator_frame)),
        "decision": decision.decision.value,
        "decision_meaning": decision.decision_meaning,
        "confidence": decision.confidence,
        "entry_zone": _entry_zone_to_json(decision),
        "stop_loss": _format_optional_number(decision.stop_loss),
        "target_1": _format_optional_number(decision.target_1),
        "target_2": _format_optional_number(decision.target_2),
        "rationale": decision.rationale,
    }
    if multi_timeframe is not None:
        payload["multi_timeframe"] = _multi_timeframe_to_json(multi_timeframe)
    if final_decision is not None:
        payload["final_decision"] = final_decision.decision.value
        payload["final_decision_reason"] = final_decision.reason
    return payload


def write_json(payload: dict[str, Any], output_dir: Path, filename: str = "output.json") -> Path:
    """Persist the signal payload as formatted JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_chart(
    indicator_frame: pd.DataFrame,
    output_dir: Path,
    symbol: str | None = None,
    setup_label: str | None = None,
    filename: str = "chart.png",
) -> Path:
    """Persist a multi-panel technical chart."""

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise OutputError("matplotlib is required to write chart.png.") from exc

    required = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "volume",
        "volume_ma_20",
    )
    missing = [column for column in required if column not in indicator_frame.columns]
    if missing:
        raise OutputError(f"Cannot chart frame missing columns: {', '.join(missing)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_frame = indicator_frame.tail(120).copy()
    chart_frame["timestamp"] = pd.to_datetime(chart_frame["timestamp"])
    chart_frame["x"] = mdates.date2num(chart_frame["timestamp"].dt.to_pydatetime())

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(13, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2, 1.4, 1.4]},
    )
    price_ax, rsi_ax, macd_ax, volume_ax = axes
    candle_width = 0.025

    for row in chart_frame.itertuples(index=False):
        color = "#1f8a70" if row.close >= row.open else "#c44536"
        price_ax.vlines(row.x, row.low, row.high, color=color, linewidth=1)
        body_low = min(row.open, row.close)
        body_height = abs(row.close - row.open)
        if body_height == 0:
            body_height = max(row.high - row.low, 1e-9) * 0.01
        price_ax.add_patch(
            Rectangle(
                (row.x - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
            )
        )

    price_ax.plot(chart_frame["x"], chart_frame["ema_20"], label="EMA20", linewidth=1.1, color="#2f80ed")
    price_ax.plot(chart_frame["x"], chart_frame["ema_50"], label="EMA50", linewidth=1.1, color="#9b51e0")
    price_ax.plot(chart_frame["x"], chart_frame["ema_200"], label="EMA200", linewidth=1.1, color="#f2994a")
    current_price = float(chart_frame.iloc[-1]["close"])
    support_resistance = calculate_support_resistance(indicator_frame, current_price)
    price_ax.axhline(
        support_resistance.support,
        label="Support",
        color="#1f8a70",
        linestyle="--",
        linewidth=1.0,
    )
    price_ax.axhline(
        support_resistance.resistance,
        label="Resistance",
        color="#c44536",
        linestyle="--",
        linewidth=1.0,
    )
    price_ax.axhline(current_price, label="Entry", color="#111827", linestyle=":", linewidth=1.0)
    title = symbol or "Phase 1 Technical Signal"
    if setup_label is not None:
        prefix = "" if "\n" in setup_label or ":" in setup_label else "Setup: "
        title = f"{title}\n{prefix}{setup_label}"
    price_ax.set_title(title)
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.2)
    price_ax.legend(loc="best")

    rsi_ax.plot(chart_frame["x"], chart_frame["rsi_14"], label="RSI14", linewidth=1.1, color="#1b998b")
    rsi_ax.axhline(70, color="#c44536", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.axhline(50, color="#6c757d", linestyle="--", linewidth=0.8, alpha=0.5)
    rsi_ax.axhline(30, color="#2f80ed", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_ylabel("RSI")
    rsi_ax.grid(True, alpha=0.2)
    rsi_ax.legend(loc="best")

    histogram_colors = ["#1f8a70" if value >= 0 else "#c44536" for value in chart_frame["macd_histogram"]]
    macd_ax.bar(chart_frame["x"], chart_frame["macd_histogram"], width=candle_width, color=histogram_colors, alpha=0.5)
    macd_ax.plot(chart_frame["x"], chart_frame["macd"], label="MACD", linewidth=1.1, color="#2f80ed")
    macd_ax.plot(chart_frame["x"], chart_frame["macd_signal"], label="Signal", linewidth=1.1, color="#f2994a")
    macd_ax.axhline(0, color="#6c757d", linewidth=0.8, alpha=0.6)
    macd_ax.set_ylabel("MACD")
    macd_ax.grid(True, alpha=0.2)
    macd_ax.legend(loc="best")

    volume_colors = ["#1f8a70" if row.close >= row.open else "#c44536" for row in chart_frame.itertuples(index=False)]
    volume_ax.bar(chart_frame["x"], chart_frame["volume"], width=candle_width, color=volume_colors, alpha=0.45, label="Volume")
    volume_ax.plot(chart_frame["x"], chart_frame["volume_ma_20"], label="Volume MA20", linewidth=1.1, color="#9b51e0")
    volume_ax.set_ylabel("Volume")
    volume_ax.set_xlabel("Time")
    volume_ax.grid(True, alpha=0.2)
    volume_ax.legend(loc="best")

    volume_ax.xaxis_date()
    volume_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
