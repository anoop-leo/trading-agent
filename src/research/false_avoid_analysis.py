"""Phase 1.21 false-avoid and WATCH LONG candidate research."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FALSE_AVOID_REPORT = "false_avoid_analysis.json"
WATCH_LONG_REPORT = "watch_long_candidate_backtest.json"
ENRICHED_JOURNAL = "signal_journal_enriched.json"

FALSE_AVOID = "FALSE_AVOID"
CORRECT_AVOID = "CORRECT_AVOID"
INCONCLUSIVE = "INCONCLUSIVE"
WATCH_LONG = "WATCH LONG"


@dataclass(frozen=True)
class FalseAvoidAnalysisConfig:
    """Configuration for Phase 1.21 research-only analysis."""

    output_dir: Path = Path("outputs")
    false_avoid_gain_pct: float = 2.0
    false_avoid_prior_drop_pct: float = -1.5
    correct_avoid_drop_pct: float = -2.0
    meaningful_up_pct: float = 2.0
    max_future_days: int = 7
    confirmation_candles: int = 6


def run_false_avoid_analysis(config: FalseAvoidAnalysisConfig | None = None) -> dict[str, Any]:
    """Generate Phase 1.21 false-avoid research artifacts without changing strategy logic."""

    config = config or FalseAvoidAnalysisConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    price_history = load_shadow_price_history(output_dir / "shadow_equity_curve.csv")
    summary = _load_json_object(output_dir / "shadow_summary_30d.json")
    daily_reports = _load_daily_reports(output_dir)
    trades = _load_csv_rows(output_dir / "shadow_trades.csv")
    signals = collect_avoid_signals(output_dir)
    enriched = [enrich_avoid_signal(signal, price_history, config) for signal in signals]
    metrics = build_false_avoid_metrics(enriched)
    reason_counts = false_avoid_reason_counts(enriched)
    candidate = build_watch_long_candidate_backtest(enriched, price_history, summary, trades, daily_reports, config)
    recommendation = build_recommendation(metrics, reason_counts, candidate)

    false_avoid_report = {
        "phase": "1.21",
        "mode": "MISSED_ENTRY_FALSE_AVOID_ANALYSIS",
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "data_sources": {
            "signal_journal_v2": str(output_dir / "signal_journal_v2.json"),
            "signal_journal": str(output_dir / "signal_journal.json"),
            "shadow_signals_v2": str(output_dir / "shadow_signals_v2.csv"),
            "shadow_signals": str(output_dir / "shadow_signals.csv"),
            "shadow_daily_reports": str(output_dir / "shadow_daily_report_*.json"),
            "shadow_summary": str(output_dir / "shadow_summary_30d.json"),
            "shadow_trades": str(output_dir / "shadow_trades.csv"),
            "shadow_equity_curve": str(output_dir / "shadow_equity_curve.csv"),
        },
        "source_quality": source_quality(signals, price_history, summary, daily_reports),
        "metrics": metrics,
        "false_avoid_rejection_reason_counts": reason_counts,
        "watch_long_candidate_rule": watch_long_rule_definition(),
        "recommendation": recommendation,
        "artifacts": {
            "false_avoid_analysis": str(output_dir / FALSE_AVOID_REPORT),
            "watch_long_candidate_backtest": str(output_dir / WATCH_LONG_REPORT),
            "signal_journal_enriched": str(output_dir / ENRICHED_JOURNAL),
        },
    }

    _write_json(output_dir / ENRICHED_JOURNAL, {"signals": enriched})
    _write_json(output_dir / WATCH_LONG_REPORT, candidate)
    _write_json(output_dir / FALSE_AVOID_REPORT, false_avoid_report)
    return {
        "false_avoid_analysis": false_avoid_report,
        "watch_long_candidate_backtest": candidate,
        "signal_journal_enriched": {"signals": enriched},
        "artifacts": false_avoid_report["artifacts"],
    }


def collect_avoid_signals(output_dir: Path) -> list[dict[str, Any]]:
    """Load and dedupe AVOID LONG rows from signal journal and shadow signal logs."""

    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    journal_v2 = _load_json_any(output_dir / "signal_journal_v2.json", default={})
    journal_v2_rows = journal_v2.get("signals", []) if isinstance(journal_v2, dict) else []
    for row in journal_v2_rows if isinstance(journal_v2_rows, list) else []:
        if isinstance(row, dict) and _is_avoid_long(row):
            _merge_candidate(candidates, normalize_signal_row(row, "signal_journal_v2"))

    journal_payload = _load_json_any(output_dir / "signal_journal.json", default=[])
    journal_rows = journal_payload if isinstance(journal_payload, list) else journal_payload.get("signals", [])
    for row in journal_rows if isinstance(journal_rows, list) else []:
        if isinstance(row, dict) and _is_avoid_long(row):
            _merge_candidate(candidates, normalize_signal_row(row, "signal_journal"))

    for row in _load_csv_rows(output_dir / "shadow_signals_v2.csv"):
        if _is_avoid_long(row):
            _merge_candidate(candidates, normalize_signal_row(row, "shadow_signals_v2"))

    for row in _load_csv_rows(output_dir / "shadow_signals.csv"):
        if _is_avoid_long(row):
            _merge_candidate(candidates, normalize_signal_row(row, "shadow_signals"))

    return sorted(candidates.values(), key=lambda signal: str(signal.get("timestamp") or ""))


def enrich_avoid_signal(
    signal: dict[str, Any],
    price_history: list[dict[str, Any]],
    config: FalseAvoidAnalysisConfig,
) -> dict[str, Any]:
    """Attach forward returns, false-avoid classification, and WATCH LONG candidate status."""

    changes = calculate_forward_changes(signal, price_history, config)
    classification = classify_avoid(changes, config)
    signal = signal.copy()
    signal.update(changes)
    signal.update(classification)
    signal["ema_relationship"] = ema_relationship(signal)
    signal["rejection_categories"] = rejection_categories(signal)
    signal["watch_long_candidate"] = evaluate_watch_long_candidate(signal)
    signal["watch_long_confirmation"] = evaluate_watch_long_confirmation(signal, price_history, config)
    return signal


def calculate_forward_changes(
    signal: dict[str, Any],
    price_history: list[dict[str, Any]],
    config: FalseAvoidAnalysisConfig,
) -> dict[str, Any]:
    price = _optional_float(signal.get("price"))
    timestamp = _parse_timestamp(signal.get("timestamp"))
    evaluations = signal.get("evaluations") if isinstance(signal.get("evaluations"), dict) else {}
    changes: dict[str, Any] = {}
    point_changes: list[float] = []

    for days in (1, 3, 7):
        key = f"plus_{days}_day"
        value = _evaluation_change_pct(evaluations.get(key) if evaluations else None)
        if value is None:
            value = _history_change_pct(price_history, timestamp, price, days)
        changes[f"price_change_plus_{days}_day"] = value
        if value is not None:
            point_changes.append(value)

    history_window = _history_window_changes(price_history, timestamp, price, config.max_future_days)
    all_changes = [*point_changes, *history_window]
    three_day_changes = [
        pct
        for pct in [
            changes["price_change_plus_1_day"],
            changes["price_change_plus_3_day"],
            *_history_window_changes(price_history, timestamp, price, 3),
        ]
        if pct is not None
    ]
    changes["max_favorable_move_after_signal"] = max(all_changes) if all_changes else None
    changes["max_adverse_move_after_signal"] = min(all_changes) if all_changes else None
    changes["max_favorable_move_3d"] = max(three_day_changes) if three_day_changes else None
    changes["max_adverse_move_3d"] = min(three_day_changes) if three_day_changes else None
    changes["has_3d_forward_evidence"] = changes["price_change_plus_3_day"] is not None
    changes["future_path_points"] = len(history_window)
    changes["three_day_path_points"] = len(three_day_changes)
    return changes


def classify_avoid(changes: dict[str, Any], config: FalseAvoidAnalysisConfig) -> dict[str, Any]:
    """Classify one AVOID LONG signal using Phase 1.21 rules."""

    max_up = _optional_float(changes.get("max_favorable_move_3d"))
    max_down = _optional_float(changes.get("max_adverse_move_3d"))
    points = int(changes.get("three_day_path_points") or 0)
    has_3d_forward_evidence = bool(changes.get("has_3d_forward_evidence"))
    if max_up is None:
        return {
            "avoid_classification": INCONCLUSIVE,
            "classification_reason": "No 1-3 day forward price evidence is available yet.",
            "classification_confidence": "LOW",
        }
    if max_up > config.false_avoid_gain_pct and (max_down is None or max_down > config.false_avoid_prior_drop_pct):
        return {
            "avoid_classification": FALSE_AVOID,
            "classification_reason": (
                "Price moved up more than 2% within 3 days without evidence of a prior drop greater than 1.5%."
            ),
            "classification_confidence": "HIGH" if points >= 6 else "LOW",
        }
    if max_down is not None and max_down < config.correct_avoid_drop_pct:
        return {
            "avoid_classification": CORRECT_AVOID,
            "classification_reason": "Price dropped more than 2% within 3 days after the avoid signal.",
            "classification_confidence": "HIGH" if points >= 6 else "LOW",
        }
    if not has_3d_forward_evidence:
        return {
            "avoid_classification": INCONCLUSIVE,
            "classification_reason": "The full 3-day window has not elapsed and no threshold was hit yet.",
            "classification_confidence": "LOW",
        }
    if max_up <= config.meaningful_up_pct:
        return {
            "avoid_classification": CORRECT_AVOID,
            "classification_reason": "Price did not move up meaningfully within 3 days.",
            "classification_confidence": "MEDIUM" if points >= 2 else "LOW",
        }
    return {
        "avoid_classification": INCONCLUSIVE,
        "classification_reason": "Forward move was mixed or lacked enough path data.",
        "classification_confidence": "LOW",
    }


def evaluate_watch_long_candidate(signal: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the proposed WATCH LONG trigger without changing the production decision."""

    setup = str(signal.get("setup") or "").upper()
    price = _optional_float(signal.get("price"))
    ema20 = _optional_float(signal.get("ema20"))
    ema50 = _optional_float(signal.get("ema50"))
    rsi = _optional_float(signal.get("rsi"))
    support = _optional_float(signal.get("support"))
    resistance = _optional_float(signal.get("resistance"))
    market_regime = str(signal.get("market_regime") or "").upper()
    daily_setup = str(signal.get("daily_setup") or "").upper()
    daily_alignment = str(signal.get("daily_alignment") or "").upper()
    four_hour_macd = str(signal.get("four_hour_macd") or signal.get("macd") or "").upper()

    criteria = {
        "setup_is_trend_or_range": setup in {"TREND_FOLLOWING", "RANGE_BOUND"},
        "price_above_1h_ema20_and_ema50": _both_present(price, ema20, ema50) and price > ema20 and price > ema50,
        "four_hour_macd_improving_or_bullish": four_hour_macd in {"BULLISH", "IMPROVING"},
        "rsi_above_45": rsi is not None and rsi > 45,
        "near_support_or_broke_resistance": _near_support_or_broke_resistance(price, support, resistance),
        "daily_trend_not_strongly_bearish": not (
            market_regime == "BEAR" or daily_setup == "BEAR_TREND" or daily_alignment == "BEARISH_ALIGNMENT"
        ),
    }
    missing_fields = _missing_watch_long_fields(signal)
    triggered = all(criteria.values()) and not missing_fields
    return {
        "candidate_state": WATCH_LONG if triggered else None,
        "triggered": triggered,
        "criteria": criteria,
        "missing_fields": missing_fields,
        "blocked_by": [name for name, passed in criteria.items() if not passed],
    }


def evaluate_watch_long_confirmation(
    signal: dict[str, Any],
    price_history: list[dict[str, Any]],
    config: FalseAvoidAnalysisConfig,
) -> dict[str, Any]:
    candidate = signal.get("watch_long_candidate") if isinstance(signal.get("watch_long_candidate"), dict) else {}
    if not candidate.get("triggered"):
        return {
            "confirmed": False,
            "reason": "WATCH LONG trigger did not pass.",
            "missing_fields": [],
        }

    price = _optional_float(signal.get("price"))
    timestamp = _parse_timestamp(signal.get("timestamp"))
    previous_swing_high = _optional_float(signal.get("previous_swing_high") or signal.get("recent_swing_high"))
    volume_ratio = _optional_float(signal.get("volume_ratio"))
    macd = str(signal.get("four_hour_macd") or signal.get("macd") or "").upper()
    rsi_rising = _truthy(signal.get("rsi_rising")) or (_optional_float(signal.get("rsi_delta")) or 0.0) > 0
    ema20 = _optional_float(signal.get("ema20"))
    next_prices = _next_prices(price_history, timestamp, config.confirmation_candles)

    missing_fields = []
    if previous_swing_high is None:
        missing_fields.append("previous_swing_high")
    if volume_ratio is None:
        missing_fields.append("volume_ratio")
    if macd not in {"BULLISH", "IMPROVING"}:
        missing_fields.append("macd_bullish_or_improving")
    if not rsi_rising:
        missing_fields.append("rsi_rising")
    if price is None or ema20 is None:
        missing_fields.append("price_or_ema20")
    if not next_prices:
        missing_fields.append("next_1_to_6_candles")

    closes_above_swing = (
        previous_swing_high is not None and any(float(row["price"]) > previous_swing_high for row in next_prices)
    )
    remains_above_ema20 = price is not None and ema20 is not None and min([price, *[float(row["price"]) for row in next_prices]]) > ema20
    confirmed = (
        not missing_fields
        and closes_above_swing
        and volume_ratio is not None
        and volume_ratio >= 1.0
        and macd in {"BULLISH", "IMPROVING"}
        and rsi_rising
        and remains_above_ema20
    )
    return {
        "confirmed": confirmed,
        "reason": "Confirmation passed within 1-6 candles." if confirmed else "Confirmation evidence is incomplete or failed.",
        "missing_fields": missing_fields,
        "criteria": {
            "close_above_previous_swing_high": closes_above_swing,
            "volume_ratio_gte_1": volume_ratio is not None and volume_ratio >= 1.0,
            "macd_bullish_or_improving": macd in {"BULLISH", "IMPROVING"},
            "rsi_rising": rsi_rising,
            "price_remains_above_ema20": remains_above_ema20,
        },
    }


def build_false_avoid_metrics(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(enriched)
    false_count = sum(1 for row in enriched if row.get("avoid_classification") == FALSE_AVOID)
    correct_count = sum(1 for row in enriched if row.get("avoid_classification") == CORRECT_AVOID)
    inconclusive_count = sum(1 for row in enriched if row.get("avoid_classification") == INCONCLUSIVE)
    classified = false_count + correct_count
    return {
        "total_avoid_signals": total,
        "false_avoid_count": false_count,
        "correct_avoid_count": correct_count,
        "inconclusive_count": inconclusive_count,
        "false_avoid_rate": _pct(false_count / classified) if classified else None,
        "average_missed_gain_1d": _mean(_values(enriched, "price_change_plus_1_day")),
        "average_missed_gain_3d": _mean(_values(enriched, "price_change_plus_3_day")),
        "average_missed_gain_7d": _mean(_values(enriched, "price_change_plus_7_day")),
        "max_missed_gain": _max(_values(enriched, "max_favorable_move_after_signal")),
        "watch_long_trigger_count": sum(1 for row in enriched if row["watch_long_candidate"]["triggered"]),
        "watch_long_confirmation_count": sum(1 for row in enriched if row["watch_long_confirmation"]["confirmed"]),
    }


def false_avoid_reason_counts(enriched: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in enriched:
        if row.get("avoid_classification") == FALSE_AVOID:
            counter.update(row.get("rejection_categories") or ["unknown_rejection_reason"])
    return dict(counter.most_common())


def build_watch_long_candidate_backtest(
    enriched: list[dict[str, Any]],
    price_history: list[dict[str, Any]],
    summary: dict[str, Any],
    trades: list[dict[str, Any]],
    daily_reports: list[dict[str, Any]],
    config: FalseAvoidAnalysisConfig,
) -> dict[str, Any]:
    confirmed = [row for row in enriched if row["watch_long_confirmation"]["confirmed"]]
    false_confirmed = [row for row in confirmed if row.get("avoid_classification") == FALSE_AVOID]
    baseline = {
        "total_trades": summary.get("total_trades", len(trades)),
        "win_rate": summary.get("win_rate"),
        "profit_factor": summary.get("profit_factor"),
        "sharpe_ratio": summary.get("sharpe_ratio"),
        "max_drawdown": summary.get("max_drawdown_pct"),
        "total_return": summary.get("total_return_pct"),
        "average_holding_time": summary.get("average_holding_time"),
        "fee_drag": None,
        "rejected_signals": summary.get("rejected_signal_count") or _sum_key(daily_reports, "rejected_signals"),
    }
    false_avoid_count = sum(1 for row in enriched if row.get("avoid_classification") == FALSE_AVOID)
    missing_reasons = Counter(
        missing
        for row in enriched
        for missing in [
            *row["watch_long_candidate"].get("missing_fields", []),
            *row["watch_long_confirmation"].get("missing_fields", []),
        ]
    )
    comparison_possible = bool(price_history) and len(price_history) >= config.confirmation_candles and bool(confirmed)
    return {
        "phase": "1.21",
        "mode": "WATCH_LONG_CANDIDATE_BACKTEST",
        "status": "RESEARCH_ONLY_SIGNAL_RECLASSIFICATION" if comparison_possible else "INSUFFICIENT_DATA",
        "live_trading_enabled": False,
        "order_endpoint_calls_allowed": False,
        "baseline": baseline,
        "candidate": {
            "candidate_state": WATCH_LONG,
            "watch_long_triggers": sum(1 for row in enriched if row["watch_long_candidate"]["triggered"]),
            "confirmed_watch_longs": len(confirmed),
            "false_avoids_converted_to_trades": len(false_confirmed),
            "missed_opportunity_reduction": _pct(len(false_confirmed) / false_avoid_count) if false_avoid_count else 0.0,
            "total_trades": None,
            "win_rate": None,
            "profit_factor": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "total_return": None,
            "average_holding_time": None,
            "fee_drag": None,
            "rejected_signals": None,
        },
        "comparison": {
            "candidate_better_than_baseline": None,
            "reason": (
                "Current artifacts do not include enough signal-level confirmation candles and exit paths "
                "to compute trade PnL metrics for WATCH LONG."
            )
            if not comparison_possible
            else "WATCH LONG confirmations were identified, but production strategy logic was not changed.",
        },
        "missing_data_counts": dict(missing_reasons.most_common()),
        "rule": watch_long_rule_definition(),
    }


def build_recommendation(
    metrics: dict[str, Any],
    reason_counts: dict[str, int],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    false_rate = metrics.get("false_avoid_rate")
    sample_size = metrics.get("total_avoid_signals", 0)
    classified = int(metrics.get("false_avoid_count", 0) or 0) + int(metrics.get("correct_avoid_count", 0) or 0)
    status = candidate.get("status")
    if false_rate is None or classified < 3:
        too_defensive = "INCONCLUSIVE"
    elif false_rate >= 30 and (sample_size >= 20 or metrics.get("inconclusive_count", 0) == 0):
        too_defensive = "YES"
    elif false_rate >= 30:
        too_defensive = "LIKELY_YES_NEEDS_MORE_SHADOW_DATA"
    elif sample_size < 20:
        too_defensive = "INCONCLUSIVE"
    else:
        too_defensive = "NO"
    return {
        "is_strategy_too_defensive": too_defensive,
        "most_common_false_avoid_driver": _first_key(reason_counts),
        "should_watch_long_be_added": "COLLECT_MORE_SHADOW_DATA" if status == "INSUFFICIENT_DATA" else "RESEARCH_CANDIDATE_ONLY",
        "should_any_avoid_long_rule_be_relaxed": "NO_NOT_FROM_CURRENT_SAMPLE",
        "is_candidate_better_than_baseline": "INCONCLUSIVE" if status == "INSUFFICIENT_DATA" else "NOT_PROVEN",
        "live_trading_approval": "NOT_APPROVED",
    }


def source_quality(
    signals: list[dict[str, Any]],
    price_history: list[dict[str, Any]],
    summary: dict[str, Any],
    daily_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    signal_sources = Counter(str(signal.get("source")) for signal in signals)
    return {
        "avoid_signal_rows": len(signals),
        "signal_sources": dict(signal_sources),
        "shadow_price_points": len(price_history),
        "shadow_signal_count": summary.get("signal_count"),
        "shadow_rejected_signal_count": summary.get("rejected_signal_count"),
        "shadow_daily_report_count": len(daily_reports),
        "limitations": [
            "Shadow smoke-run artifacts have aggregate rejected counts but limited forward price path.",
            "Older signal_journal rows have forward point evaluations but not full indicator snapshots.",
        ],
    }


def watch_long_rule_definition() -> dict[str, Any]:
    return {
        "candidate_state": WATCH_LONG,
        "trigger": [
            "setup is TREND_FOLLOWING or RANGE_BOUND",
            "price is above 1H EMA20 and EMA50",
            "4H MACD is improving or bullish",
            "RSI is above 45",
            "price is within 3% of support or has broken resistance",
            "daily trend is not strongly bearish",
        ],
        "confirmation_window_candles": "1 to 6",
        "confirmation": [
            "candle closes above previous swing high",
            "volume_ratio >= 1.0",
            "MACD bullish or improving",
            "RSI rising",
            "price remains above EMA20",
        ],
        "production_strategy_changed": False,
    }


def normalize_signal_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    evaluations = row.get("evaluations") if isinstance(row.get("evaluations"), dict) else {}
    rejection_reasons = _parse_rejection_reasons(row.get("rejection_reasons") or row.get("rejected_entry_reasons"))
    return {
        "id": row.get("id"),
        "source": source,
        "timestamp": row.get("timestamp") or row.get("signal_timestamp"),
        "symbol": row.get("symbol"),
        "setup": row.get("setup"),
        "final_decision": row.get("final_decision") or row.get("action"),
        "price": _optional_float(row.get("price") or row.get("signal_price")),
        "market_regime": row.get("market_regime"),
        "rsi": _optional_float(row.get("rsi")),
        "macd": row.get("macd"),
        "ema20": _optional_float(row.get("ema20") or row.get("ema_20")),
        "ema50": _optional_float(row.get("ema50") or row.get("ema_50")),
        "ema200": _optional_float(row.get("ema200") or row.get("ema_200")),
        "daily_alignment": row.get("daily_alignment"),
        "four_hour_alignment": row.get("four_hour_alignment"),
        "one_hour_alignment": row.get("one_hour_alignment"),
        "multi_timeframe_alignment": row.get("multi_timeframe_alignment") or row.get("alignment"),
        "daily_setup": row.get("daily_setup"),
        "four_hour_macd": row.get("four_hour_macd"),
        "support": _optional_float(row.get("support")),
        "resistance": _optional_float(row.get("resistance")),
        "rr_ratio": _optional_float(row.get("rr_ratio")),
        "volume_ratio": _optional_float(row.get("volume_ratio")),
        "trend_score": _optional_float(row.get("trend_score")),
        "rejection_reasons": rejection_reasons,
        "evaluations": evaluations,
    }


def load_shadow_price_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _load_csv_rows(path):
        timestamp = _parse_timestamp(row.get("timestamp"))
        price = _optional_float(row.get("price"))
        if timestamp is not None and price is not None:
            rows.append({"timestamp": timestamp, "price": price})
    return sorted(rows, key=lambda row: row["timestamp"])


def ema_relationship(signal: dict[str, Any]) -> str | None:
    price = _optional_float(signal.get("price"))
    emas = {
        "EMA20": _optional_float(signal.get("ema20")),
        "EMA50": _optional_float(signal.get("ema50")),
        "EMA200": _optional_float(signal.get("ema200")),
    }
    if price is None or any(value is None for value in emas.values()):
        return None
    above = [name for name, value in emas.items() if value is not None and price > value]
    below = [name for name, value in emas.items() if value is not None and price <= value]
    return f"above {', '.join(above)}; below {', '.join(below)}"


def rejection_categories(signal: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    setup = str(signal.get("setup") or "").upper()
    if setup:
        categories.append(setup)
    categories.extend(str(reason) for reason in signal.get("rejection_reasons") or [])

    volume_ratio = _optional_float(signal.get("volume_ratio"))
    alignment = str(signal.get("multi_timeframe_alignment") or "").upper()
    price = _optional_float(signal.get("price"))
    ema200 = _optional_float(signal.get("ema200"))
    macd = str(signal.get("macd") or signal.get("four_hour_macd") or "").upper()
    rsi = _optional_float(signal.get("rsi"))
    rr_ratio = _optional_float(signal.get("rr_ratio"))

    if volume_ratio is not None and volume_ratio < 1.0:
        categories.append("low volume")
    if alignment and alignment not in {"BULLISH_ALIGNMENT", "PULLBACK_IN_UPTREND"}:
        categories.append("weak multi-timeframe alignment")
    if price is not None and ema200 is not None and price < ema200:
        categories.append("below EMA200")
    if macd == "BEARISH":
        categories.append("MACD bearish")
    if rsi is not None and 45 <= rsi <= 55:
        categories.append("RSI neutral")
    if rr_ratio is not None and rr_ratio < 2.0:
        categories.append("risk/reward too low")
    return categories or ["unknown_rejection_reason"]


def _merge_candidate(candidates: dict[tuple[Any, ...], dict[str, Any]], signal: dict[str, Any]) -> None:
    key = (signal.get("timestamp"), signal.get("symbol"), signal.get("setup"), signal.get("final_decision"))
    existing = candidates.get(key)
    if existing is None:
        candidates[key] = signal
        return
    merged = existing.copy()
    for field, value in signal.items():
        if value not in (None, "", [], {}):
            if field == "evaluations" and existing.get("evaluations"):
                continue
            merged[field] = value
    candidates[key] = merged


def _is_avoid_long(row: dict[str, Any]) -> bool:
    decision = str(row.get("final_decision") or row.get("decision") or row.get("action") or "").upper()
    return decision in {"AVOID LONG", "AVOID"}


def _evaluation_change_pct(evaluation: Any) -> float | None:
    if not isinstance(evaluation, dict):
        return None
    return _optional_float(evaluation.get("price_change_pct"))


def _history_change_pct(
    price_history: list[dict[str, Any]],
    timestamp: pd.Timestamp | None,
    price: float | None,
    days: int,
) -> float | None:
    if timestamp is None or price is None or price <= 0:
        return None
    target = timestamp + pd.Timedelta(days=days)
    future = [row for row in price_history if row["timestamp"] >= target]
    if not future:
        return None
    return _pct((float(future[0]["price"]) / price) - 1)


def _history_window_changes(
    price_history: list[dict[str, Any]],
    timestamp: pd.Timestamp | None,
    price: float | None,
    days: int,
) -> list[float]:
    if timestamp is None or price is None or price <= 0:
        return []
    end = timestamp + pd.Timedelta(days=days)
    return [
        _pct((float(row["price"]) / price) - 1)
        for row in price_history
        if timestamp < row["timestamp"] <= end
    ]


def _next_prices(price_history: list[dict[str, Any]], timestamp: pd.Timestamp | None, candles: int) -> list[dict[str, Any]]:
    if timestamp is None:
        return []
    return [row for row in price_history if row["timestamp"] > timestamp][:candles]


def _near_support_or_broke_resistance(price: float | None, support: float | None, resistance: float | None) -> bool:
    if price is None or price <= 0:
        return False
    near_support = support is not None and support > 0 and abs(price - support) / price <= 0.03
    broke_resistance = resistance is not None and price > resistance
    return near_support or broke_resistance


def _missing_watch_long_fields(signal: dict[str, Any]) -> list[str]:
    required = ("setup", "price", "ema20", "ema50", "rsi", "four_hour_macd")
    missing = [field for field in required if signal.get(field) in (None, "")]
    if signal.get("support") in (None, "") and signal.get("resistance") in (None, ""):
        missing.append("support_or_resistance")
    return missing


def _both_present(*values: float | None) -> bool:
    return all(value is not None for value in values)


def _parse_rejection_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, dict):
        return [str(key) for key, count in value.items() if count]
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", "|").split("|") if part.strip()]
    return []


def _load_json_any(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json_any(path, default={})
    return payload if isinstance(payload, dict) else {}


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_daily_reports(output_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(output_dir.glob("shadow_daily_report_*.json")):
        reports.append(_load_json_object(path))
    return reports


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _pct(value: float) -> float:
    return round(value * 100.0, 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _max(values: list[float]) -> float | None:
    return round(max(values), 4) if values else None


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for row in rows if (value := _optional_float(row.get(key))) is not None]


def _sum_key(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(_optional_float(row.get(key)) or 0 for row in rows))


def _first_key(values: dict[str, Any]) -> str | None:
    return next(iter(values), None)
