import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from monitoring.monitoring_config import MonitoringConfig
from monitoring.signal_alerts import (
    format_signal_alert,
    is_actionable,
    run_signal_alerts_job,
)


def _payload(symbol="BTCUSDT", entry="BUY", gate=None, trend="BULLISH", summary=None, reason="ok") -> dict:
    return {
        "symbol": symbol,
        "pre_gate_decision": "BUY",
        "final_decision": "BUY" if entry == "BUY" else "WAIT",
        "entry_decision": entry,
        "gate_triggered": gate,
        "trend_bias": trend,
        "alert_summary": summary,
        "watch_levels": None,
        "final_decision_reason": reason,
    }


class ActionableTests(unittest.TestCase):
    def test_buy_is_actionable(self) -> None:
        self.assertTrue(is_actionable(_payload(entry="BUY")))

    def test_wait_for_pullback_is_actionable(self) -> None:
        self.assertTrue(is_actionable(_payload(entry="WAIT_FOR_PULLBACK")))

    def test_gated_wait_is_actionable(self) -> None:
        self.assertTrue(is_actionable(_payload(entry="WAIT", gate="MISSING_ENTRY_SCORE")))

    def test_plain_wait_without_gate_not_actionable(self) -> None:
        self.assertFalse(is_actionable(_payload(entry="WAIT", gate=None)))

    def test_avoid_not_actionable(self) -> None:
        self.assertFalse(is_actionable(_payload(entry="AVOID LONG", gate=None)))


class FormatTests(unittest.TestCase):
    def test_uses_alert_summary_when_present(self) -> None:
        text = format_signal_alert(_payload(entry="WAIT_FOR_PULLBACK", gate="RR_BELOW_1R",
                                            summary="BTCUSDT — BULLISH trend, entry WAIT_FOR_PULLBACK. Watch pullback ..."))
        self.assertIn("Watch pullback", text)
        self.assertIn("(gate: RR_BELOW_1R)", text)

    def test_falls_back_to_reason(self) -> None:
        text = format_signal_alert(_payload(entry="BUY", reason="clean entry"))
        self.assertIn("BTCUSDT", text)
        self.assertIn("clean entry", text)


class RunJobTests(unittest.TestCase):
    def _run(self, base: Path, signal_fn, alert_state="alert_state.json"):
        config_path = base / "mc.json"
        config_path.write_text(json.dumps(MonitoringConfig(
            signal_alert_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")
        ).to_dict()) + "\n")
        return run_signal_alerts_job(
            monitoring_config_path=config_path,
            alert_state_path=base / alert_state,
            signal_alerts_path=base / "signal_alerts.json",
            signal_fn=signal_fn,
            sleep_fn=lambda _s: None,
            send_alerts=False,
        )

    def test_alerts_only_for_actionable_symbols(self) -> None:
        def fn(symbol):
            if symbol == "BTCUSDT":
                return _payload("BTCUSDT", entry="BUY")
            if symbol == "ETHUSDT":
                return _payload("ETHUSDT", entry="WAIT", gate="RR_BELOW_1R")
            return _payload("SOLUSDT", entry="AVOID LONG", gate=None)  # not actionable

        with TemporaryDirectory() as d:
            result = self._run(Path(d), fn)
        alerted = " ".join(result["alerts"])
        self.assertIn("BTCUSDT", alerted)
        self.assertIn("ETHUSDT", alerted)
        self.assertNotIn("SOLUSDT", alerted)
        self.assertEqual(set(result["signals"]), {"BTCUSDT", "ETHUSDT", "SOLUSDT"})

    def test_dedup_unchanged_state_does_not_refire(self) -> None:
        fn = lambda symbol: _payload(symbol, entry="BUY")
        with TemporaryDirectory() as d:
            base = Path(d)
            first = self._run(base, fn)
            second = self._run(base, fn)  # same alert_state file persists
        self.assertTrue(first["alerts"])
        self.assertEqual(second["alerts"], [])  # unchanged -> no re-fire

    def test_state_change_refires(self) -> None:
        with TemporaryDirectory() as d:
            base = Path(d)
            self._run(base, lambda s: _payload(s, entry="BUY"))
            changed = self._run(base, lambda s: _payload(s, entry="WAIT_FOR_PULLBACK", gate="RR_BELOW_1R"))
        self.assertTrue(changed["alerts"])  # entry_decision changed -> re-fires

    def test_one_bad_symbol_does_not_kill_scan(self) -> None:
        def fn(symbol):
            if symbol == "ETHUSDT":
                raise RuntimeError("data down")
            return _payload(symbol, entry="BUY")

        with TemporaryDirectory() as d:
            result = self._run(Path(d), fn)
        self.assertIn("ETHUSDT", result["errors"])
        self.assertIn("BTCUSDT", result["signals"])


if __name__ == "__main__":
    unittest.main()
