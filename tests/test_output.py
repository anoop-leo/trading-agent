import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from decision.decision_engine import Decision, DecisionResult, FinalDecisionResult, PriceZone
from scoring.market_regime_skill import MarketRegime, MarketRegimeResult
from scoring.multi_timeframe_skill import Alignment, MultiTimeframeResult, TimeframeSignal
from scoring.risk_reward_skill import RiskRewardResult
from scoring.setup_detection_skill import Setup, SetupResult
from scoring.support_resistance_skill import SupportResistanceResult
from trading_agent.config import AgentConfig
from trading_agent.models import SignalScores
from trading_agent.output import build_output_payload, macd_direction, write_chart, write_json


class OutputTests(unittest.TestCase):
    def test_build_output_payload_is_json_compatible(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "high": 70000.0,
                    "low": 61800.0,
                    "close": 63500.0,
                    "ema_20": 62800.0,
                    "ema_50": 65000.0,
                    "ema_200": 68000.0,
                    "rsi_14": 61.4,
                    "macd": 20.0,
                    "macd_signal": 50.0,
                    "volume": 180.0,
                    "volume_ma_20": 100.0,
                }
            ]
        )

        payload = build_output_payload(
            AgentConfig(symbol="BTCUSDT"),
            frame,
            SignalScores(10, 8, 7, 7, 4, 10, 5),
            DecisionResult(
                decision=Decision.BUY,
                decision_meaning="Potential long entry setup detected.",
                confidence=73,
                entry_zone=PriceZone(low=62800.0, high=63500.0),
                stop_loss=61800.0,
                target_1=68000.0,
                target_2=70000.0,
                rationale=[
                    "BTCUSDT is above EMA20 but below longer trend averages",
                    "RSI is healthy and MACD is bearish",
                    "Bottom detection suggests early reversal formation",
                    "Trade quality scores are SR 4/10, RR 10/10 at 3.82R, and market regime NEUTRAL",
                    "Decision is BUY with 73% confidence from deterministic Phase 1 rules",
                ],
            ),
            SupportResistanceResult(
                support=61800.0,
                resistance=70000.0,
                distance_to_support=1700.0,
                distance_to_resistance=6500.0,
                sr_score=4,
            ),
            RiskRewardResult(risk=1700.0, reward=6500.0, rr_ratio=3.82, rr_score=10),
            MarketRegimeResult(market_regime=MarketRegime.NEUTRAL, regime_score=5),
            SetupResult(
                setup=Setup.BOTTOMING,
                setup_score=8,
                setup_confidence=72,
                setup_reason=["Bottom score is elevated", "Price near support"],
            ),
            MultiTimeframeResult(
                timeframes={
                    "1h": TimeframeSignal(
                        timeframe="1h",
                        trend_score=10,
                        momentum_score=8,
                        volume_score=7,
                        bottom_score=7,
                        sr_score=4,
                        rr_score=10,
                        regime_score=5,
                        setup="BOTTOMING",
                        setup_confidence=72,
                        decision="BUY",
                        price=63500.0,
                        rsi=61.4,
                        macd="bearish",
                        ema20=62800.0,
                        ema50=65000.0,
                        ema200=68000.0,
                        market_regime="NEUTRAL",
                    )
                },
                alignment=Alignment.MIXED_ALIGNMENT,
                alignment_score=50,
                summary="Timeframes conflict. Wait for cleaner alignment.",
            ),
            FinalDecisionResult(
                decision=Decision.WAIT,
                decision_meaning="No clear new long setup. Wait.",
                reason="Timeframes conflict, so wait for cleaner confirmation.",
            ),
        )

        self.assertEqual(
            payload,
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "market_data_source": "BINANCE",
                "position_mode": "NO_POSITION",
                "price": 63500,
                "ema20": 62800,
                "ema50": 65000,
                "ema200": 68000,
                "rsi": 61.4,
                "macd": "bearish",
                "volume_ratio": 1.8,
                "trend_score": 10,
                "momentum_score": 8,
                "volume_score": 7,
                "bottom_score": 7,
                "support": 61800,
                "resistance": 70000,
                "distance_to_support": 1700,
                "distance_to_resistance": 6500,
                "sr_score": 4,
                "risk": 1700,
                "reward": 6500,
                "rr_ratio": 3.82,
                "rr_score": 10,
                "market_regime": "NEUTRAL",
                "regime_score": 5,
                "setup": "BOTTOMING",
                "setup_score": 8,
                "setup_confidence": 72,
                "setup_reason": ["Bottom score is elevated", "Price near support"],
                "recent_swing_high": 70000,
                "recent_swing_low": 61800,
                "decision": "BUY",
                "decision_meaning": "Potential long entry setup detected.",
                "confidence": 73,
                "entry_zone": {
                    "low": 62800,
                    "high": 63500,
                },
                "stop_loss": 61800,
                "target_1": 68000,
                "target_2": 70000,
                "rationale": [
                    "BTCUSDT is above EMA20 but below longer trend averages",
                    "RSI is healthy and MACD is bearish",
                    "Bottom detection suggests early reversal formation",
                    "Trade quality scores are SR 4/10, RR 10/10 at 3.82R, and market regime NEUTRAL",
                    "Decision is BUY with 73% confidence from deterministic Phase 1 rules",
                ],
                "multi_timeframe": {
                    "alignment": "MIXED_ALIGNMENT",
                    "alignment_score": 50,
                    "summary": "Timeframes conflict. Wait for cleaner alignment.",
                    "timeframes": {
                        "1h": {
                            "setup": "BOTTOMING",
                            "decision": "BUY",
                            "trend_score": 10,
                            "momentum_score": 8,
                            "volume_score": 7,
                            "bottom_score": 7,
                            "sr_score": 4,
                            "rr_score": 10,
                            "regime_score": 5,
                            "setup_confidence": 72,
                            "price": 63500,
                            "rsi": 61.4,
                            "macd": "bearish",
                            "ema20": 62800,
                            "ema50": 65000,
                            "ema200": 68000,
                            "market_regime": "NEUTRAL",
                        },
                    },
                },
                "final_decision": "WAIT",
                "final_decision_reason": "Timeframes conflict, so wait for cleaner confirmation.",
            },
        )

    def test_macd_direction_labels(self) -> None:
        self.assertEqual(macd_direction(pd.Series({"macd": 2.0, "macd_signal": 1.0})), "bullish")
        self.assertEqual(macd_direction(pd.Series({"macd": 1.0, "macd_signal": 2.0})), "bearish")
        self.assertEqual(macd_direction(pd.Series({"macd": 1.0, "macd_signal": 1.0})), "neutral")

    def test_write_json_creates_output_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested"
            path = write_json({"symbol": "BTCUSDT"}, output_dir)

            self.assertEqual(path.name, "output.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"symbol": "BTCUSDT"})

    def test_write_chart_creates_png(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
        close = np.linspace(100.0, 160.0, 220)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "ema_20": close - 0.5,
                "ema_50": close - 1.0,
                "ema_200": close - 1.5,
                "rsi_14": np.linspace(45.0, 65.0, 220),
                "macd": np.linspace(-2.0, 2.0, 220),
                "macd_signal": np.linspace(-1.5, 1.5, 220),
                "macd_histogram": np.linspace(-0.5, 0.5, 220),
                "volume": np.linspace(1000.0, 2000.0, 220),
                "volume_ma_20": np.linspace(950.0, 1800.0, 220),
            }
        )

        with TemporaryDirectory() as temp_dir, TemporaryDirectory() as mpl_config_dir:
            with patch.dict(os.environ, {"MPLCONFIGDIR": mpl_config_dir}):
                path = write_chart(frame, Path(temp_dir))

            self.assertEqual(path.name, "chart.png")
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
