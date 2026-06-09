import unittest

from scoring.market_regime_skill import MarketRegime, calculate_market_regime


class MarketRegimeSkillTests(unittest.TestCase):
    def test_bull_regime_when_price_above_ema200(self) -> None:
        result = calculate_market_regime(current_price=110.0, ema50=100.0, ema200=105.0)

        self.assertEqual(result.market_regime, MarketRegime.BULL)
        self.assertEqual(result.regime_score, 10)

    def test_neutral_regime_when_price_between_ema50_and_ema200(self) -> None:
        result = calculate_market_regime(current_price=102.0, ema50=100.0, ema200=105.0)

        self.assertEqual(result.market_regime, MarketRegime.NEUTRAL)
        self.assertEqual(result.regime_score, 5)

    def test_bear_regime_when_price_below_ema50(self) -> None:
        result = calculate_market_regime(current_price=95.0, ema50=100.0, ema200=105.0)

        self.assertEqual(result.market_regime, MarketRegime.BEAR)
        self.assertEqual(result.regime_score, 0)


if __name__ == "__main__":
    unittest.main()
