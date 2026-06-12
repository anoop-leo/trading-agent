import unittest

from backtesting.profiles import PROFILE_NAMES, StrategyProfileError, get_strategy_profile


class StrategyProfileTests(unittest.TestCase):
    def test_profile_parameters_are_defined(self) -> None:
        conservative = get_strategy_profile("conservative")
        balanced = get_strategy_profile("balanced")
        aggressive = get_strategy_profile("aggressive")

        self.assertEqual(PROFILE_NAMES, ("conservative", "balanced", "aggressive"))
        self.assertEqual(conservative.min_rr_ratio, 2.5)
        self.assertEqual(conservative.min_volume_ratio, 1.2)
        self.assertTrue(conservative.require_4h_macd_bullish)
        self.assertTrue(conservative.require_price_above_1h_ema20)
        self.assertEqual(conservative.allocation_per_trade, 0.20)
        self.assertEqual(balanced.min_rr_ratio, 2.0)
        self.assertEqual(balanced.min_volume_ratio, 1.0)
        self.assertEqual(balanced.allocation_per_trade, 0.25)
        self.assertEqual(aggressive.min_rr_ratio, 1.5)
        self.assertEqual(aggressive.min_volume_ratio, 0.8)
        self.assertFalse(aggressive.require_4h_macd_bullish)
        self.assertFalse(aggressive.require_price_above_1h_ema20)
        self.assertEqual(aggressive.allocation_per_trade, 0.30)
        self.assertTrue(aggressive.require_alignment)
        self.assertTrue(aggressive.require_rr_ratio)
        self.assertTrue(aggressive.require_volume_ratio)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(StrategyProfileError):
            get_strategy_profile("fast-and-loose")


if __name__ == "__main__":
    unittest.main()
