import unittest

from decision.investor_recommendation import (
    build_crypto_position_recommendation,
    bucket_for_crypto_symbol,
)


class BucketMappingTests(unittest.TestCase):
    def test_btc_is_core(self) -> None:
        self.assertEqual(bucket_for_crypto_symbol("BTC"), "core")

    def test_established_alts_are_growth(self) -> None:
        self.assertEqual(bucket_for_crypto_symbol("ETH"), "growth")
        self.assertEqual(bucket_for_crypto_symbol("SOL"), "growth")

    def test_smaller_alts_are_speculative(self) -> None:
        self.assertEqual(bucket_for_crypto_symbol("HYPE"), "speculative")
        self.assertEqual(bucket_for_crypto_symbol("ONDO"), "speculative")

    def test_unknown_symbol_defaults_to_speculative(self) -> None:
        self.assertEqual(bucket_for_crypto_symbol("DOGE"), "speculative")


class BuildCryptoPositionRecommendationTests(unittest.TestCase):
    def test_btc_buy_action_with_dca_multiplier_range(self) -> None:
        payload = {
            "agent": "BTC_INVESTOR",
            "symbol": "BTC",
            "accumulation_score": 80,
            "final_investor_action": "INCREASE_DCA_GRADUALLY",
            "confidence_adjusted_dca_multiplier": "1.5x to 2.5x normal DCA",
        }
        recommendation = build_crypto_position_recommendation(payload, default_position_usd=1_000.0)

        self.assertEqual(recommendation.symbol, "BTC")
        self.assertEqual(recommendation.bucket, "core")
        self.assertEqual(recommendation.asset_class, "crypto")
        self.assertEqual(recommendation.action, "buy")
        self.assertEqual(recommendation.conviction_score, 0.8)
        self.assertEqual(recommendation.suggested_size_usd, 2_000.0)

    def test_do_not_accumulate_maps_to_hold_with_zero_size(self) -> None:
        payload = {
            "agent": "CRYPTO_INVESTOR",
            "symbol": "LINK",
            "investor_score": 20,
            "final_investor_action": "DO_NOT_ACCUMULATE",
            "suggested_dca_multiplier": "0x normal DCA",
        }
        recommendation = build_crypto_position_recommendation(payload, default_position_usd=1_000.0)

        self.assertEqual(recommendation.action, "hold")
        self.assertEqual(recommendation.suggested_size_usd, 0.0)
        self.assertEqual(recommendation.bucket, "speculative")

    def test_single_value_multiplier_is_parsed(self) -> None:
        payload = {
            "agent": "CRYPTO_INVESTOR",
            "symbol": "ETH",
            "investor_score": 70,
            "final_investor_action": "ACCUMULATE_SLOWLY",
            "suggested_dca_multiplier": "1.0x to 1.25x normal DCA",
        }
        recommendation = build_crypto_position_recommendation(payload, default_position_usd=1_000.0)

        self.assertEqual(recommendation.bucket, "growth")
        self.assertEqual(recommendation.suggested_size_usd, 1_125.0)


if __name__ == "__main__":
    unittest.main()
