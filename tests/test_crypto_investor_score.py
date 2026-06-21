import unittest

from scoring.crypto_investor_score import (
    CRYPTO_INVESTOR_WEIGHTS,
    band_for_crypto_investor_score,
    calculate_crypto_investor_score,
    factor_payload,
    score_distance_from_200d_ma,
    score_liquidity_proxy,
    score_monthly_trend,
    score_thesis_risk,
    score_volatility_risk,
    score_volume_trend,
    score_weekly_rsi,
)


class CryptoInvestorScoreTests(unittest.TestCase):
    def test_scores_deep_discount_and_constructive_rsi(self) -> None:
        self.assertEqual(score_distance_from_200d_ma(-35), 20)
        self.assertEqual(score_weekly_rsi(38), 13)
        self.assertEqual(score_monthly_trend("BELOW_EMA20"), 15)

    def test_scores_volume_volatility_liquidity_and_thesis_risk(self) -> None:
        self.assertEqual(score_volume_trend(1.6), 10)
        self.assertEqual(score_volatility_risk(3.5), 8)
        self.assertEqual(score_liquidity_proxy(60_000_000), 8)
        self.assertEqual(score_thesis_risk("MODERATE"), 3)

    def test_calculates_investor_score_from_factor_payloads(self) -> None:
        factors = {
            "distance_from_200d_ma": factor_payload(-20, 16, CRYPTO_INVESTOR_WEIGHTS["distance_from_200d_ma"]),
            "weekly_rsi": factor_payload(42, 9, CRYPTO_INVESTOR_WEIGHTS["weekly_rsi"]),
            "thesis_risk_check": factor_payload("LOW", 5, CRYPTO_INVESTOR_WEIGHTS["thesis_risk_check"]),
        }

        self.assertEqual(calculate_crypto_investor_score(factors), 30)

    def test_high_thesis_risk_caps_accumulation_band(self) -> None:
        band = band_for_crypto_investor_score(85, thesis_risk_level="HIGH")

        self.assertEqual(band.investor_band, "ACCUMULATION_ZONE")
        self.assertEqual(band.final_investor_action, "ACCUMULATE_SLOWLY")
        self.assertEqual(band.suggested_dca_multiplier, "1.0x to 1.25x normal DCA")


if __name__ == "__main__":
    unittest.main()
