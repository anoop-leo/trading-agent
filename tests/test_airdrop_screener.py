import unittest

from airdrop.screener import (
    VERDICT_DO_NOT_INTERACT,
    VERDICT_HIGH_RISK,
    VERDICT_PASSED,
    format_risk_report,
    screen_token,
)


def _clean_data(**overrides) -> dict:
    data = {
        "token_name": "Clean Token", "token_symbol": "CLN",
        "is_open_source": "1", "is_honeypot": "0", "cannot_sell_all": "0", "cannot_buy": "0",
        "is_mintable": "0", "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
        "transfer_pausable": "0", "is_blacklisted": "0", "owner_change_balance": "0",
        "buy_tax": "0", "sell_tax": "0",
        "holders": [{"address": "0xab", "percent": "0.08", "is_contract": "0", "is_locked": "0"}],
        "is_in_dex": "1", "dex": [{"liquidity": "500000", "name": "UniV2"}],
        "lp_holders": [{"address": "0xlk", "tag": "unicrypt lock", "percent": "0.99", "is_locked": "1"}],
        "trust_list": "1",
    }
    data.update(overrides)
    return data


def _envelope(data, available=True, error=None, address="0xToken", chain="ethereum") -> dict:
    return {"available": available, "data": data, "error": error, "address": address, "chain": chain}


class VerdictTests(unittest.TestCase):
    def test_clean_token_passes_basic_checks(self) -> None:
        report = screen_token(_envelope(_clean_data()))
        self.assertEqual(report["verdict"], VERDICT_PASSED)
        self.assertEqual(report["red_flags"], [])

    def test_honeypot_is_caught(self) -> None:
        report = screen_token(_envelope(_clean_data(is_honeypot="1")))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertTrue(any("HONEYPOT" in flag for flag in report["red_flags"]))

    def test_mintable_owner_is_red_flag(self) -> None:
        report = screen_token(_envelope(_clean_data(is_mintable="1")))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertTrue(any("MINT" in flag for flag in report["red_flags"]))

    def test_pausable_and_blacklist_are_red_flags(self) -> None:
        report = screen_token(_envelope(_clean_data(transfer_pausable="1", is_blacklisted="1")))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertEqual(len(report["red_flags"]), 2)

    def test_high_sell_tax_is_red_flag(self) -> None:
        report = screen_token(_envelope(_clean_data(sell_tax="0.5")))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)

    def test_small_tax_is_caution_not_pass(self) -> None:
        report = screen_token(_envelope(_clean_data(sell_tax="0.03")))
        self.assertEqual(report["verdict"], VERDICT_HIGH_RISK)
        self.assertEqual(report["red_flags"], [])
        self.assertTrue(report["cautions"])

    def test_whale_concentration_is_red_flag(self) -> None:
        data = _clean_data(holders=[{"address": "0xwhale", "percent": "0.80", "is_contract": "0", "is_locked": "0"}])
        report = screen_token(_envelope(data))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertTrue(any("rug risk" in flag for flag in report["red_flags"]))

    def test_unlocked_lp_is_red_flag(self) -> None:
        data = _clean_data(lp_holders=[{"address": "0xp", "tag": "", "percent": "1.0", "is_locked": "0"}])
        report = screen_token(_envelope(data))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertTrue(any("pulled" in flag for flag in report["red_flags"]))

    def test_distributed_unlocked_lp_is_caution_not_red(self) -> None:
        # Many small unlocked LP providers (organic-looking) -> WARN, not a rug FAIL.
        data = _clean_data(lp_holders=[
            {"address": "0x1", "tag": "", "percent": "0.20", "is_locked": "0"},
            {"address": "0x2", "tag": "", "percent": "0.15", "is_locked": "0"},
        ])
        report = screen_token(_envelope(data))
        self.assertEqual(report["red_flags"], [])
        self.assertEqual(report["verdict"], VERDICT_HIGH_RISK)

    def test_no_dex_liquidity_is_red_flag(self) -> None:
        report = screen_token(_envelope(_clean_data(is_in_dex="0", dex=[])))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)

    def test_on_dex_but_no_liquidity_figure_is_unknown_not_red(self) -> None:
        # Blue-chip/base tokens: is_in_dex=1 but no per-pair liquidity returned.
        report = screen_token(_envelope(_clean_data(is_in_dex="1", dex=[])))
        self.assertNotIn("Effectively no liquidity (~$0)", report["red_flags"])
        self.assertTrue(any("could not be determined" in u for u in report["unknowns"]))

    def test_unverified_source_is_red_flag(self) -> None:
        report = screen_token(_envelope(_clean_data(is_open_source="0")))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)

    def test_null_critical_fields_block_pass_without_red_flag(self) -> None:
        # SQUID-like: source can't determine several critical traits -> never PASSED.
        data = _clean_data(is_honeypot=None, is_mintable=None, transfer_pausable=None, is_blacklisted=None)
        report = screen_token(_envelope(data))
        self.assertEqual(report["verdict"], VERDICT_HIGH_RISK)
        self.assertEqual(report["red_flags"], [])
        self.assertTrue(report["unknowns"])

    def test_unavailable_data_defaults_to_do_not_interact(self) -> None:
        report = screen_token(_envelope(None, available=False, error="token not indexed"))
        self.assertEqual(report["verdict"], VERDICT_DO_NOT_INTERACT)
        self.assertFalse(report["data_available"])


class HardRuleTests(unittest.TestCase):
    def test_every_report_has_claim_warning_and_disclaimer(self) -> None:
        for data in (_clean_data(), _clean_data(is_honeypot="1")):
            report = screen_token(_envelope(data))
            self.assertIn("PRIMARY DANGER", report["claim_mechanics_warning"])
            self.assertIn("CONNECTING YOUR WALLET", report["claim_mechanics_warning"])
            self.assertIn("Most airdrops are worthless or malicious", report["disclaimer"])

    def test_verdict_is_always_one_of_the_three(self) -> None:
        report = screen_token(_envelope(_clean_data()))
        self.assertIn(report["verdict"], {VERDICT_DO_NOT_INTERACT, VERDICT_HIGH_RISK, VERDICT_PASSED})

    def test_report_never_emits_investment_recommendation_language(self) -> None:
        text = format_risk_report(screen_token(_envelope(_clean_data()))).lower()
        for forbidden in ("good bet", "price target", "strong buy", "good buy", "recommend buying"):
            self.assertNotIn(forbidden, text)

    def test_report_never_instructs_to_connect_or_claim(self) -> None:
        text = format_risk_report(screen_token(_envelope(_clean_data())))
        # The only wallet mention is the danger warning, which tells you NOT to be led into it.
        self.assertNotIn("connect your wallet to claim", text.lower())
        self.assertIn("never tell you to connect or claim", text.lower())

    def test_formatted_report_shows_verdict_and_renders(self) -> None:
        text = format_risk_report(screen_token(_envelope(_clean_data(is_honeypot="1"))))
        self.assertIn("DO NOT INTERACT", text)
        self.assertIn("RED FLAGS", text)


if __name__ == "__main__":
    unittest.main()
