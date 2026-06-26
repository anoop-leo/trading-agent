"""Airdrop due-diligence screener: turn read-only token-security data into a
structured RISK REPORT with a conservative verdict.

Design rules (enforced here, not optional):
- Default verdict is DO_NOT_INTERACT. A token only earns a softer verdict by
  affirmatively clearing every check.
- Any red flag (FAIL) -> DO_NOT_INTERACT.
- A critical field the data source could not determine (null) is UNKNOWN, never a
  pass -- it blocks PASSED and lands at HIGH_RISK at best.
- Output is safety-to-interact / scam-likelihood ONLY. This module never emits an
  investment opinion, price target, "good bet", or buy/sell recommendation, and
  never tells the user to connect a wallet or claim.
- Every report carries the fixed claim-mechanics danger callout and disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Check statuses. PASS/WARN/FAIL/UNKNOWN feed the verdict; INFO is context only.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
INFO = "INFO"

VERDICT_DO_NOT_INTERACT = "DO_NOT_INTERACT"
VERDICT_HIGH_RISK = "HIGH_RISK_RESEARCH_FURTHER"
VERDICT_PASSED = "PASSED_BASIC_CHECKS_STILL_YOUR_CALL"

# Thresholds (documented, not magic): edit here to retune.
TAX_FAIL_FRACTION = 0.10          # >=10% buy/sell tax is a strong scam signal
CONCENTRATION_FAIL_FRACTION = 0.50  # one non-contract wallet holding a majority
CONCENTRATION_WARN_FRACTION = 0.30
LIQUIDITY_FAIL_USD = 1_000.0      # effectively no exit liquidity
LIQUIDITY_WARN_USD = 20_000.0
LP_LOCKED_OK_FRACTION = 0.95      # share of LP that must be locked/burned to clear

# Boolean GoPlus fields where "1" is an owner rug-vector / honeypot trait => FAIL.
DANGER_FAIL_FIELDS: dict[str, str] = {
    "is_honeypot": "HONEYPOT: token cannot be sold after buying",
    "cannot_sell_all": "Cannot sell entire balance",
    "cannot_buy": "Buying is restricted",
    "is_mintable": "Owner can MINT new supply (inflation / rug vector)",
    "can_take_back_ownership": "Ownership can be reclaimed after being renounced",
    "hidden_owner": "Hidden owner present",
    "selfdestruct": "Contract can self-destruct",
    "transfer_pausable": "Owner can PAUSE all transfers (freeze your ability to sell)",
    "is_blacklisted": "Blacklist function present (owner can block your wallet from selling)",
    "owner_change_balance": "Owner can arbitrarily change balances",
}

# Softer traits where "1" warrants caution but is not an automatic red flag.
DANGER_WARN_FIELDS: dict[str, str] = {
    "trading_cooldown": "Trading cooldown enforced between transactions",
    "slippage_modifiable": "Tax/slippage can be changed by the owner",
    "personal_slippage_modifiable": "Owner can set a per-wallet tax/slippage",
    "is_anti_whale": "Anti-whale max-tx/max-wallet limits present",
    "anti_whale_modifiable": "Anti-whale limits can be changed by the owner",
    "external_call": "Contract makes external calls (added complexity/risk)",
    "is_proxy": "Upgradeable proxy contract (logic can be swapped later)",
    "gas_abuse": "Token has been associated with gas-abuse behavior",
}

CLAIM_MECHANICS_WARNING = (
    "CLAIM MECHANICS -- PRIMARY DANGER: This screen looks only at the token contract. "
    "It cannot see the claim website. If claiming requires CONNECTING YOUR WALLET or "
    "SIGNING a transaction/message on an external site, treat that as the single biggest "
    "risk here: wallet-draining scams operate through exactly that step (malicious "
    "approvals, 'set approval for all', permit signatures). A clean token contract does "
    "NOT make a claim site safe. This tool will never tell you to connect or claim."
)

FIXED_DISCLAIMER = (
    "DISCLAIMER: Most airdrops are worthless or malicious. This screen reduces but does "
    "NOT eliminate risk, and says nothing about whether the token is a good investment "
    "(it never will). Any wallet interaction is always your own manual decision."
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _bool_field(data: dict[str, Any], key: str) -> str | None:
    """GoPlus booleans are '1'/'0' strings, or null/absent when undetermined."""

    value = data.get(key)
    if value in ("1", 1):
        return "1"
    if value in ("0", 0):
        return "0"
    return None


def _parse_fraction(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _danger_checks(data: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for field, danger_detail in DANGER_FAIL_FIELDS.items():
        value = _bool_field(data, field)
        if value == "1":
            checks.append(Check(field, FAIL, danger_detail))
        elif value == "0":
            checks.append(Check(field, PASS, f"OK: {field} not present"))
        else:
            checks.append(Check(field, UNKNOWN, f"Could not determine {field} (source returned no value)"))
    for field, warn_detail in DANGER_WARN_FIELDS.items():
        if _bool_field(data, field) == "1":
            checks.append(Check(field, WARN, warn_detail))
    return checks


def _source_check(data: dict[str, Any]) -> Check:
    value = _bool_field(data, "is_open_source")
    if value == "1":
        return Check("source_verified", PASS, "Contract source code is verified/open")
    if value == "0":
        return Check("source_verified", FAIL, "Contract source code is NOT verified (unverified bytecode)")
    return Check("source_verified", UNKNOWN, "Source-verification status could not be determined")


def _tax_check(data: dict[str, Any], key: str, label: str) -> Check:
    fraction = _parse_fraction(data.get(key))
    if fraction is None:
        return Check(key, UNKNOWN, f"{label} could not be determined")
    pct = fraction * 100
    if fraction >= TAX_FAIL_FRACTION:
        return Check(key, FAIL, f"{label} is {pct:.1f}% (>= {TAX_FAIL_FRACTION * 100:.0f}% — punitive)")
    if fraction > 0:
        return Check(key, WARN, f"{label} is {pct:.1f}%")
    return Check(key, PASS, f"{label} is 0%")


def _concentration_check(data: dict[str, Any]) -> Check:
    holders = data.get("holders")
    if not holders:
        return Check("holder_concentration", UNKNOWN, "No holder distribution data returned")

    top_eoa = 0.0
    top_eoa_addr = ""
    top_contract = 0.0
    for holder in holders:
        if str(holder.get("is_locked")) == "1":
            continue
        percent = _parse_fraction(holder.get("percent")) or 0.0
        if str(holder.get("is_contract")) == "1":
            top_contract = max(top_contract, percent)
        elif percent > top_eoa:
            top_eoa = percent
            top_eoa_addr = str(holder.get("address", ""))

    if top_eoa >= CONCENTRATION_FAIL_FRACTION:
        return Check("holder_concentration", FAIL,
                     f"Single wallet holds {top_eoa * 100:.1f}% of supply ({top_eoa_addr}) — rug risk")
    if top_eoa >= CONCENTRATION_WARN_FRACTION:
        return Check("holder_concentration", WARN,
                     f"Top wallet holds {top_eoa * 100:.1f}% of supply ({top_eoa_addr})")
    if top_contract >= CONCENTRATION_FAIL_FRACTION:
        return Check("holder_concentration", WARN,
                     f"Top holder is a contract at {top_contract * 100:.1f}% (could be a pool/bridge — verify)")
    return Check("holder_concentration", PASS,
                 f"No single wallet dominates supply (largest wallet {top_eoa * 100:.1f}%)")


def _liquidity_check(data: dict[str, Any]) -> Check:
    in_dex = _bool_field(data, "is_in_dex")
    dex = data.get("dex") or []
    total_liquidity = sum(_parse_fraction(entry.get("liquidity")) or 0.0 for entry in dex)
    if in_dex == "0" and not dex:
        return Check("liquidity", FAIL, "Not listed on any DEX — no exit liquidity to sell into")
    if not dex:
        # On a DEX per is_in_dex, but the source returned no per-pair liquidity figures
        # (common for base/blue-chip tokens). Undetermined, not zero -- never FAIL on this.
        return Check("liquidity", UNKNOWN, "Listed on a DEX but the liquidity amount could not be determined")
    if total_liquidity < LIQUIDITY_FAIL_USD:
        return Check("liquidity", FAIL, f"Effectively no liquidity (~${total_liquidity:,.0f})")
    if total_liquidity < LIQUIDITY_WARN_USD:
        return Check("liquidity", WARN, f"Thin liquidity (~${total_liquidity:,.0f}) — high slippage / easy to pull")
    return Check("liquidity", PASS, f"DEX liquidity ~${total_liquidity:,.0f}")


def _lp_lock_check(data: dict[str, Any]) -> Check:
    lp_holders = data.get("lp_holders")
    if not lp_holders:
        return Check("lp_locked", UNKNOWN, "LP lock status could not be determined")
    locked_fraction = 0.0
    top_unlocked = 0.0
    for holder in lp_holders:
        tag = str(holder.get("tag", "")).lower()
        address = str(holder.get("address", "")).lower()
        is_locked = str(holder.get("is_locked")) == "1"
        burned = address in {"0x000000000000000000000000000000000000dead", "0x0000000000000000000000000000000000000000"}
        percent = _parse_fraction(holder.get("percent")) or 0.0
        if is_locked or burned or any(word in tag for word in ("lock", "burn")):
            locked_fraction += percent
        else:
            top_unlocked = max(top_unlocked, percent)
    if locked_fraction >= LP_LOCKED_OK_FRACTION:
        return Check("lp_locked", PASS, f"{locked_fraction * 100:.0f}% of LP is locked/burned")
    if top_unlocked >= 0.50:
        return Check("lp_locked", FAIL,
                     f"A single unlocked address holds {top_unlocked * 100:.0f}% of LP — liquidity can be pulled (rug)")
    if locked_fraction > 0:
        return Check("lp_locked", WARN, f"Only {locked_fraction * 100:.0f}% of LP is locked/burned")
    return Check("lp_locked", WARN, "LP lock could not be confirmed — for a new/airdrop token, treat unlocked LP as a rug risk")


def _context_checks(data: dict[str, Any]) -> list[Check]:
    """Non-scoring context: things this source can't confirm but the report owes."""

    checks = [
        Check("contract_age", INFO,
              "Contract age/creation date is not available from this source "
              "(would require a block-explorer API key — propose before adding)."),
    ]
    if _bool_field(data, "trust_list") == "1":
        checks.append(Check("audit_status", INFO, "Listed on the source's trusted/known-token list"))
    else:
        checks.append(Check("audit_status", INFO, "No audit information available from this source"))
    return checks


def _verdict(checks: list[Check]) -> str:
    statuses = {check.status for check in checks}
    if FAIL in statuses:
        return VERDICT_DO_NOT_INTERACT
    if UNKNOWN in statuses or WARN in statuses:
        return VERDICT_HIGH_RISK
    return VERDICT_PASSED


def screen_token(envelope: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    """Pure: turn a token-security envelope into a structured risk report dict."""

    address = envelope.get("address", "")
    chain = envelope.get("chain", "")

    if not envelope.get("available"):
        # No data is itself a conservative result: cannot clear any check.
        report_checks = [Check("data_availability", UNKNOWN,
                               envelope.get("error") or "No security data available for this token.")]
        return _assemble_report(
            address=address, chain=chain, symbol=symbol,
            token_name=None, token_symbol=symbol,
            checks=report_checks, verdict=VERDICT_DO_NOT_INTERACT,
            data_available=False,
        )

    data = envelope["data"]
    checks: list[Check] = [_source_check(data)]
    checks.extend(_danger_checks(data))
    checks.append(_tax_check(data, "buy_tax", "Buy tax"))
    checks.append(_tax_check(data, "sell_tax", "Sell tax"))
    checks.append(_concentration_check(data))
    checks.append(_liquidity_check(data))
    checks.append(_lp_lock_check(data))

    verdict = _verdict(checks)
    checks.extend(_context_checks(data))  # INFO, added after verdict so it never alters it

    return _assemble_report(
        address=address, chain=chain, symbol=symbol,
        token_name=data.get("token_name"), token_symbol=data.get("token_symbol") or symbol,
        checks=checks, verdict=verdict, data_available=True,
    )


def _assemble_report(
    *, address: str, chain: str, symbol: str | None,
    token_name: str | None, token_symbol: str | None,
    checks: list[Check], verdict: str, data_available: bool,
) -> dict[str, Any]:
    return {
        "address": address,
        "chain": chain,
        "symbol": symbol,
        "token_name": token_name,
        "token_symbol": token_symbol,
        "data_available": data_available,
        "verdict": verdict,
        "red_flags": [c.detail for c in checks if c.status == FAIL],
        "cautions": [c.detail for c in checks if c.status == WARN],
        "unknowns": [c.detail for c in checks if c.status == UNKNOWN],
        "passed": [c.detail for c in checks if c.status == PASS],
        "context": [c.detail for c in checks if c.status == INFO],
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks],
        "claim_mechanics_warning": CLAIM_MECHANICS_WARNING,
        "disclaimer": FIXED_DISCLAIMER,
    }


_VERDICT_HEADLINE = {
    VERDICT_DO_NOT_INTERACT: "DO NOT INTERACT (default — at least one red flag or undetermined critical check)",
    VERDICT_HIGH_RISK: "HIGH RISK — RESEARCH FURTHER (no hard red flag, but it did not cleanly clear every check)",
    VERDICT_PASSED: "PASSED BASIC CHECKS — STILL YOUR CALL (passing != safe != a good investment)",
}


def format_risk_report(report: dict[str, Any]) -> str:
    lines: list[str] = ["=" * 72, "AIRDROP TOKEN RISK REPORT", "=" * 72]
    name = report.get("token_name") or "(unknown name)"
    sym = report.get("token_symbol") or report.get("symbol") or "?"
    lines.append(f"Token:   {name} ({sym})")
    lines.append(f"Address: {report.get('address') or '(none)'}")
    lines.append(f"Chain:   {report.get('chain') or '(unspecified)'}")
    lines.append("")
    lines.append(f"VERDICT: {_VERDICT_HEADLINE.get(report['verdict'], report['verdict'])}")
    lines.append("")

    def _section(title: str, items: list[str], bullet: str) -> None:
        if not items:
            return
        lines.append(title)
        for item in items:
            lines.append(f"  {bullet} {item}")
        lines.append("")

    _section("RED FLAGS:", report.get("red_flags", []), "[X]")
    _section("CAUTIONS:", report.get("cautions", []), "[!]")
    _section("COULD NOT VERIFY:", report.get("unknowns", []), "[?]")
    _section("PASSED:", report.get("passed", []), "[OK]")
    _section("CONTEXT:", report.get("context", []), "[i]")

    lines.append(report["claim_mechanics_warning"])
    lines.append("")
    lines.append(report["disclaimer"])
    lines.append("=" * 72)
    return "\n".join(lines)
