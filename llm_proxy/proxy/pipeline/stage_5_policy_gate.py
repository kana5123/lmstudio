"""Stage 5 — policy.gate.

Combines outputs from Stages 2 and 4 to make an allow/block decision.
"""

from __future__ import annotations

from proxy.schemas import (
    Stage2PresidioAnalyzeResult,
    Stage4InjectionDetectResult,
    Stage5PolicyGateResult,
)

# Entity types that, at high confidence, should hard-block (independent of
# whether anonymization can mask them — we'd rather refuse than risk leak).
_HARD_BLOCK_ENTITY_TYPES = {"US_SSN", "CREDIT_CARD"}
_HARD_BLOCK_THRESHOLD = 0.95


def run_stage_5(
    s2: Stage2PresidioAnalyzeResult,
    s4: Stage4InjectionDetectResult,
) -> Stage5PolicyGateResult:
    reasons: list[str] = []
    matched_rules: list[str] = []

    if s4.label != "BENIGN":
        reasons.append(f"injection.label={s4.label} (confidence={s4.confidence:.2f})")
        matched_rules.append("rule_injection_block")

    high_risk = [
        e for e in s2.entities
        if e.entity_type in _HARD_BLOCK_ENTITY_TYPES
        and e.score >= _HARD_BLOCK_THRESHOLD
    ]
    if high_risk:
        reasons.append(
            "high-risk PII present: "
            + ", ".join(sorted({e.entity_type for e in high_risk}))
        )
        matched_rules.append("rule_high_risk_pii_block")

    decision = "block" if reasons else "allow"
    return Stage5PolicyGateResult(
        decision=decision,  # type: ignore[arg-type]
        reasons=reasons,
        matched_rules=matched_rules,
    )
