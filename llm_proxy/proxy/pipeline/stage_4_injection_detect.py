"""Stage 4 — injection.detect.

A lightweight regex/keyword classifier that flags obvious prompt-injection
or jailbreak patterns. This is not a substitute for a real model
(PromptGuard-86M, LLM Guard) — it's a placeholder with the same I/O shape
so the rest of the pipeline can be developed and the detector swapped
later without changing the schema.
"""

from __future__ import annotations

import re

from proxy.schemas import Stage4InjectionDetectResult

# Pattern → severity. Matching any pushes us toward INJECTION/JAILBREAK.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (?:all |any |the )?(?:previous|above|prior)\s+(?:instructions?|prompts?)", "INJECTION"),
    (r"disregard (?:all |any |the )?(?:previous|above|prior)\s+(?:instructions?|prompts?)", "INJECTION"),
    (r"forget (?:everything|all|your)", "INJECTION"),
    (r"system\s*[:\-]?\s*you\s+are", "INJECTION"),
    (r"```\s*system\b", "INJECTION"),
    (r"\bDAN\b", "JAILBREAK"),
    (r"do anything now", "JAILBREAK"),
    (r"developer\s+mode", "JAILBREAK"),
    (r"jailbreak", "JAILBREAK"),
    (r"pretend (?:to be|you (?:are|have))", "JAILBREAK"),
    (r"roleplay as", "JAILBREAK"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _INJECTION_PATTERNS]

CLASSIFIER_NAME = "regex-injection-v0"


def run_stage_4(text: str) -> Stage4InjectionDetectResult:
    matches: list[str] = []
    label_votes: dict[str, int] = {"BENIGN": 0, "INJECTION": 0, "JAILBREAK": 0}

    for pattern, label in _COMPILED:
        if pattern.search(text):
            matches.append(pattern.pattern)
            label_votes[label] += 1

    if label_votes["JAILBREAK"]:
        label = "JAILBREAK"
    elif label_votes["INJECTION"]:
        label = "INJECTION"
    else:
        label = "BENIGN"

    total = sum(label_votes.values()) or 1
    if label == "BENIGN":
        scores = {"BENIGN": 1.0, "INJECTION": 0.0, "JAILBREAK": 0.0}
        confidence = 1.0
    else:
        scores = {k: v / total for k, v in label_votes.items()}
        scores.setdefault("BENIGN", max(0.0, 1.0 - sum(scores.values())))
        confidence = scores[label]

    return Stage4InjectionDetectResult(
        classifier=CLASSIFIER_NAME,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        scores=scores,
        matched_patterns=matches,
    )
