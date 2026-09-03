"""Stage 3 — presidio.anonymize.

Replaces each detected entity with a numbered placeholder
(e.g. `<PERSON_1>`, `<EMAIL_ADDRESS_2>`) and records a mapping table so
Stage 8 can reverse the replacement after the LLM responds.

Implementation note: Presidio's recognizers can produce overlapping
entities (e.g. an email is also matched by the URL recognizer). We
de-conflict by keeping the highest-score entity when spans overlap, then
do a single right-to-left substitution so coordinates of un-touched
spans never shift.
"""

from __future__ import annotations

from collections import defaultdict

from proxy.schemas import (
    AnonymizeOp,
    EntitySpan,
    Stage2PresidioAnalyzeResult,
    Stage3PresidioAnonymizeResult,
)


def _overlap(a: EntitySpan, b: EntitySpan) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def _dedupe(entities: list[EntitySpan]) -> list[EntitySpan]:
    """Greedy conflict resolution: keep highest-score, drop overlaps."""
    by_score = sorted(entities, key=lambda e: (-e.score, e.start))
    kept: list[EntitySpan] = []
    for e in by_score:
        if not any(_overlap(e, k) for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


def run_stage_3(
    text: str, s2: Stage2PresidioAnalyzeResult
) -> Stage3PresidioAnonymizeResult:
    deconflicted = _dedupe(s2.entities)

    # Number per entity_type in original-text order: PERSON_1, PERSON_2, ...
    counters: dict[str, int] = defaultdict(int)
    tokens: list[str] = []
    for e in deconflicted:
        counters[e.entity_type] += 1
        tokens.append(f"<{e.entity_type}_{counters[e.entity_type]}>")

    mapping: dict[str, str] = {tok: text[e.start:e.end] for e, tok in zip(deconflicted, tokens)}

    # Right-to-left substitution so untouched-span coordinates never shift.
    new_text = text
    for e, tok in sorted(zip(deconflicted, tokens), key=lambda x: -x[0].start):
        new_text = new_text[: e.start] + tok + new_text[e.end :]

    # Operation list: scan left-to-right in the new text to find each token.
    operations: list[AnonymizeOp] = []
    cursor = 0
    for e, tok in zip(deconflicted, tokens):
        idx = new_text.index(tok, cursor)
        operations.append(
            AnonymizeOp(
                start=idx,
                end=idx + len(tok),
                entity_type=e.entity_type,
                text=tok,
                operator="replace",
            )
        )
        cursor = idx + len(tok)

    return Stage3PresidioAnonymizeResult(
        anonymized_text=new_text,
        operations=operations,
        mapping=mapping,
    )
