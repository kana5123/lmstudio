"""Stage 7 — output.scan.

Re-runs Presidio's analyzer on the LLM's response text to catch any PII
the model may have leaked despite the anonymized input. Sets `masked=True`
when leaks are found and produces a masked version of the text.
"""

from __future__ import annotations

from proxy.pipeline._engines import get_analyzer
from proxy.schemas import EntitySpan, Stage6LLMCallResult, Stage7OutputScanResult


def _mask_leaks(text: str, leaks: list[EntitySpan]) -> str:
    """Replace each leaked span with `<LEAKED_{TYPE}>`. Right-to-left to
    preserve indices."""
    out = text
    for span in sorted(leaks, key=lambda s: s.start, reverse=True):
        repl = f"<LEAKED_{span.entity_type}>"
        out = out[: span.start] + repl + out[span.end :]
    return out


def run_stage_7(s6: Stage6LLMCallResult, *, language: str = "en") -> Stage7OutputScanResult:
    analyzer = get_analyzer()
    text = s6.response.message.content or ""
    raw = analyzer.analyze(text=text, language=language, score_threshold=0.5)
    leaks = [
        EntitySpan(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
            recognizer=(r.recognition_metadata or {}).get("recognizer_name"),
        )
        for r in raw
    ]

    if not leaks:
        return Stage7OutputScanResult(leaked_entities=[], masked=False, masked_text=None)

    return Stage7OutputScanResult(
        leaked_entities=leaks,
        masked=True,
        masked_text=_mask_leaks(text, leaks),
    )
