"""Stage 2 — presidio.analyze.

Detects PII entities (emails, phones, names, credit cards, ...) in the
text extracted at Stage 1. Pure CPU work; no network.

Language fallback: this prototype ships only the English spaCy model.
Inputs in other languages still pass through the pipeline — we silently
fall back to `en` so Presidio runs against whatever it can recognize
(Latin-script PII like emails, URLs, IPs, credit cards still fire even
in non-English text).
"""

from __future__ import annotations

import logging

from proxy.pipeline._engines import get_analyzer
from proxy.schemas import EntitySpan, Stage2PresidioAnalyzeResult

logger = logging.getLogger("proxy.stage_2")

_DEFAULT_THRESHOLD = 0.5
_FALLBACK_LANG = "en"


def run_stage_2(
    text: str,
    *,
    language: str = "en",
    score_threshold: float = _DEFAULT_THRESHOLD,
) -> Stage2PresidioAnalyzeResult:
    analyzer = get_analyzer()
    supported = set(getattr(analyzer, "supported_languages", [_FALLBACK_LANG]))
    effective_lang = language if language in supported else _FALLBACK_LANG
    if effective_lang != language:
        logger.info(
            "language=%s not supported by Presidio; falling back to %s",
            language, effective_lang,
        )

    raw = analyzer.analyze(
        text=text, language=effective_lang, score_threshold=score_threshold
    )
    entities = [
        EntitySpan(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
            recognizer=(r.recognition_metadata or {}).get("recognizer_name"),
        )
        for r in raw
    ]
    return Stage2PresidioAnalyzeResult(
        text_length=len(text),
        language=effective_lang,
        entities=entities,
        score_threshold=score_threshold,
    )
