"""Stage 1 — input.parse.

Runs once per request, immediately after FastAPI has validated the body
into an `OllamaChatRequest` object. Responsibilities:

  1. Mint a request_id (ULID) and decide the trace_id (extracted from the
     incoming `traceparent` header if present, otherwise newly minted).
  2. Concatenate every message's text content into a single string —
     `extracted_text`. This is what stages 2 (Presidio analyze) and
     4 (injection.detect) will scan.
  3. Derive a coarse language hint for Presidio.

Stage 1 does not touch the network and does not block. It is pure
in-memory work.
"""

from __future__ import annotations

from ulid import ULID

from proxy.schemas import (
    OllamaChatRequest,
    PipelineContext,
    Stage1InputParseResult,
)
from proxy.tracing import extract_or_new_trace_id


# Hangul (Korean) Unicode block — used for a coarse ko/en hint. We can swap
# this for a real language detector (lingua, fasttext) when needed; Presidio
# only requires "en" or "ko" here so a heuristic is enough for Stage 1.
_HANGUL_RANGE = (0xAC00, 0xD7A3)


def _detect_language(text: str) -> str:
    """Return 'ko' if any Hangul codepoint appears, else 'en'."""
    for ch in text:
        if _HANGUL_RANGE[0] <= ord(ch) <= _HANGUL_RANGE[1]:
            return "ko"
    return "en"


def _extract_text(req: OllamaChatRequest) -> str:
    """Join every message's textual content with newlines.

    Ollama messages may be either `Message` objects or plain dicts (the
    SDK accepts both). We normalize to text only, ignoring images and
    tool_calls — those don't carry PII text.
    """
    parts: list[str] = []
    for m in req.messages or []:
        # m is either a Pydantic Message or a Mapping (dict) — handle both.
        content = m.get("content") if isinstance(m, dict) else m.content
        if content:
            parts.append(content)
    return "\n".join(parts)


def run_stage_1(
    req: OllamaChatRequest,
    *,
    traceparent: str | None,
    client_ip: str | None,
) -> tuple[PipelineContext, Stage1InputParseResult]:
    """Run input.parse and return the seeded PipelineContext.

    Returns both the context and the stage 1 result so the caller can
    forward both to the visualization channel and to stage 2.
    """
    request_id = str(ULID())
    trace_id = extract_or_new_trace_id(traceparent)

    extracted_text = _extract_text(req)
    language = _detect_language(extracted_text)

    result = Stage1InputParseResult(
        request_id=request_id,
        trace_id=trace_id,
        received_at=__import_utcnow(),
        client_ip=client_ip,
        ollama_request=req,
        extracted_text=extracted_text,
        language=language,
        message_count=len(req.messages or []),
    )

    ctx = PipelineContext(
        trace_id=trace_id,
        request_id=request_id,
        started_at=result.received_at,
        stage_1=result,
    )
    return ctx, result


def __import_utcnow():
    """Indirection so tests can monkeypatch the clock if they need to."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
