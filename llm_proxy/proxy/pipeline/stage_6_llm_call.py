"""Stage 6 — llm.call.

POSTs the anonymized chat to Ollama (`/api/chat`). If Ollama is
unreachable, returns a deterministic stub response so the rest of the
pipeline can be exercised end-to-end without infrastructure.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

from proxy.schemas import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaMessage,
    Stage3PresidioAnonymizeResult,
    Stage6LLMCallResult,
)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _make_anonymized_request(
    original: OllamaChatRequest, anonymized_text: str
) -> dict:
    """Build a JSON body for Ollama with the *anonymized* user message.

    Strategy: concatenate all original messages into a single anonymized
    user message. This is the simplest correct mapping when there are
    multiple messages — for a real product we'd anonymize per-message,
    preserving roles. For the prototype, single-shot is fine.
    """
    body = original.model_dump(exclude_none=True)
    body["messages"] = [{"role": "user", "content": anonymized_text}]
    body["stream"] = False
    return body


def _build_stub(anonymized_text: str, model: str) -> OllamaChatResponse:
    """Return a deterministic fake response that echoes the anonymized prompt.

    The point of the stub is to keep the placeholder tokens in the
    output so Stages 7/8 have something realistic to operate on.
    """
    content = f"[stub] received: {anonymized_text}"
    return OllamaChatResponse(
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        message=OllamaMessage(role="assistant", content=content),
        done=True,
        done_reason="stop",
        total_duration=0,
        load_duration=0,
        prompt_eval_count=len(anonymized_text.split()),
        prompt_eval_duration=0,
        eval_count=len(content.split()),
        eval_duration=0,
    )


async def run_stage_6(
    original_request: OllamaChatRequest,
    s3: Stage3PresidioAnonymizeResult,
    *,
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> Stage6LLMCallResult:
    base = base_url or OLLAMA_BASE
    body = _make_anonymized_request(original_request, s3.anonymized_text)
    used_stub = False
    response: OllamaChatResponse

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(f"{base}/api/chat", json=body)
            r.raise_for_status()
            response = OllamaChatResponse.model_validate(r.json())
    except Exception:
        response = _build_stub(s3.anonymized_text, original_request.model)
        used_stub = True

    otel_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": original_request.model,
        "gen_ai.response.model": response.model or original_request.model,
        "gen_ai.usage.input_tokens": response.prompt_eval_count or 0,
        "gen_ai.usage.output_tokens": response.eval_count or 0,
        "server.address": base,
    }

    return Stage6LLMCallResult(
        response=response,
        otel_attrs=otel_attrs,
        used_stub=used_stub,
    )
