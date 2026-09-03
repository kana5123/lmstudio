"""Stage 9 — response.assemble.

Builds the final HTTP response body. We keep Ollama's response envelope
intact — clients of this proxy can use the standard `ollama` Python SDK
or any Ollama-compatible client unchanged. Only `message.content` is
swapped for the de-anonymized text, and an `x_proxy_metadata` field is
added (clients that don't know about it simply ignore it).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from proxy.schemas import (
    PipelineContext,
    Stage6LLMCallResult,
    Stage8DeanonymizeResult,
    Stage9ResponseAssembleResult,
)


def run_stage_9(
    ctx: PipelineContext,
    s6: Stage6LLMCallResult,
    s8: Stage8DeanonymizeResult,
) -> Stage9ResponseAssembleResult:
    body: dict[str, Any] = s6.response.model_dump(mode="json", exclude_none=True)
    body["message"] = {**body.get("message", {}), "content": s8.deanonymized_text}

    duration_ms = (
        datetime.now(timezone.utc) - ctx.started_at
    ).total_seconds() * 1000.0

    body["x_proxy_metadata"] = {
        "trace_id": ctx.trace_id,
        "request_id": ctx.request_id,
        "pipeline_duration_ms": round(duration_ms, 2),
        "ollama_used_stub": s6.used_stub,
    }

    return Stage9ResponseAssembleResult(
        final_response=body,
        pipeline_duration_ms=round(duration_ms, 2),
        stages_executed=9,
    )


def build_blocked_response(ctx: PipelineContext, reasons: list[str]) -> Stage9ResponseAssembleResult:
    """Build a 4xx response body when Stage 5 blocks the request."""
    duration_ms = (
        datetime.now(timezone.utc) - ctx.started_at
    ).total_seconds() * 1000.0
    body = {
        "error": "Request blocked by safety policy",
        "reasons": reasons,
        "x_proxy_metadata": {
            "trace_id": ctx.trace_id,
            "request_id": ctx.request_id,
            "pipeline_duration_ms": round(duration_ms, 2),
        },
    }
    return Stage9ResponseAssembleResult(
        final_response=body,
        pipeline_duration_ms=round(duration_ms, 2),
        stages_executed=5,
    )
