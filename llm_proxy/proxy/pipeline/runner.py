"""Pipeline runner — orchestrates all 9 stages.

Responsibilities:
  1. Run each stage in the correct order, threading results via PipelineContext.
  2. Emit a `stage.completed` (or `.blocked`) event per stage to the
     visualization channel. Each event carries the *transformation* of
     this stage (input text + output text) so the dashboard can render
     a side-by-side diff view.
  3. Honor the Stage 5 policy decision (skip stages 6-9 when blocked).

Stages 2 and 4 are independent (both read the original extracted text)
and run concurrently via `asyncio.gather` to shave latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from proxy import events
from proxy.pipeline.stage_1_input_parse import run_stage_1
from proxy.pipeline.stage_2_presidio_analyze import run_stage_2
from proxy.pipeline.stage_3_presidio_anonymize import run_stage_3
from proxy.pipeline.stage_4_injection_detect import run_stage_4
from proxy.pipeline.stage_5_policy_gate import run_stage_5
from proxy.pipeline.stage_6_llm_call import run_stage_6
from proxy.pipeline.stage_7_output_scan import run_stage_7
from proxy.pipeline.stage_8_deanonymize import run_stage_8
from proxy.pipeline.stage_9_response_assemble import (
    build_blocked_response,
    run_stage_9,
)
from proxy.schemas import (
    OllamaChatRequest,
    PipelineContext,
    Stage9ResponseAssembleResult,
)

logger = logging.getLogger("proxy.pipeline")


def _emit(
    trace_id: str,
    name: str,
    idx: int,
    started: float,
    payload: dict[str, Any],
    status: str = "completed",
) -> None:
    duration_ms = (time.perf_counter() - started) * 1000.0
    events.emit(
        trace_id=trace_id,
        stage=name,
        stage_index=idx,
        status=status,
        duration_ms=round(duration_ms, 3),
        payload=payload,
    )


async def run_pipeline(
    req: OllamaChatRequest,
    *,
    traceparent: str | None,
    client_ip: str | None,
) -> tuple[PipelineContext, Stage9ResponseAssembleResult, int]:
    """Run all 9 stages. Returns (context, final stage 9 result, http_status)."""
    # ── Stage 1 ──
    t0 = time.perf_counter()
    ctx, s1 = run_stage_1(req, traceparent=traceparent, client_ip=client_ip)

    # Header event — start of trace, full original input.
    events.emit(
        trace_id=ctx.trace_id,
        stage="request.received",
        stage_index=0,
        status="started",
        duration_ms=0.0,
        payload={
            "model": req.model,
            "client_ip": client_ip,
            "request_id": s1.request_id,
            "original_input": s1.extracted_text,
            "message_count": s1.message_count,
            "stream": bool(req.stream),
        },
    )

    _emit(ctx.trace_id, "input.parse", 1, t0, {
        # 입력: 클라이언트가 보낸 원본 HTTP body
        "raw_request": req.model_dump(exclude_none=True),
        # 출력 메타: 파싱 결과
        "model": s1.ollama_request.model,
        "message_count": s1.message_count,
        "language": s1.language,
        "text_length": len(s1.extracted_text),
        "extracted_text": s1.extracted_text,
    })

    # ── Stages 2 + 4 in parallel ──
    t2 = time.perf_counter()
    t4 = time.perf_counter()
    s2, s4 = await asyncio.gather(
        asyncio.to_thread(run_stage_2, s1.extracted_text, language=s1.language),
        asyncio.to_thread(run_stage_4, s1.extracted_text),
    )
    ctx.stage_2 = s2
    ctx.stage_4 = s4
    _emit(ctx.trace_id, "presidio.analyze", 2, t2, {
        "text": s1.extracted_text,
        "text_length": s2.text_length,
        "language": s2.language,
        "entities_count": len(s2.entities),
        "entities": [e.model_dump() for e in s2.entities],
    })
    _emit(ctx.trace_id, "injection.detect", 4, t4, {
        "text": s1.extracted_text,
        "classifier": s4.classifier,
        "label": s4.label,
        "confidence": s4.confidence,
        "scores": s4.scores,
        "matched_patterns": s4.matched_patterns,
    })

    # ── Stage 3 ──
    t3 = time.perf_counter()
    s3 = await asyncio.to_thread(run_stage_3, s1.extracted_text, s2)
    ctx.stage_3 = s3
    _emit(ctx.trace_id, "presidio.anonymize", 3, t3, {
        "original_text": s1.extracted_text,
        "anonymized_text": s3.anonymized_text,
        "operations_count": len(s3.operations),
        "operations": [o.model_dump() for o in s3.operations],
        "token_count": len(s3.mapping),
    })

    # ── Stage 5 ──
    t5 = time.perf_counter()
    s5 = run_stage_5(s2, s4)
    ctx.stage_5 = s5
    _emit(ctx.trace_id, "policy.gate", 5, t5, {
        "decision": s5.decision,
        "reasons": s5.reasons,
        "matched_rules": s5.matched_rules,
        "based_on": {
            "injection_label": s4.label,
            "entities_seen": len(s2.entities),
        },
    })

    if s5.decision == "block":
        s9 = build_blocked_response(ctx, s5.reasons)
        ctx.stage_9 = s9
        events.emit(
            trace_id=ctx.trace_id,
            stage="request.completed",
            stage_index=99,
            status="blocked",
            duration_ms=s9.pipeline_duration_ms,
            payload={
                "http_status": 403,
                "stages_executed": 5,
                "reasons": s5.reasons,
                "final_response": s9.final_response,
            },
        )
        return ctx, s9, 403

    # ── Stage 6 ──
    t6 = time.perf_counter()
    s6 = await run_stage_6(req, s3)
    ctx.stage_6 = s6
    response_content = s6.response.message.content or ""
    _emit(ctx.trace_id, "llm.call", 6, t6, {
        "request_text": s3.anonymized_text,
        "response_text": response_content,
        **s6.otel_attrs,
        "used_stub": s6.used_stub,
        "done_reason": s6.response.done_reason,
    })

    # ── Stage 7 ──
    t7 = time.perf_counter()
    s7 = await asyncio.to_thread(run_stage_7, s6, language=s1.language)
    ctx.stage_7 = s7
    _emit(ctx.trace_id, "output.scan", 7, t7, {
        "scanned_text": response_content,
        "leaked_count": len(s7.leaked_entities),
        "leaked_entities": [e.model_dump() for e in s7.leaked_entities],
        "masked": s7.masked,
        "masked_text": s7.masked_text,
    })

    # ── Stage 8 ──
    t8 = time.perf_counter()
    s8 = run_stage_8(s3, s6, s7)
    ctx.stage_8 = s8
    _emit(ctx.trace_id, "deanonymize", 8, t8, {
        "before_text": s7.masked_text or response_content,
        "after_text": s8.deanonymized_text,
        "tokens_replaced": s8.tokens_replaced,
    })

    # ── Stage 9 ──
    t9 = time.perf_counter()
    s9 = run_stage_9(ctx, s6, s8)
    ctx.stage_9 = s9
    final_msg = s9.final_response.get("message", {}).get("content", "")
    _emit(ctx.trace_id, "response.assemble", 9, t9, {
        "final_message_content": final_msg,
        "pipeline_duration_ms": s9.pipeline_duration_ms,
        "stages_executed": s9.stages_executed,
    })

    events.emit(
        trace_id=ctx.trace_id,
        stage="request.completed",
        stage_index=99,
        status="success",
        duration_ms=s9.pipeline_duration_ms,
        payload={
            "http_status": 200,
            "stages_executed": 9,
            "final_response": s9.final_response,
            "model": s6.response.model or req.model,
            "input_tokens": s6.otel_attrs.get("gen_ai.usage.input_tokens", 0),
            "output_tokens": s6.otel_attrs.get("gen_ai.usage.output_tokens", 0),
        },
    )
    return ctx, s9, 200
