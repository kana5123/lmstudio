"""Pydantic schemas for the LLM proxy.

Re-exports Ollama's official Pydantic types as the contract between client
and proxy (we mirror Ollama's `/api/chat` shape exactly), and defines the
proxy-internal types: PipelineContext + Stage 1-9 results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Re-export Ollama's official Pydantic models
from ollama._types import (
    ChatRequest as OllamaChatRequest,
    ChatResponse as OllamaChatResponse,
    Message as OllamaMessage,
    Options as OllamaOptions,
    Tool as OllamaTool,
)

__all__ = [
    "OllamaChatRequest",
    "OllamaChatResponse",
    "OllamaMessage",
    "OllamaOptions",
    "OllamaTool",
    "EntitySpan",
    "AnonymizeOp",
    "Stage1InputParseResult",
    "Stage2PresidioAnalyzeResult",
    "Stage3PresidioAnonymizeResult",
    "Stage4InjectionDetectResult",
    "Stage5PolicyGateResult",
    "Stage6LLMCallResult",
    "Stage7OutputScanResult",
    "Stage8DeanonymizeResult",
    "Stage9ResponseAssembleResult",
    "PipelineContext",
    "StageEvent",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Common building blocks ─────────────────────────────────────────────────

class EntitySpan(BaseModel):
    """Mirrors a subset of Presidio RecognizerResult.to_dict() — the fields
    we keep on the wire for visualization."""
    entity_type: str
    start: int
    end: int
    score: float
    recognizer: str | None = None


class AnonymizeOp(BaseModel):
    """Mirrors a Presidio OperatorResult — one anonymization replacement."""
    start: int
    end: int
    entity_type: str
    text: str          # the replacement token, e.g. "<PERSON_1>"
    operator: str      # "replace", "mask", "hash", ...


# ─── Stage 1 ────────────────────────────────────────────────────────────────

class Stage1InputParseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_id: str
    trace_id: str
    received_at: datetime
    client_ip: str | None = None
    ollama_request: OllamaChatRequest
    extracted_text: str
    language: str = "en"
    message_count: int


# ─── Stage 2 ────────────────────────────────────────────────────────────────

class Stage2PresidioAnalyzeResult(BaseModel):
    text_length: int
    language: str
    entities: list[EntitySpan]
    score_threshold: float


# ─── Stage 3 ────────────────────────────────────────────────────────────────

class Stage3PresidioAnonymizeResult(BaseModel):
    anonymized_text: str
    operations: list[AnonymizeOp]
    # Mapping table for de-anonymization: replacement token -> original text.
    mapping: dict[str, str]


# ─── Stage 4 ────────────────────────────────────────────────────────────────

class Stage4InjectionDetectResult(BaseModel):
    classifier: str
    label: Literal["BENIGN", "INJECTION", "JAILBREAK"]
    confidence: float
    scores: dict[str, float]
    matched_patterns: list[str] = []


# ─── Stage 5 ────────────────────────────────────────────────────────────────

class Stage5PolicyGateResult(BaseModel):
    decision: Literal["allow", "block"]
    reasons: list[str] = []
    matched_rules: list[str] = []


# ─── Stage 6 ────────────────────────────────────────────────────────────────

class Stage6LLMCallResult(BaseModel):
    """Wraps Ollama's ChatResponse plus OTel-style attributes."""
    response: OllamaChatResponse
    otel_attrs: dict[str, Any]
    used_stub: bool = False  # True when Ollama was unreachable and we faked


# ─── Stage 7 ────────────────────────────────────────────────────────────────

class Stage7OutputScanResult(BaseModel):
    leaked_entities: list[EntitySpan] = []
    masked: bool = False
    masked_text: str | None = None


# ─── Stage 8 ────────────────────────────────────────────────────────────────

class Stage8DeanonymizeResult(BaseModel):
    deanonymized_text: str
    tokens_replaced: int


# ─── Stage 9 ────────────────────────────────────────────────────────────────

class Stage9ResponseAssembleResult(BaseModel):
    final_response: dict[str, Any]
    pipeline_duration_ms: float
    stages_executed: int


# ─── Pipeline context ───────────────────────────────────────────────────────

class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    trace_id: str
    request_id: str
    started_at: datetime = Field(default_factory=_utcnow)

    stage_1: Stage1InputParseResult | None = None
    stage_2: Stage2PresidioAnalyzeResult | None = None
    stage_3: Stage3PresidioAnonymizeResult | None = None
    stage_4: Stage4InjectionDetectResult | None = None
    stage_5: Stage5PolicyGateResult | None = None
    stage_6: Stage6LLMCallResult | None = None
    stage_7: Stage7OutputScanResult | None = None
    stage_8: Stage8DeanonymizeResult | None = None
    stage_9: Stage9ResponseAssembleResult | None = None


# ─── Event envelope (CloudEvents v1.0 style) ────────────────────────────────

class StageEvent(BaseModel):
    specversion: Literal["1.0"] = "1.0"
    id: str
    source: str
    type: str
    time: datetime
    datacontenttype: str = "application/json"
    subject: str
    traceparent: str
    data: dict[str, Any]
