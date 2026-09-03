"""Round-trip serialization tests for Stage 1 schemas.

A round-trip test takes an object → JSON → object and asserts equality.
It catches missing serializers (datetime, enums), default-value
asymmetries, and accidental field renames during refactoring.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from proxy.schemas import (
    OllamaChatRequest,
    OllamaMessage,
    PipelineContext,
    Stage1InputParseResult,
)


def test_ollama_message_roundtrip():
    msg = OllamaMessage(role="user", content="hello")
    restored = OllamaMessage.model_validate_json(msg.model_dump_json())
    assert msg == restored


def test_ollama_chat_request_roundtrip():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="hi")],
        stream=False,
    )
    restored = OllamaChatRequest.model_validate_json(req.model_dump_json())
    assert req.model_dump() == restored.model_dump()


def test_ollama_chat_request_rejects_empty_model():
    """Ollama enforces `model` min_length=1."""
    with pytest.raises(Exception):  # pydantic ValidationError
        OllamaChatRequest(model="", messages=[])


def test_ollama_chat_request_accepts_dict_messages():
    """Ollama's SDK accepts plain dicts in `messages` — make sure we do too."""
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.messages is not None
    assert len(req.messages) == 1


def test_stage_1_result_roundtrip():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="안녕 alice@example.com")],
    )
    s1 = Stage1InputParseResult(
        request_id="01HW3R5K8YPNZQ4XJVB6F2T9MD",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        received_at=datetime(2026, 5, 8, 14, 23, 15, tzinfo=timezone.utc),
        client_ip="127.0.0.1",
        ollama_request=req,
        extracted_text="안녕 alice@example.com",
        language="ko",
        message_count=1,
    )
    restored = Stage1InputParseResult.model_validate_json(s1.model_dump_json())
    assert s1.model_dump() == restored.model_dump()


def test_pipeline_context_roundtrip_with_stage_1_only():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="x")],
    )
    s1 = Stage1InputParseResult(
        request_id="01HW3R5K8YPNZQ4XJVB6F2T9MD",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        received_at=datetime(2026, 5, 8, 14, 23, 15, tzinfo=timezone.utc),
        ollama_request=req,
        extracted_text="x",
        language="en",
        message_count=1,
    )
    ctx = PipelineContext(
        trace_id=s1.trace_id, request_id=s1.request_id, stage_1=s1
    )
    restored = PipelineContext.model_validate_json(ctx.model_dump_json())
    assert ctx.model_dump() == restored.model_dump()
