"""Behavioral tests for Stage 1 (input.parse).

We test two layers:
  - `run_stage_1` directly: pure-function behavior (text extraction,
    language detection, ID generation, traceparent propagation).
  - The FastAPI route is exercised in `test_pipeline_e2e.py`.
"""

from __future__ import annotations

from proxy.pipeline.stage_1_input_parse import run_stage_1
from proxy.schemas import OllamaChatRequest, OllamaMessage


def test_run_stage_1_extracts_text_from_messages():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[
            OllamaMessage(role="system", content="be helpful"),
            OllamaMessage(role="user", content="alice@example.com please"),
        ],
    )
    ctx, s1 = run_stage_1(req, traceparent=None, client_ip="127.0.0.1")
    assert "alice@example.com" in s1.extracted_text
    assert "be helpful" in s1.extracted_text
    assert s1.message_count == 2


def test_run_stage_1_detects_korean():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="안녕하세요")],
    )
    _, s1 = run_stage_1(req, traceparent=None, client_ip=None)
    assert s1.language == "ko"


def test_run_stage_1_detects_english():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="hello world")],
    )
    _, s1 = run_stage_1(req, traceparent=None, client_ip=None)
    assert s1.language == "en"


def test_run_stage_1_extracts_traceparent_trace_id():
    incoming = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="x")],
    )
    ctx, _ = run_stage_1(req, traceparent=incoming, client_ip=None)
    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_run_stage_1_mints_trace_id_when_header_missing():
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="x")],
    )
    ctx, _ = run_stage_1(req, traceparent=None, client_ip=None)
    assert len(ctx.trace_id) == 32
    int(ctx.trace_id, 16)


def test_run_stage_1_handles_dict_messages():
    """Ollama SDK allows messages as plain dicts; stage 1 must too."""
    req = OllamaChatRequest(
        model="llama3.2",
        messages=[{"role": "user", "content": "hello"}],
    )
    _, s1 = run_stage_1(req, traceparent=None, client_ip=None)
    assert s1.extracted_text == "hello"
