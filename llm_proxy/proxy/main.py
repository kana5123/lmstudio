"""FastAPI entry point for the LLM proxy.

Endpoint:
  POST /api/chat   — Ollama-compatible chat completion (anonymized + scanned)
  GET  /health     — liveness probe

On startup we boot the telemetry publisher (background task that drains
the event queue and POSTs batches to the visualizer over HTTP).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from proxy import events
from proxy.pipeline.runner import run_pipeline
from proxy.schemas import OllamaChatRequest

VISUALIZER_URL = os.environ.get("VISUALIZER_URL", "http://127.0.0.1:8766")


@asynccontextmanager
async def lifespan(app: FastAPI):
    events.start_publisher(VISUALIZER_URL)
    try:
        yield
    finally:
        await events.stop_publisher()


app = FastAPI(
    title="LLM Proxy",
    description="Ollama-compatible proxy with PII anonymization (Presidio) "
    "and real-time observability.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(
    body: OllamaChatRequest,
    request: Request,
    traceparent: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    client_ip = request.client.host if request.client else None
    ctx, s9, http_status = await run_pipeline(
        body, traceparent=traceparent, client_ip=client_ip
    )
    return JSONResponse(
        content=s9.final_response,
        status_code=http_status,
        headers={
            "x-trace-id": ctx.trace_id,
            "x-request-id": ctx.request_id,
        },
    )
