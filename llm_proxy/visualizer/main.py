"""Visualization server.

Receives stage events from the proxy via `POST /events` (HTTP webhook
batches) and forwards them to web dashboards via Server-Sent Events on
`GET /api/v1/traces/stream`. Maintains an in-memory ring buffer of recent
traces *and* persists every event to SQLite so a long-term log archive
exists even after restarts.

Why two servers? Decoupling at the network boundary makes it explicit
that a visualizer outage cannot affect the proxy's user-facing latency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from visualizer import db

logger = logging.getLogger("visualizer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="LLM Proxy Visualizer", version="0.2.0", lifespan=lifespan)


# ─── In-memory state (live SSE fan-out) ─────────────────────────────────────
_RECENT_TRACES: "OrderedDict[str, list[dict]]" = OrderedDict()
_MAX_TRACES = 200
_SUBSCRIBERS: list[asyncio.Queue] = []
_SUB_LOCK = asyncio.Lock()


# ─── Schemas ────────────────────────────────────────────────────────────────

class _CloudEventLike(BaseModel):
    id: str
    source: str | None = None
    type: str
    time: str
    subject: str | None = None
    traceparent: str | None = None
    data: dict[str, Any]


# ─── Event ingestion ────────────────────────────────────────────────────────

@app.post("/events")
async def receive_events(batch: list[_CloudEventLike]) -> dict[str, int]:
    accepted = 0
    for ev in batch:
        ev_dict = ev.model_dump()
        trace_id = (ev.subject or "trace/unknown").split("/", 1)[-1]

        # Persist to SQLite (off the loop)
        try:
            await db.record(ev_dict)
        except Exception as e:
            logger.warning("db.record failed: %s", e)

        # Live ring buffer
        bucket = _RECENT_TRACES.get(trace_id) or []
        bucket.append(ev_dict)
        _RECENT_TRACES[trace_id] = bucket
        _RECENT_TRACES.move_to_end(trace_id)
        while len(_RECENT_TRACES) > _MAX_TRACES:
            _RECENT_TRACES.popitem(last=False)

        # SSE fan-out
        async with _SUB_LOCK:
            dead: list[asyncio.Queue] = []
            for q in _SUBSCRIBERS:
                try:
                    q.put_nowait(ev_dict)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                if q in _SUBSCRIBERS:
                    _SUBSCRIBERS.remove(q)
        accepted += 1
    return {"accepted": accepted}


# ─── Live read APIs ─────────────────────────────────────────────────────────

@app.get("/api/v1/traces")
def list_recent_traces() -> dict[str, Any]:
    return {
        "traces": [
            {"trace_id": tid, "event_count": len(events)}
            for tid, events in reversed(_RECENT_TRACES.items())
        ]
    }


@app.get("/api/v1/traces/{trace_id}")
def get_recent_trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "events": _RECENT_TRACES.get(trace_id, []),
    }


@app.get("/api/v1/traces/stream")
async def stream_events() -> StreamingResponse:
    """SSE stream of every incoming stage event."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    async with _SUB_LOCK:
        _SUBSCRIBERS.append(queue)

    async def replay_recent() -> list[dict]:
        flat: list[dict] = []
        for events in _RECENT_TRACES.values():
            flat.extend(events)
        return flat[-50:]

    async def gen():
        try:
            for ev in await replay_recent():
                yield _format_sse(ev)
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _format_sse(ev)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            async with _SUB_LOCK:
                if queue in _SUBSCRIBERS:
                    _SUBSCRIBERS.remove(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(event: dict) -> str:
    return (
        f"event: {event.get('type', 'message')}\n"
        f"id: {event.get('id', '')}\n"
        f"data: {json.dumps(event)}\n\n"
    )


# ─── Log archive APIs (SQLite-backed) ───────────────────────────────────────

@app.get("/api/v1/logs/traces")
def logs_list_traces(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> dict[str, Any]:
    return {"traces": db.list_traces(limit=limit, offset=offset, status=status)}


@app.get("/api/v1/logs/traces/{trace_id}")
def logs_get_trace(trace_id: str) -> dict[str, Any]:
    summary = db.get_trace(trace_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"summary": summary, "events": db.get_events_for_trace(trace_id)}


@app.get("/api/v1/logs/stats")
def logs_stats() -> dict[str, Any]:
    return db.stats()


@app.get("/api/v1/logs/entities")
def logs_entities(limit: int = 10) -> dict[str, Any]:
    return {"entities": db.entity_stats(limit=limit)}


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ─── Static dashboard ───────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/logs")
def archive_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "logs.html")
