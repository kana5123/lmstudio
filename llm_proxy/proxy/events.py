"""In-process event bus + HTTP webhook publisher to the visualizer.

Design:
  - Hot path (`emit`) is non-blocking: drop on full, never raise.
  - A background task drains the queue and POSTs batches to the visualizer
    over HTTP. If the visualizer is down, batches are dropped (best-effort
    telemetry — never block the proxy on observability failures).

This keeps the proxy and visualizer truly decoupled: a visualizer outage
never affects user-facing latency.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from ulid import ULID

from proxy.schemas import StageEvent

logger = logging.getLogger("proxy.events")

# Bounded queue. Drop-on-full instead of OOM.
_QUEUE: asyncio.Queue[StageEvent] = asyncio.Queue(maxsize=10_000)
_DROPPED = 0
_BACKGROUND_TASK: asyncio.Task | None = None


def _format_traceparent(trace_id: str) -> str:
    """Build a minimal valid W3C traceparent (parent span = zeros)."""
    return f"00-{trace_id}-0000000000000000-01"


def emit(
    *,
    trace_id: str,
    stage: str,
    stage_index: int,
    status: str,
    duration_ms: float,
    payload: dict[str, Any],
    source: str = "/proxy/gateway-1",
) -> None:
    """Fire-and-forget event emission. Never raises."""
    global _DROPPED
    event = StageEvent(
        id=str(ULID()),
        source=source,
        type=f"ai.gateway.stage.{status}",
        time=datetime.now(timezone.utc),
        subject=f"trace/{trace_id}",
        traceparent=_format_traceparent(trace_id),
        data={
            "stage": stage,
            "stage_index": stage_index,
            "status": status,
            "duration_ms": duration_ms,
            **payload,
        },
    )
    try:
        _QUEUE.put_nowait(event)
    except asyncio.QueueFull:
        _DROPPED += 1
        if _DROPPED % 100 == 1:
            logger.warning("telemetry queue full — dropped %d events", _DROPPED)


def dropped_count() -> int:
    return _DROPPED


async def _drain_loop(visualizer_url: str) -> None:
    """Background task: drain queue → POST batches to visualizer."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            batch: list[StageEvent] = []
            # Wait for at least one event, then collect up to 50 with a 0.5s window.
            try:
                first = await _QUEUE.get()
                batch.append(first)
                while len(batch) < 50:
                    try:
                        ev = await asyncio.wait_for(_QUEUE.get(), timeout=0.5)
                        batch.append(ev)
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break

            if not batch:
                continue

            try:
                await client.post(
                    f"{visualizer_url}/events",
                    json=[e.model_dump(mode="json") for e in batch],
                )
            except Exception as e:
                # Best-effort — never propagate. The proxy is unaffected.
                logger.debug("visualizer publish failed: %s", e)


def start_publisher(visualizer_url: str) -> None:
    """Start the background drain task. Idempotent."""
    global _BACKGROUND_TASK
    if _BACKGROUND_TASK is None or _BACKGROUND_TASK.done():
        loop = asyncio.get_event_loop()
        _BACKGROUND_TASK = loop.create_task(_drain_loop(visualizer_url))


async def stop_publisher() -> None:
    global _BACKGROUND_TASK
    if _BACKGROUND_TASK is not None:
        _BACKGROUND_TASK.cancel()
        try:
            await _BACKGROUND_TASK
        except asyncio.CancelledError:
            pass
        _BACKGROUND_TASK = None
