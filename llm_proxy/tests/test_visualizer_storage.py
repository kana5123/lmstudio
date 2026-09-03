"""Tests for the visualizer's SQLite log storage."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def temp_db(monkeypatch):
    """Run each test against a fresh, isolated SQLite file."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        monkeypatch.setenv("PROXY_LOG_DB", str(db_path))
        # Reload db module so it picks up the new path
        import importlib
        from visualizer import db as _db
        importlib.reload(_db)
        _db.init_db()
        yield _db


def _ev(stage: str, idx: int, status: str = "completed", **payload):
    return {
        "id": f"ev-{stage}-{idx}",
        "source": "/proxy/test",
        "type": f"ai.gateway.stage.{status}",
        "time": datetime.now(timezone.utc).isoformat(),
        "subject": "trace/abc123",
        "data": {
            "stage": stage,
            "stage_index": idx,
            "status": status,
            "duration_ms": 12.5,
            **payload,
        },
    }


@pytest.mark.asyncio
async def test_records_request_received_creates_trace_row(temp_db):
    await temp_db.record(_ev("request.received", 0, status="started", model="llama3.2"))
    rows = temp_db.list_traces()
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "abc123"
    assert rows[0]["model"] == "llama3.2"
    assert rows[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_request_completed_finalizes_trace(temp_db):
    await temp_db.record(_ev("request.received", 0, status="started", model="llama3.2"))
    await temp_db.record(_ev("presidio.analyze", 2))
    await temp_db.record(_ev(
        "request.completed", 99, status="success",
        http_status=200, stages_executed=9, input_tokens=42, output_tokens=18,
    ))
    rows = temp_db.list_traces()
    assert rows[0]["status"] == "success"
    assert rows[0]["http_status"] == 200
    assert rows[0]["stages_executed"] == 9
    assert rows[0]["input_tokens"] == 42


@pytest.mark.asyncio
async def test_get_events_for_trace_returns_payloads(temp_db):
    await temp_db.record(_ev("request.received", 0, status="started", model="m"))
    await temp_db.record(_ev("input.parse", 1, extracted_text="hi"))
    await temp_db.record(_ev("presidio.analyze", 2, entities=[{"entity_type":"X","start":0,"end":1,"score":1.0}]))
    events = temp_db.get_events_for_trace("abc123")
    assert len(events) == 3
    # sorted by stage_index
    assert events[0]["stage_index"] == 0
    assert events[1]["stage_index"] == 1
    assert events[2]["stage_index"] == 2
    assert events[1]["data"]["extracted_text"] == "hi"
    assert events[2]["data"]["entities"][0]["entity_type"] == "X"


@pytest.mark.asyncio
async def test_stats_aggregates_traces_and_per_stage(temp_db):
    for i in range(3):
        tid = f"t{i}"
        # Note: we override the subject in the helper closure below
        ev1 = _ev("request.received", 0, status="started", model="m")
        ev1["subject"] = f"trace/{tid}"
        ev1["id"] = f"r-{tid}"
        await temp_db.record(ev1)
        ev2 = _ev("input.parse", 1)
        ev2["subject"] = f"trace/{tid}"
        ev2["id"] = f"p-{tid}"
        await temp_db.record(ev2)
        ev3 = _ev("request.completed", 99, status="success",
                  http_status=200, stages_executed=9)
        ev3["subject"] = f"trace/{tid}"
        ev3["id"] = f"c-{tid}"
        await temp_db.record(ev3)

    s = temp_db.stats()
    assert s["total_traces"] == 3
    assert s["by_status"]["success"] == 3
    per_stage = {row["stage_index"]: row for row in s["per_stage"]}
    assert per_stage[1]["n"] == 3
