"""SQLite-backed log storage for the visualizer.

Two tables:
  traces  — one row per request (summary)
  events  — one row per stage event (full payload as JSON)

Uses sync sqlite3 wrapped in `asyncio.to_thread` so writes don't block
the event loop. WAL mode for concurrent reads while writes happen.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(os.environ.get(
    "PROXY_LOG_DB",
    str(Path(__file__).parent.parent / "proxy_logs.db"),
))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    status        TEXT,             -- 'pending' | 'success' | 'blocked'
    model         TEXT,
    duration_ms   REAL,
    stages_executed INTEGER DEFAULT 0,
    http_status   INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_traces_started_at ON traces(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_status     ON traces(status);

CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    trace_id      TEXT NOT NULL,
    stage         TEXT NOT NULL,
    stage_index   INTEGER NOT NULL,
    status        TEXT NOT NULL,
    time          TEXT NOT NULL,
    duration_ms   REAL,
    type          TEXT,
    payload_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id, stage_index);
CREATE INDEX IF NOT EXISTS idx_events_time  ON events(time DESC);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(_SCHEMA)
        c.commit()


def _record_sync(event: dict[str, Any]) -> None:
    """Insert one event + upsert its trace summary."""
    data = event.get("data", {}) or {}
    trace_id = (event.get("subject") or "trace/unknown").split("/", 1)[-1]
    stage = data.get("stage", "unknown")
    stage_idx = int(data.get("stage_index", 0))
    status = data.get("status", "completed")
    time_iso = event.get("time", "")
    duration_ms = data.get("duration_ms")
    payload_json = json.dumps(data, ensure_ascii=False)

    with _conn() as c:
        cur = c.cursor()

        # Insert event (ignore if id duplicates — events should be unique by ULID)
        cur.execute(
            "INSERT OR IGNORE INTO events "
            "(id, trace_id, stage, stage_index, status, time, duration_ms, type, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.get("id"),
                trace_id,
                stage,
                stage_idx,
                status,
                time_iso,
                duration_ms,
                event.get("type"),
                payload_json,
            ),
        )

        # Upsert trace summary
        if stage == "request.received":
            cur.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, started_at, status, model, stages_executed) "
                "VALUES (?, ?, COALESCE((SELECT status FROM traces WHERE trace_id=?), 'pending'), ?, "
                "       COALESCE((SELECT stages_executed FROM traces WHERE trace_id=?), 0))",
                (trace_id, time_iso, trace_id, data.get("model"), trace_id),
            )
        elif stage == "request.completed":
            cur.execute(
                "UPDATE traces SET ended_at=?, status=?, duration_ms=?, "
                "stages_executed=?, http_status=?, input_tokens=?, output_tokens=?, "
                "model=COALESCE(?, model) "
                "WHERE trace_id=?",
                (
                    time_iso,
                    status,  # 'success' | 'blocked'
                    duration_ms,
                    int(data.get("stages_executed", 0)),
                    int(data.get("http_status", 0)),
                    int(data.get("input_tokens", 0) or 0),
                    int(data.get("output_tokens", 0) or 0),
                    data.get("model"),
                    trace_id,
                ),
            )
        else:
            # Maintain stages_executed count
            cur.execute(
                "UPDATE traces SET stages_executed = MAX(stages_executed, ?) "
                "WHERE trace_id = ? AND ? BETWEEN 1 AND 9",
                (stage_idx, trace_id, stage_idx),
            )

        c.commit()


async def record(event: dict[str, Any]) -> None:
    await asyncio.to_thread(_record_sync, event)


def list_traces(
    *, limit: int = 50, offset: int = 0, status: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM traces"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return dict(row) if row else None


def get_events_for_trace(trace_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE trace_id = ? ORDER BY stage_index, time",
            (trace_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["data"] = json.loads(d.pop("payload_json"))
            except Exception:
                d["data"] = {}
            out.append(d)
        return out


def entity_stats(limit: int = 10) -> list[dict[str, Any]]:
    """Top entity types detected (across all Stage-2 events)."""
    sql = """
        SELECT json_extract(value, '$.entity_type') AS entity_type,
               COUNT(*) AS n,
               AVG(CAST(json_extract(value, '$.score') AS REAL)) AS avg_score
        FROM events,
             json_each(json_extract(payload_json, '$.entities'))
        WHERE stage = 'presidio.analyze'
          AND json_extract(payload_json, '$.entities') IS NOT NULL
        GROUP BY entity_type
        ORDER BY n DESC
        LIMIT ?
    """
    with _conn() as c:
        try:
            return [dict(r) for r in c.execute(sql, (limit,)).fetchall()]
        except Exception:
            return []


def stats() -> dict[str, Any]:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM traces").fetchone()["n"]
        by_status = {
            r["status"] or "pending": r["n"]
            for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM traces GROUP BY status"
            ).fetchall()
        }
        avg = c.execute(
            "SELECT AVG(duration_ms) AS avg_ms, "
            "       MIN(duration_ms) AS min_ms, "
            "       MAX(duration_ms) AS max_ms "
            "FROM traces WHERE duration_ms IS NOT NULL"
        ).fetchone()
        per_stage = [
            dict(r)
            for r in c.execute(
                "SELECT stage, stage_index, "
                "       COUNT(*) AS n, "
                "       AVG(duration_ms) AS avg_ms, "
                "       MAX(duration_ms) AS max_ms "
                "FROM events "
                "WHERE stage_index BETWEEN 1 AND 9 "
                "GROUP BY stage_index, stage "
                "ORDER BY stage_index"
            ).fetchall()
        ]
        return {
            "total_traces": total,
            "by_status": by_status,
            "duration_ms": dict(avg) if avg else {},
            "per_stage": per_stage,
        }
