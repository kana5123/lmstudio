"""Trace-id helpers per W3C Trace Context (https://www.w3.org/TR/trace-context/).

A `traceparent` header looks like:
    00-{trace_id_32_hex}-{span_id_16_hex}-{flags_2_hex}

Total length: 55 chars. We accept an incoming traceparent if it's well-formed,
otherwise we mint a fresh trace_id.
"""

from __future__ import annotations

import re
import secrets

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)


def new_trace_id() -> str:
    """Generate a new 16-byte (32 hex) trace_id."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Generate a new 8-byte (16 hex) span_id."""
    return secrets.token_hex(8)


def parse_traceparent(header: str | None) -> tuple[str, str] | None:
    """Return (trace_id, parent_span_id) if header is valid, else None."""
    if not header:
        return None
    m = _TRACEPARENT_RE.match(header.strip())
    if not m:
        return None
    return m.group("trace_id"), m.group("span_id")


def extract_or_new_trace_id(traceparent: str | None) -> str:
    """Extract trace_id from incoming header, or mint a new one."""
    parsed = parse_traceparent(traceparent)
    return parsed[0] if parsed else new_trace_id()
