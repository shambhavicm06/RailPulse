"""
Append-only audit trail.

A dispatcher tool makes claims that later have to be defended ("the system said
48 minutes, we held the train, here is what actually happened"). This module
records, for every API call and every operational decision:

* **who** (authenticated username + role, or ``anonymous``),
* **what** (method, path, status, duration),
* **when** (UTC epoch + ISO timestamp),
* the **request/response digest** for prediction calls (inputs and the headline
  output), so a prediction can be replayed and reviewed — without storing
  personal data or full payloads.

Storage is SQLite in WAL mode: append-only in practice, queryable for review,
and safe for the FastAPI threadpool (a fresh connection per write).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DB = Path(__import__("os").environ.get("SWR_AUDIT_DB", ROOT / "data" / "audit.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    iso_ts        TEXT    NOT NULL,
    username      TEXT,
    role          TEXT,
    event         TEXT    NOT NULL,
    method        TEXT,
    path          TEXT,
    status        INTEGER,
    duration_ms   REAL,
    client_ip     TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts   ON audit_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(username);
CREATE INDEX IF NOT EXISTS idx_audit_evt  ON audit_log(event);
"""

_write_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUDIT_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _write_lock, _connect() as conn:
        conn.executescript(_SCHEMA)


def log_event(event: str, *, username: Optional[str] = None, role: Optional[str] = None,
              method: Optional[str] = None, path: Optional[str] = None,
              status_code: Optional[int] = None, duration_ms: Optional[float] = None,
              client_ip: Optional[str] = None, detail: Any = None) -> None:
    """Append one row. Never raises: auditing must not break the request path."""
    try:
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, default=str)[:4000]
        elif detail is not None:
            detail = str(detail)[:4000]
        now = time.time()
        with _write_lock, _connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, iso_ts, username, role, event, method, path,"
                " status, duration_ms, client_ip, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now, datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
                 username, role, event, method, path, status_code, duration_ms, client_ip,
                 detail),
            )
    except Exception:  # noqa: BLE001 - auditing is best-effort by design
        pass


def recent(limit: int = 100, *, username: Optional[str] = None,
           event: Optional[str] = None) -> list[dict]:
    """Most recent entries first, optionally filtered."""
    init_db()
    sql = ("SELECT ts, iso_ts, username, role, event, method, path, status, duration_ms,"
           " client_ip, detail FROM audit_log")
    clauses, params = [], []
    if username:
        clauses.append("username = ?")
        params.append(username)
    if event:
        clauses.append("event = ?")
        params.append(event)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    keys = ["ts", "iso_ts", "username", "role", "event", "method", "path", "status",
            "duration_ms", "client_ip", "detail"]
    return [dict(zip(keys, r)) for r in rows]


def summary() -> dict:
    """Counts used by the dashboard's 'governance' tile."""
    init_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        preds = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event = 'prediction'").fetchone()[0]
        denied = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event = 'auth_denied'").fetchone()[0]
        logins = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event = 'login_success'").fetchone()[0]
        last = conn.execute("SELECT iso_ts FROM audit_log ORDER BY ts DESC LIMIT 1").fetchone()
    return {"entries": total, "predictions": preds, "auth_denied": denied,
            "successful_logins": logins,
            "last_entry": last[0] if last else None,
            "database": str(AUDIT_DB.relative_to(ROOT)) if AUDIT_DB.is_relative_to(ROOT)
                        else str(AUDIT_DB)}


class AuditMiddleware(BaseHTTPMiddleware):
    """Records every request, including the authenticated identity."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        principal = getattr(request.state, "principal", None)
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            path = request.url.path
            # Skip high-volume static/tile traffic; keep the operational trail.
            if not (path.startswith("/static") or path.startswith("/lib")
                    or path.startswith("/tiles") or path.startswith("/data-static")):
                duration = (time.perf_counter() - started) * 1000
                event = "request"
                if status_code in (401, 403):
                    event = "auth_denied"
                elif status_code >= 500:
                    event = "error"
                from auth import _client_ip  # local import avoids a cycle
                log_event(
                    event,
                    username=getattr(principal, "username", None),
                    role=getattr(principal, "role", None),
                    method=request.method, path=path, status_code=status_code,
                    duration_ms=round(duration, 2), client_ip=_client_ip(request),
                )
