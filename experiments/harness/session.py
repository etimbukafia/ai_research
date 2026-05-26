"""session_manager.py

Append-only session event log with pluggable backends.

Backends
--------
- JSONLSessionManager   – one .jsonl file per session (local dev / testing)
- SQLiteSessionManager  – single SQLite database (embedded, single-process)
- PostgresSessionManager – asyncpg-backed (production, multi-process)
- S3SessionManager      – S3-compatible object store (cloud, high durability)

All backends share the same SessionReader / SessionWriter protocols so they
are interchangeable without touching call-sites.
"""

from __future__ import annotations

import json
import asyncio
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, Optional

from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class SessionEvent(BaseModel):
    """A single recorded event in a session log.

    Attributes
    ----------
    role:
        One of ``"user"``, ``"assistant"``, ``"tool"``, or ``"function"``.
    content:
        Raw text payload of the message.
    timestamp:
        UTC time of recording; auto-filled if omitted.

    Extra fields are preserved (e.g. ``tool_call_id``, ``name``).
    """

    role: Literal["user", "assistant", "tool", "function"]
    content: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"extra": "allow"}

    def serialise(self) -> str:
        """Return a compact JSON string safe for line-oriented storage."""
        return self.model_dump_json()

    @classmethod
    def deserialise(cls, raw: str, *, source: str = "<unknown>") -> "SessionEvent":
        """Parse *raw* JSON; raises ``ValueError`` with context on failure."""
        try:
            return cls.model_validate_json(raw)
        except Exception as exc:
            raise ValueError(f"Corrupt event in {source!r}: {raw!r}") from exc


# ---------------------------------------------------------------------------
# Protocols  (split reader / writer so implementers can mix and match)
# ---------------------------------------------------------------------------

class SessionReader(ABC):
    """Read-only access to a session store."""

    @abstractmethod
    def replay(self, session_id: str) -> Iterator[SessionEvent]:
        """Yield all events for *session_id* in insertion order (sync)."""

    @abstractmethod
    async def replay_async(self, session_id: str) -> AsyncIterator[SessionEvent]:
        """Yield all events for *session_id* in insertion order (async)."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """Return the IDs of all known sessions."""

    @abstractmethod
    async def list_sessions_async(self) -> list[str]:
        """Async variant of :meth:`list_sessions`."""


class SessionWriter(ABC):
    """Write-only access to a session store."""

    @abstractmethod
    def append(self, session_id: str, event: SessionEvent | Dict[str, Any]) -> None:
        """Append *event* to *session_id* (sync, blocking)."""

    @abstractmethod
    async def append_async(self, session_id: str, event: SessionEvent | Dict[str, Any]) -> None:
        """Append *event* to *session_id* (async, non-blocking)."""


class SessionManager(SessionReader, SessionWriter, ABC):
    """Combined reader + writer; the standard interface for most callers."""

    # Convenience: accept raw dicts and normalise to SessionEvent
    @staticmethod
    def _coerce(event: SessionEvent | Dict[str, Any]) -> SessionEvent:
        if isinstance(event, SessionEvent):
            return event
        return SessionEvent(**event)


# ---------------------------------------------------------------------------
# JSONL backend
# ---------------------------------------------------------------------------

class JSONLSessionManager(SessionManager):
    """Append-only session store backed by one ``.jsonl`` file per session.

    Suitable for local development and testing.  Not safe for concurrent
    writers from multiple *processes* (a single process is fine because
    writes are serialised per-session via ``threading.Lock``).

    Parameters
    ----------
    base_path:
        Directory where ``.jsonl`` files are created.
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _file_path(self, session_id: str) -> Path:
        return self.base_path / f"{session_id}.jsonl"

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    # ------------------------------------------------------------------
    # SessionWriter
    # ------------------------------------------------------------------

    def append(self, session_id: str, event: SessionEvent | Dict[str, Any]) -> None:
        ev = self._coerce(event)
        lock = self._lock_for(session_id)
        with lock:
            with self._file_path(session_id).open("a", encoding="utf-8") as f:
                f.write(ev.serialise() + "\n")

    async def append_async(
        self, session_id: str, event: SessionEvent | Dict[str, Any]
    ) -> None:
        await asyncio.to_thread(self.append, session_id, event)

    # ------------------------------------------------------------------
    # SessionReader
    # ------------------------------------------------------------------

    def replay(self, session_id: str) -> Iterator[SessionEvent]:
        path = self._file_path(session_id)
        if not path.exists():
            return
        source = str(path)
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                yield SessionEvent.deserialise(raw, source=f"{source}:{lineno}")

    async def replay_async(self, session_id: str) -> AsyncIterator[SessionEvent]:
        events = await asyncio.to_thread(list, self.replay(session_id))
        for ev in events:
            yield ev

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.base_path.glob("*.jsonl")]

    async def list_sessions_async(self) -> list[str]:
        return await asyncio.to_thread(self.list_sessions)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    event_json  TEXT    NOT NULL,
    recorded_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_events_session_id
    ON session_events(session_id);
"""


class SQLiteSessionManager(SessionManager):
    """Session store backed by a SQLite database.

    Thread-safe for a single process.  Uses ``check_same_thread=False``
    with a ``threading.Lock`` so the same connection can be shared across
    threads without spawning per-thread connections.

    Parameters
    ----------
    db_path:
        Path to the ``.db`` file.  Pass ``":memory:"`` for an in-process
        ephemeral store (useful in tests).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SQLITE_SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self):
        """Yield the connection inside a serialised transaction."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # SessionWriter
    # ------------------------------------------------------------------

    def append(self, session_id: str, event: SessionEvent | Dict[str, Any]) -> None:
        ev = self._coerce(event)
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO session_events (session_id, event_json, recorded_at) "
                "VALUES (?, ?, ?)",
                (session_id, ev.serialise(), ev.timestamp.isoformat()),
            )

    async def append_async(
        self, session_id: str, event: SessionEvent | Dict[str, Any]
    ) -> None:
        await asyncio.to_thread(self.append, session_id, event)

    # ------------------------------------------------------------------
    # SessionReader
    # ------------------------------------------------------------------

    def replay(self, session_id: str) -> Iterator[SessionEvent]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_json FROM session_events "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            rows = cur.fetchall()
        source = f"sqlite:{self.db_path}:{session_id}"
        for (raw,) in rows:
            yield SessionEvent.deserialise(raw, source=source)

    async def replay_async(self, session_id: str) -> AsyncIterator[SessionEvent]:
        events = await asyncio.to_thread(list, self.replay(session_id))
        for ev in events:
            yield ev

    def list_sessions(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT session_id FROM session_events ORDER BY session_id"
            )
            return [row[0] for row in cur.fetchall()]

    async def list_sessions_async(self) -> list[str]:
        return await asyncio.to_thread(self.list_sessions)

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# PostgreSQL backend  (requires: pip install asyncpg)
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_events (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT      NOT NULL,
    event_json  JSONB     NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_events_session_id
    ON session_events(session_id);
"""


class PostgresSessionManager(SessionManager):
    """Session store backed by PostgreSQL via ``asyncpg``.

    This backend is *natively async*; the sync shims (``append``,
    ``replay``, ``list_sessions``) spin up a temporary event loop and
    are provided only for interface compliance — prefer the async variants
    in production async code.

    Parameters
    ----------
    dsn:
        A libpq connection string, e.g.
        ``"postgresql://user:pass@host:5432/dbname"``.
    pool_min / pool_max:
        asyncpg connection pool size bounds.

    Usage
    -----
    Call ``await manager.initialise()`` once before use, and
    ``await manager.close()`` on shutdown (or use as an async context
    manager).
    """

    def __init__(
        self,
        dsn: str,
        *,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Optional[Any] = None  # asyncpg.Pool

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """Create the connection pool and ensure the schema exists."""
        import asyncpg  # deferred so the class is importable without asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_PG_SCHEMA)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def __aenter__(self) -> "PostgresSessionManager":
        await self.initialise()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError(
                "Pool not initialised — call `await manager.initialise()` first."
            )
        return self._pool

    # ------------------------------------------------------------------
    # SessionWriter
    # ------------------------------------------------------------------

    async def append_async(
        self, session_id: str, event: SessionEvent | Dict[str, Any]
    ) -> None:
        ev = self._coerce(event)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO session_events (session_id, event_json, recorded_at) "
                "VALUES ($1, $2::jsonb, $3)",
                session_id,
                ev.serialise(),
                ev.timestamp,
            )

    def append(self, session_id: str, event: SessionEvent | Dict[str, Any]) -> None:
        asyncio.run(self.append_async(session_id, event))

    # ------------------------------------------------------------------
    # SessionReader
    # ------------------------------------------------------------------

    async def replay_async(self, session_id: str) -> AsyncIterator[SessionEvent]:
        pool = self._require_pool()
        source = f"postgres:{session_id}"
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT event_json::text FROM session_events "
                "WHERE session_id = $1 ORDER BY id",
                session_id,
            )
        for row in rows:
            yield SessionEvent.deserialise(row[0], source=source)

    def replay(self, session_id: str) -> Iterator[SessionEvent]:
        async def _collect() -> list[SessionEvent]:
            return [ev async for ev in self.replay_async(session_id)]

        return iter(asyncio.run(_collect()))

    async def list_sessions_async(self) -> list[str]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT session_id FROM session_events ORDER BY session_id"
            )
        return [row[0] for row in rows]

    def list_sessions(self) -> list[str]:
        return asyncio.run(self.list_sessions_async())

