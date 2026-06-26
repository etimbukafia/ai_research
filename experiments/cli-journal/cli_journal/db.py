from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Entity, Episode, JournalSession, OrganizationJob, SemanticFactHint, Thought, utc_now_iso


DEFAULT_DB_PATH = Path.home() / ".cli-journal" / "journal.sqlite3"
DEFAULT_PROFILE_ID = "default"

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    description TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_referenced TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    UNIQUE(profile_id, canonical_name)
);

CREATE INDEX IF NOT EXISTS idx_entities_profile_type
    ON entities(profile_id, type, canonical_name);

CREATE TABLE IF NOT EXISTS thoughts (
    thought_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    thought_type TEXT NOT NULL,
    body TEXT NOT NULL,
    thought TEXT,
    tags_json TEXT NOT NULL,
    entity_refs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_thoughts_profile_created
    ON thoughts(profile_id, created_at);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    significance TEXT NOT NULL,
    thought_id TEXT,
    thought TEXT,
    tags_json TEXT NOT NULL,
    entity_refs_json TEXT NOT NULL,
    salience_score REAL NOT NULL,
    consolidated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_profile_time
    ON episodes(profile_id, occurred_at);

CREATE TABLE IF NOT EXISTS organization_jobs (
    job_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    thought_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_organization_jobs_status
    ON organization_jobs(profile_id, status, created_at);

CREATE TABLE IF NOT EXISTS semantic_fact_hints (
    hint_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    source_episode_refs_json TEXT NOT NULL,
    support_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_semantic_fact_hints_profile_status
    ON semantic_fact_hints(profile_id, status, updated_at);

CREATE INDEX IF NOT EXISTS idx_semantic_fact_hints_subject_predicate
    ON semantic_fact_hints(profile_id, subject_entity_id, predicate, status);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    rolling_summary TEXT NOT NULL,
    active_thought_ids_json TEXT NOT NULL,
    active_entity_ids_json TEXT NOT NULL,
    recent_queries_json TEXT NOT NULL,
    last_exchange_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_profile_active
    ON sessions(profile_id, status, last_active_at);

CREATE TABLE IF NOT EXISTS app_logs (
    log_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    level TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_app_logs_profile_created
    ON app_logs(profile_id, created_at);
"""


class JournalDatabase:
    """Small SQLite store for thoughts, entities, episodic memory, and semantic facts."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def ensure_profile(self, profile_id: str = DEFAULT_PROFILE_ID, name: str = "User") -> None:
        now = utc_now_iso()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO profiles (profile_id, name, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
                (profile_id, name, now, now),
            )

    def add_log(
        self,
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
        level: str,
        source: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        log_id = f"log_{uuid4().hex[:12]}"
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO app_logs (log_id, profile_id, level, source, message, context_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    log_id,
                    profile_id,
                    level,
                    source,
                    message,
                    _json_dumps(context or {}),
                    utc_now_iso(),
                ),
            )
        return log_id

    def list_logs(self, profile_id: str = DEFAULT_PROFILE_ID, *, limit: int = 50) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM app_logs WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()

    def count_logs(self, profile_id: str = DEFAULT_PROFILE_ID) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) AS count FROM app_logs WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
        return int(row["count"])

    def upsert_entity(self, entity: Entity) -> Entity:
        entity.last_referenced = utc_now_iso()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO entities (entity_id, profile_id, canonical_name, type, aliases_json, description, "
                "first_seen, last_referenced, confidence_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, canonical_name) DO UPDATE SET "
                "type = excluded.type, aliases_json = excluded.aliases_json, description = excluded.description, "
                "last_referenced = excluded.last_referenced, confidence_score = max(confidence_score, excluded.confidence_score)",
                (
                    entity.entity_id,
                    entity.profile_id,
                    entity.canonical_name,
                    entity.type,
                    _json_dumps(entity.aliases),
                    entity.description,
                    entity.first_seen,
                    entity.last_referenced,
                    entity.confidence_score,
                ),
            )
            row = conn.execute(
                "SELECT * FROM entities WHERE profile_id = ? AND canonical_name = ?",
                (entity.profile_id, entity.canonical_name),
            ).fetchone()
        return _entity_from_row(row)

    def list_entities(self, profile_id: str = DEFAULT_PROFILE_ID, *, limit: int = 100) -> list[Entity]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE profile_id = ? ORDER BY last_referenced DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
        return [_entity_from_row(row) for row in rows]

    def find_entity_by_name(self, profile_id: str, name: str) -> Entity | None:
        key = name.strip().lower()
        for entity in self.list_entities(profile_id, limit=1000):
            names = [entity.canonical_name, *entity.aliases]
            if any(value.lower() == key for value in names):
                return entity
        return None

    def add_thought(self, thought: Thought) -> Thought:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO thoughts (thought_id, profile_id, thought_type, body, thought, tags_json, entity_refs_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thought.thought_id,
                    thought.profile_id,
                    thought.thought_type,
                    thought.body,
                    thought.thought,
                    _json_dumps(thought.tags),
                    _json_dumps(thought.entity_refs),
                    thought.created_at,
                ),
            )
        return thought

    def list_thoughts(self, profile_id: str = DEFAULT_PROFILE_ID, *, limit: int = 20) -> list[Thought]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM thoughts WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
        return [_thought_from_row(row) for row in rows]

    def get_thought(self, profile_id: str, thought_id: str) -> Thought | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM thoughts WHERE profile_id = ? AND thought_id = ?",
                (profile_id, thought_id),
            ).fetchone()
        return _thought_from_row(row) if row else None

    def update_thought_organization(
        self,
        profile_id: str,
        thought_id: str,
        *,
        thought_type: str,
        thought: str | None,
        tags: list[str],
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE thoughts SET thought_type = ?, thought = ?, tags_json = ? "
                "WHERE profile_id = ? AND thought_id = ?",
                (thought_type, thought, _json_dumps(tags), profile_id, thought_id),
            )

    def search_thoughts(self, profile_id: str, query: str, *, limit: int = 10) -> list[tuple[Thought, float]]:
        thoughts = self.list_thoughts(profile_id, limit=1000)
        return _rank(
            [
                (thought, _search_score(query, thought.body, " ".join(thought.tags), thought.thought or ""))
                for thought in thoughts
            ],
            limit,
        )

    def add_episode(self, episode: Episode) -> Episode:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO episodes (episode_id, profile_id, occurred_at, event_type, description, significance, "
                "thought_id, thought, tags_json, entity_refs_json, salience_score, consolidated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.episode_id,
                    episode.profile_id,
                    episode.occurred_at,
                    episode.event_type,
                    episode.description,
                    episode.significance,
                    episode.thought_id,
                    episode.thought,
                    _json_dumps(episode.tags),
                    _json_dumps(episode.entity_refs),
                    episode.salience_score,
                    episode.consolidated_at,
                ),
            )
        return episode

    def list_episodes(
        self,
        profile_id: str = DEFAULT_PROFILE_ID,
        *,
        unconsolidated: bool = False,
        limit: int = 50,
    ) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if unconsolidated:
            sql += " AND consolidated_at IS NULL"
        sql += " ORDER BY occurred_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_episode_from_row(row) for row in rows]

    def get_episode(self, profile_id: str, episode_id: str) -> Episode | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE profile_id = ? AND episode_id = ?",
                (profile_id, episode_id),
            ).fetchone()
        return _episode_from_row(row) if row else None

    def update_episode_organization(
        self,
        profile_id: str,
        episode_id: str,
        *,
        event_type: str,
        thought: str | None,
        tags: list[str],
        significance: str,
        salience_score: float,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE episodes SET event_type = ?, thought = ?, tags_json = ?, significance = ?, salience_score = ? "
                "WHERE profile_id = ? AND episode_id = ?",
                (event_type, thought, _json_dumps(tags), significance, salience_score, profile_id, episode_id),
            )

    def mark_episodes_consolidated(self, profile_id: str, episode_ids: list[str]) -> None:
        if not episode_ids:
            return
        now = utc_now_iso()
        with self._tx() as conn:
            conn.executemany(
                "UPDATE episodes SET consolidated_at = ? WHERE profile_id = ? AND episode_id = ?",
                [(now, profile_id, episode_id) for episode_id in episode_ids],
            )

    def create_semantic_fact_hint(self, hint: SemanticFactHint) -> SemanticFactHint:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO semantic_fact_hints (hint_id, profile_id, subject_entity_id, predicate, value, "
                "confidence_score, source_episode_refs_json, support_count, status, rationale, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    hint.hint_id,
                    hint.profile_id,
                    hint.subject_entity_id,
                    hint.predicate,
                    hint.value,
                    hint.confidence_score,
                    _json_dumps(hint.source_episode_refs),
                    hint.support_count,
                    hint.status,
                    hint.rationale,
                    hint.created_at,
                    hint.updated_at,
                ),
            )
        return hint

    def update_semantic_fact_hint(self, hint: SemanticFactHint) -> SemanticFactHint:
        hint.updated_at = utc_now_iso()
        hint.support_count = len(_unique_strings(hint.source_episode_refs))
        with self._tx() as conn:
            conn.execute(
                "UPDATE semantic_fact_hints SET subject_entity_id = ?, predicate = ?, value = ?, confidence_score = ?, "
                "source_episode_refs_json = ?, support_count = ?, status = ?, rationale = ?, updated_at = ? "
                "WHERE profile_id = ? AND hint_id = ?",
                (
                    hint.subject_entity_id,
                    hint.predicate,
                    hint.value,
                    hint.confidence_score,
                    _json_dumps(_unique_strings(hint.source_episode_refs)),
                    hint.support_count,
                    hint.status,
                    hint.rationale,
                    hint.updated_at,
                    hint.profile_id,
                    hint.hint_id,
                ),
            )
        return hint

    def list_semantic_fact_hints(
        self,
        profile_id: str = DEFAULT_PROFILE_ID,
        *,
        status: str | None = "pending",
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        limit: int = 50,
    ) -> list[SemanticFactHint]:
        sql = "SELECT * FROM semantic_fact_hints WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if subject_entity_id is not None:
            sql += " AND subject_entity_id = ?"
            params.append(subject_entity_id)
        if predicate is not None:
            sql += " AND predicate = ?"
            params.append(predicate)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_semantic_fact_hint_from_row(row) for row in rows]

    def get_semantic_fact_hint(self, profile_id: str, hint_id: str) -> SemanticFactHint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM semantic_fact_hints WHERE profile_id = ? AND hint_id = ?",
                (profile_id, hint_id),
            ).fetchone()
        return _semantic_fact_hint_from_row(row) if row else None

    def mark_semantic_fact_hint_promoted(self, profile_id: str, hint_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE semantic_fact_hints SET status = ?, updated_at = ? WHERE profile_id = ? AND hint_id = ?",
                ("promoted", utc_now_iso(), profile_id, hint_id),
            )

    def create_organization_job(self, job: OrganizationJob) -> OrganizationJob:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO organization_jobs (job_id, profile_id, thought_id, episode_id, status, attempts, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.profile_id,
                    job.thought_id,
                    job.episode_id,
                    job.status,
                    job.attempts,
                    job.error,
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def list_organization_jobs(
        self,
        profile_id: str = DEFAULT_PROFILE_ID,
        *,
        status: str = "pending",
        limit: int = 25,
    ) -> list[OrganizationJob]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM organization_jobs WHERE profile_id = ? AND status = ? ORDER BY created_at ASC LIMIT ?",
                (profile_id, status, limit),
            ).fetchall()
        return [_organization_job_from_row(row) for row in rows]

    def mark_organization_job_running(self, job: OrganizationJob) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE organization_jobs SET status = ?, attempts = attempts + 1, updated_at = ? WHERE job_id = ?",
                ("running", utc_now_iso(), job.job_id),
            )

    def complete_organization_job(self, job_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE organization_jobs SET status = ?, error = NULL, updated_at = ? WHERE job_id = ?",
                ("completed", utc_now_iso(), job_id),
            )

    def fail_organization_job(self, job_id: str, error: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE organization_jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                ("pending", error[:1000], utc_now_iso(), job_id),
            )

    def create_session(self, session: JournalSession) -> JournalSession:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, profile_id, name, status, started_at, last_active_at, rolling_summary, "
                "active_thought_ids_json, active_entity_ids_json, recent_queries_json, last_exchange_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.session_id,
                    session.profile_id,
                    session.name,
                    session.status,
                    session.started_at,
                    session.last_active_at,
                    session.rolling_summary,
                    _json_dumps(session.active_thought_ids),
                    _json_dumps(session.active_entity_ids),
                    _json_dumps(session.recent_queries),
                    _json_dumps(session.last_exchange),
                ),
            )
        return session

    def get_session(self, profile_id: str, session_id: str) -> JournalSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE profile_id = ? AND session_id = ?",
                (profile_id, session_id),
            ).fetchone()
        return _session_from_row(row) if row else None

    def get_latest_active_session(self, profile_id: str) -> JournalSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE profile_id = ? AND status = ? ORDER BY last_active_at DESC LIMIT 1",
                (profile_id, "active"),
            ).fetchone()
        return _session_from_row(row) if row else None

    def save_session(self, session: JournalSession) -> JournalSession:
        session.last_active_at = utc_now_iso()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, profile_id, name, status, started_at, last_active_at, rolling_summary, "
                "active_thought_ids_json, active_entity_ids_json, recent_queries_json, last_exchange_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "name = excluded.name, status = excluded.status, last_active_at = excluded.last_active_at, "
                "rolling_summary = excluded.rolling_summary, active_thought_ids_json = excluded.active_thought_ids_json, "
                "active_entity_ids_json = excluded.active_entity_ids_json, recent_queries_json = excluded.recent_queries_json, "
                "last_exchange_json = excluded.last_exchange_json",
                (
                    session.session_id,
                    session.profile_id,
                    session.name,
                    session.status,
                    session.started_at,
                    session.last_active_at,
                    session.rolling_summary,
                    _json_dumps(session.active_thought_ids),
                    _json_dumps(session.active_entity_ids),
                    _json_dumps(session.recent_queries),
                    _json_dumps(session.last_exchange),
                ),
            )
        return session

    def list_sessions(self, profile_id: str = DEFAULT_PROFILE_ID, *, limit: int = 20) -> list[JournalSession]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE profile_id = ? ORDER BY last_active_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def close(self) -> None:
        self._conn.close()


def _rank(items: list[tuple[Any, float]], limit: int) -> list[tuple[Any, float]]:
    return [(item, score) for item, score in sorted(items, key=lambda pair: pair[1], reverse=True) if score > 0][:limit]


def _search_score(query: str, *fields: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    haystack = " ".join(fields).lower()
    score = 0.0
    for token in query_tokens:
        if token in haystack:
            score += 1.0
    if query.lower() in haystack:
        score += 2.0
    return score


def _tokens(value: str) -> list[str]:
    stopwords = {"a", "an", "and", "as", "at", "but", "for", "i", "in", "is", "it", "of", "on", "or", "the", "to"}
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2 and token not in stopwords]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _entity_from_row(row: sqlite3.Row) -> Entity:
    return Entity(
        entity_id=row["entity_id"],
        profile_id=row["profile_id"],
        canonical_name=row["canonical_name"],
        type=row["type"],
        aliases=_json_loads(row["aliases_json"], []),
        description=row["description"],
        first_seen=row["first_seen"],
        last_referenced=row["last_referenced"],
        confidence_score=row["confidence_score"],
    )


def _thought_from_row(row: sqlite3.Row) -> Thought:
    return Thought(
        thought_id=row["thought_id"],
        profile_id=row["profile_id"],
        thought_type=row["thought_type"],
        body=row["body"],
        thought=row["thought"],
        tags=_json_loads(row["tags_json"], []),
        entity_refs=_json_loads(row["entity_refs_json"], []),
        created_at=row["created_at"],
    )


def _episode_from_row(row: sqlite3.Row) -> Episode:
    return Episode(
        episode_id=row["episode_id"],
        profile_id=row["profile_id"],
        occurred_at=row["occurred_at"],
        event_type=row["event_type"],
        description=row["description"],
        significance=row["significance"],
        thought_id=row["thought_id"],
        thought=row["thought"],
        tags=_json_loads(row["tags_json"], []),
        entity_refs=_json_loads(row["entity_refs_json"], []),
        salience_score=row["salience_score"],
        consolidated_at=row["consolidated_at"],
    )


def _organization_job_from_row(row: sqlite3.Row) -> OrganizationJob:
    return OrganizationJob(
        job_id=row["job_id"],
        profile_id=row["profile_id"],
        thought_id=row["thought_id"],
        episode_id=row["episode_id"],
        status=row["status"],
        attempts=row["attempts"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _session_from_row(row: sqlite3.Row) -> JournalSession:
    exchange = _json_loads(row["last_exchange_json"], None)
    return JournalSession(
        session_id=row["session_id"],
        profile_id=row["profile_id"],
        name=row["name"],
        status=row["status"],
        started_at=row["started_at"],
        last_active_at=row["last_active_at"],
        rolling_summary=row["rolling_summary"],
        active_thought_ids=_json_loads(row["active_thought_ids_json"], []),
        active_entity_ids=_json_loads(row["active_entity_ids_json"], []),
        recent_queries=_json_loads(row["recent_queries_json"], []),
        last_exchange=exchange,
    )


def _semantic_fact_hint_from_row(row: sqlite3.Row) -> SemanticFactHint:
    return SemanticFactHint(
        hint_id=row["hint_id"],
        profile_id=row["profile_id"],
        subject_entity_id=row["subject_entity_id"],
        predicate=row["predicate"],
        value=row["value"],
        confidence_score=row["confidence_score"],
        source_episode_refs=_json_loads(row["source_episode_refs_json"], []),
        support_count=row["support_count"],
        status=row["status"],
        rationale=row["rationale"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
