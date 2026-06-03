from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from experiments.personal_assistant.src.memory import (
    AffectiveState,
    PersonalAssistantEpisodicMemory,
    PersonalAssistantMemory,
    PersonalAssistantProceduralMemory,
    PersonalAssistantSemanticMemory,
    PersonalAssistantWorkingMemory,
)
from experiments.personal_assistant.src.ddc import DDCCycleLog, DDCReviewItem, DDCReviewStatus
from experiments.personal_assistant.src.entities import (
    ContextEntity,
    EntityReviewItem,
    EntityReviewStatus,
)
from experiments.personal_assistant.src.planning import PlannerContinuation, PlannerContinuationStatus


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = APP_ROOT / "personal_assistant.sqlite3"
DEFAULT_PROFILE_ID = UUID("d3b07384-d113-4956-a5e2-4c5b3648a301")
DEFAULT_PROFILE_NAME = "Mike"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_profile (
    profile_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    last_session_id TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS working_memory (
    profile_id                  TEXT PRIMARY KEY,
    current_focus               TEXT,
    active_goals_json           TEXT NOT NULL,
    active_tasks_json           TEXT NOT NULL,
    active_goals_completed_json TEXT NOT NULL,
    open_loops_json             TEXT NOT NULL,
    open_loops_completed_json   TEXT NOT NULL,
    pending_decisions_json      TEXT NOT NULL,
    waiting_on_json             TEXT NOT NULL,
    last_session_id             TEXT,
    updated_at                  TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    profile_id                    TEXT PRIMARY KEY,
    preferences_json              TEXT NOT NULL,
    triggers_json                 TEXT NOT NULL,
    prefers_direct_language       INTEGER NOT NULL,
    dislikes_open_ended_questions INTEGER NOT NULL,
    best_focus_time               TEXT NOT NULL,
    sensitive_to_noise            INTEGER NOT NULL,
    updated_at                    TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS procedural_memory (
    profile_id                         TEXT PRIMARY KEY,
    successful_interventions_json      TEXT NOT NULL,
    routines_that_worked_json          TEXT NOT NULL,
    effective_grouping_strategies_json TEXT NOT NULL,
    preferred_planning_structures_json TEXT NOT NULL,
    updated_at                         TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS affective_state (
    profile_id           TEXT PRIMARY KEY,
    stress_level         REAL NOT NULL,
    energy_level         REAL NOT NULL,
    cognitive_load       REAL NOT NULL,
    social_energy        REAL NOT NULL,
    emotional_regulation REAL NOT NULL,
    executive_function   REAL NOT NULL,
    updated_at           TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS episodic_memory (
    episode_id      TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL,
    last_session_id TEXT,
    title           TEXT,
    summary         TEXT,
    category        TEXT NOT NULL,
    people_json     TEXT NOT NULL,
    related_goals_json TEXT NOT NULL,
    commitments_json   TEXT NOT NULL,
    follow_ups_json    TEXT NOT NULL,
    risks_json         TEXT NOT NULL,
    salience        REAL NOT NULL,
    occurred_on     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_episodic_memory_profile_timestamp
    ON episodic_memory(profile_id, occurred_on);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    profile_id  TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS ddc_review_items (
    review_id       TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL,
    category        TEXT NOT NULL,
    risk            TEXT NOT NULL,
    status          TEXT NOT NULL,
    source_task     TEXT NOT NULL,
    missing_context TEXT NOT NULL,
    proposed_memory TEXT NOT NULL,
    reason          TEXT NOT NULL,
    session_id      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_ddc_review_items_profile_status
    ON ddc_review_items(profile_id, status, updated_at);

CREATE TABLE IF NOT EXISTS ddc_cycle_logs (
    cycle_id        TEXT PRIMARY KEY,
    review_id       TEXT NOT NULL,
    profile_id      TEXT NOT NULL,
    source_task     TEXT NOT NULL,
    category        TEXT NOT NULL,
    action          TEXT NOT NULL,
    promoted_memory TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_ddc_cycle_logs_profile_created
    ON ddc_cycle_logs(profile_id, created_at);

CREATE TABLE IF NOT EXISTS context_metadata (
    profile_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY(profile_id, key),
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE TABLE IF NOT EXISTS planner_continuations (
    continuation_id         TEXT PRIMARY KEY,
    profile_id              TEXT NOT NULL,
    original_user_task      TEXT NOT NULL,
    planner_output_json     TEXT NOT NULL,
    blocking_questions_json TEXT NOT NULL,
    status                  TEXT NOT NULL,
    session_id              TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_planner_continuations_profile_status
    ON planner_continuations(profile_id, status, updated_at);

CREATE TABLE IF NOT EXISTS context_entities (
    entity_id    TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL,
    name         TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    description  TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    source_task  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_context_entities_profile_type
    ON context_entities(profile_id, entity_type, updated_at);

CREATE TABLE IF NOT EXISTS entity_review_items (
    review_id    TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL,
    name         TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    description  TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    reason       TEXT NOT NULL,
    risk         TEXT NOT NULL,
    status       TEXT NOT NULL,
    source_task  TEXT NOT NULL,
    session_id   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES assistant_profile(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_review_items_profile_status
    ON entity_review_items(profile_id, status, updated_at);
"""


class PersonalAssistantDatabase:
    """SQLite profile and typed memory store for a single personal assistant user."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as conn:
            self._migrate_legacy_working_memory(conn)
            self._migrate_legacy_episodic_memory(conn)
            conn.executescript(_SCHEMA)
        self.ensure_profile()

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def ensure_profile(
        self,
        profile_id: UUID = DEFAULT_PROFILE_ID,
        name: str = DEFAULT_PROFILE_NAME,
    ) -> None:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO assistant_profile (profile_id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id) DO NOTHING",
                (str(profile_id), name, now, now),
            )

    def get_profile_name(self, profile_id: UUID = DEFAULT_PROFILE_ID) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM assistant_profile WHERE profile_id = ?",
                (str(profile_id),),
            ).fetchone()
        if row is None:
            self.ensure_profile(profile_id)
            return DEFAULT_PROFILE_NAME
        return row["name"]

    def set_profile_name(self, name: str, profile_id: UUID = DEFAULT_PROFILE_ID) -> None:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO assistant_profile (profile_id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id) DO UPDATE SET "
                "name = excluded.name, updated_at = excluded.updated_at",
                (str(profile_id), name, now, now),
            )

    def get_personal_assistant_memory(
        self,
        profile_id: UUID = DEFAULT_PROFILE_ID,
    ) -> PersonalAssistantMemory:
        self.ensure_profile(profile_id)
        memory = PersonalAssistantMemory(
            name=str(profile_id),
            working=self._get_working_memory(profile_id),
            semantic=self._get_semantic_memory(profile_id),
            procedural=self._get_procedural_memory(profile_id),
            episodic=self._get_episodic_memory(profile_id),
            affective=self._get_affective_state(profile_id),
        )
        memory.update(memory.render())
        self.save_personal_assistant_memory(profile_id, memory)
        return memory

    def save_personal_assistant_memory(
        self,
        profile_id: UUID,
        memory: PersonalAssistantMemory,
    ) -> None:
        self.ensure_profile(profile_id)
        now = _utc_now()
        with self._tx() as conn:
            self._upsert_working_memory(conn, profile_id, memory.working, now)
            self._upsert_semantic_memory(conn, profile_id, memory.semantic, now)
            self._upsert_procedural_memory(conn, profile_id, memory.procedural, now)
            self._upsert_affective_state(conn, profile_id, memory.affective, now)
            self._replace_episodic_memory(conn, profile_id, memory.episodic, memory.working.last_session_id, now)
            conn.execute(
                "UPDATE assistant_profile "
                "SET last_session_id = ?, updated_at = ? "
                "WHERE profile_id = ?",
                (
                    str(memory.working.last_session_id) if memory.working.last_session_id else None,
                    now,
                    str(profile_id),
                ),
            )
            conn.execute(
                "INSERT INTO memory_snapshots (profile_id, memory_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(profile_id) DO UPDATE SET "
                "memory_json = excluded.memory_json, updated_at = excluded.updated_at",
                (str(profile_id), memory.to_json(), now),
            )

    def insert_ddc_review_items(self, items: list[DDCReviewItem]) -> None:
        now = _utc_now()
        with self._tx() as conn:
            for item in items:
                conn.execute(
                    "INSERT INTO ddc_review_items ("
                    "review_id, profile_id, category, risk, status, source_task, "
                    "missing_context, proposed_memory, reason, session_id, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.review_id,
                        str(item.profile_id),
                        item.category,
                        item.risk,
                        item.status,
                        item.source_task,
                        item.missing_context,
                        item.proposed_memory,
                        item.reason,
                        str(item.session_id) if item.session_id else None,
                        item.created_at,
                        now,
                    ),
                )

    def list_ddc_review_items(
        self,
        profile_id: UUID,
        *,
        status: DDCReviewStatus | None = None,
        limit: int = 50,
    ) -> list[DDCReviewItem]:
        query = "SELECT * FROM ddc_review_items WHERE profile_id = ?"
        params: list[Any] = [str(profile_id)]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_ddc_review_item_from_row(row) for row in rows]

    def get_ddc_review_item(self, profile_id: UUID, review_id: str) -> Optional[DDCReviewItem]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ddc_review_items WHERE profile_id = ? AND review_id = ?",
                (str(profile_id), review_id),
            ).fetchone()
        return _ddc_review_item_from_row(row) if row else None

    def set_ddc_review_status(
        self,
        profile_id: UUID,
        review_id: str,
        status: DDCReviewStatus,
    ) -> DDCReviewItem:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE ddc_review_items SET status = ?, updated_at = ? "
                "WHERE profile_id = ? AND review_id = ?",
                (status, now, str(profile_id), review_id),
            )
        item = self.get_ddc_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown DDC review item: {review_id}")
        return item

    def update_ddc_review_memory(
        self,
        profile_id: UUID,
        review_id: str,
        proposed_memory: str,
    ) -> DDCReviewItem:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE ddc_review_items SET proposed_memory = ?, updated_at = ? "
                "WHERE profile_id = ? AND review_id = ?",
                (proposed_memory, now, str(profile_id), review_id),
            )
        item = self.get_ddc_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown DDC review item: {review_id}")
        return item

    def insert_ddc_cycle_log(self, log: DDCCycleLog) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO ddc_cycle_logs ("
                "cycle_id, review_id, profile_id, source_task, category, action, promoted_memory, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    log.cycle_id,
                    log.review_id,
                    str(log.profile_id),
                    log.source_task,
                    log.category,
                    log.action,
                    log.promoted_memory,
                    log.created_at,
                ),
            )

    def list_ddc_cycle_logs(self, profile_id: UUID, *, limit: int = 20) -> list[DDCCycleLog]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM ddc_cycle_logs WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (str(profile_id), limit),
            ).fetchall()
        return [_ddc_cycle_log_from_row(row) for row in rows]

    def get_context_revision(self, profile_id: UUID) -> int:
        self.ensure_profile(profile_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM context_metadata WHERE profile_id = ? AND key = ?",
                (str(profile_id), "personal_context_revision"),
            ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except ValueError:
            return 0

    def increment_context_revision(self, profile_id: UUID) -> int:
        revision = self.get_context_revision(profile_id) + 1
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO context_metadata (profile_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id, key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                (str(profile_id), "personal_context_revision", str(revision), now),
            )
        return revision

    def insert_planner_continuation(self, continuation: PlannerContinuation) -> None:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO planner_continuations ("
                "continuation_id, profile_id, original_user_task, planner_output_json, "
                "blocking_questions_json, status, session_id, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    continuation.continuation_id,
                    str(continuation.profile_id),
                    continuation.original_user_task,
                    continuation.planner_output_json,
                    continuation.blocking_questions_json,
                    continuation.status,
                    str(continuation.session_id) if continuation.session_id else None,
                    continuation.created_at,
                    now,
                ),
            )

    def get_planner_continuation(
        self,
        profile_id: UUID,
        continuation_id: str,
    ) -> Optional[PlannerContinuation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM planner_continuations WHERE profile_id = ? AND continuation_id = ?",
                (str(profile_id), continuation_id),
            ).fetchone()
        return _planner_continuation_from_row(row) if row else None

    def get_pending_planner_continuation(
        self,
        profile_id: UUID,
    ) -> Optional[PlannerContinuation]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM planner_continuations "
                "WHERE profile_id = ? AND status = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (str(profile_id), "pending"),
            ).fetchone()
        return _planner_continuation_from_row(row) if row else None

    def list_planner_continuations(
        self,
        profile_id: UUID,
        *,
        status: PlannerContinuationStatus | None = None,
        limit: int = 20,
    ) -> list[PlannerContinuation]:
        query = "SELECT * FROM planner_continuations WHERE profile_id = ?"
        params: list[Any] = [str(profile_id)]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_planner_continuation_from_row(row) for row in rows]

    def set_planner_continuation_status(
        self,
        profile_id: UUID,
        continuation_id: str,
        status: PlannerContinuationStatus,
    ) -> PlannerContinuation:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE planner_continuations SET status = ?, updated_at = ? "
                "WHERE profile_id = ? AND continuation_id = ?",
                (status, now, str(profile_id), continuation_id),
            )
        continuation = self.get_planner_continuation(profile_id, continuation_id)
        if continuation is None:
            raise ValueError(f"Unknown planner continuation: {continuation_id}")
        return continuation

    def insert_entity_review_items(self, items: list[EntityReviewItem]) -> None:
        now = _utc_now()
        with self._tx() as conn:
            for item in items:
                conn.execute(
                    "INSERT INTO entity_review_items ("
                    "review_id, profile_id, name, entity_type, description, aliases_json, "
                    "reason, risk, status, source_task, session_id, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.review_id,
                        str(item.profile_id),
                        item.name,
                        item.entity_type,
                        item.description,
                        _json_dumps(item.aliases),
                        item.reason,
                        item.risk,
                        item.status,
                        item.source_task,
                        str(item.session_id) if item.session_id else None,
                        item.created_at,
                        now,
                    ),
                )

    def list_entity_review_items(
        self,
        profile_id: UUID,
        *,
        status: EntityReviewStatus | None = None,
        limit: int = 50,
    ) -> list[EntityReviewItem]:
        query = "SELECT * FROM entity_review_items WHERE profile_id = ?"
        params: list[Any] = [str(profile_id)]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_entity_review_item_from_row(row) for row in rows]

    def get_entity_review_item(self, profile_id: UUID, review_id: str) -> Optional[EntityReviewItem]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entity_review_items WHERE profile_id = ? AND review_id = ?",
                (str(profile_id), review_id),
            ).fetchone()
        return _entity_review_item_from_row(row) if row else None

    def update_entity_review_item(
        self,
        profile_id: UUID,
        review_id: str,
        *,
        name: str,
        entity_type: str,
        description: str,
        aliases: list[str],
    ) -> EntityReviewItem:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE entity_review_items "
                "SET name = ?, entity_type = ?, description = ?, aliases_json = ?, updated_at = ? "
                "WHERE profile_id = ? AND review_id = ?",
                (name, entity_type, description, _json_dumps(aliases), now, str(profile_id), review_id),
            )
        item = self.get_entity_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown entity review item: {review_id}")
        return item

    def set_entity_review_status(
        self,
        profile_id: UUID,
        review_id: str,
        status: EntityReviewStatus,
    ) -> EntityReviewItem:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE entity_review_items SET status = ?, updated_at = ? "
                "WHERE profile_id = ? AND review_id = ?",
                (status, now, str(profile_id), review_id),
            )
        item = self.get_entity_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown entity review item: {review_id}")
        return item

    def insert_context_entity(self, entity: ContextEntity) -> ContextEntity:
        now = _utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO context_entities ("
                "entity_id, profile_id, name, entity_type, description, aliases_json, "
                "source_task, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.entity_id,
                    str(entity.profile_id),
                    entity.name,
                    entity.entity_type,
                    entity.description,
                    _json_dumps(entity.aliases),
                    entity.source_task,
                    entity.created_at,
                    now,
                ),
            )
        saved = self.get_context_entity(entity.profile_id, entity.entity_id)
        if saved is None:
            raise ValueError(f"Failed to save context entity: {entity.entity_id}")
        return saved

    def get_context_entity(self, profile_id: UUID, entity_id: str) -> Optional[ContextEntity]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM context_entities WHERE profile_id = ? AND entity_id = ?",
                (str(profile_id), entity_id),
            ).fetchone()
        return _context_entity_from_row(row) if row else None

    def list_context_entities(
        self,
        profile_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[ContextEntity]:
        query = "SELECT * FROM context_entities WHERE profile_id = ?"
        params: list[Any] = [str(profile_id)]
        if entity_type is not None:
            query += " AND entity_type = ?"
            params.append(entity_type)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_context_entity_from_row(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def _migrate_legacy_episodic_memory(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'episodic_memory'"
        ).fetchone()
        if row is None:
            return

        columns = {
            column[1]
            for column in conn.execute("PRAGMA table_info(episodic_memory)").fetchall()
        }
        if {"title", "summary", "category", "occurred_on", "risks_json"}.issubset(columns):
            return

        backup_name = f"episodic_memory_legacy_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        conn.execute(f"ALTER TABLE episodic_memory RENAME TO {backup_name}")

    def _migrate_legacy_working_memory(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'working_memory'"
        ).fetchone()
        if row is None:
            return

        columns = {
            column[1]
            for column in conn.execute("PRAGMA table_info(working_memory)").fetchall()
        }
        if {"current_focus", "active_tasks_json", "pending_decisions_json", "waiting_on_json"}.issubset(columns):
            return

        backup_name = f"working_memory_legacy_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        conn.execute(f"ALTER TABLE working_memory RENAME TO {backup_name}")

    def _get_working_memory(self, profile_id: UUID) -> PersonalAssistantWorkingMemory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM working_memory WHERE profile_id = ?",
                (str(profile_id),),
            ).fetchone()
        if row is None:
            return PersonalAssistantWorkingMemory()
        return PersonalAssistantWorkingMemory(
            current_focus=row["current_focus"],
            active_goals=_json_loads(row["active_goals_json"], []),
            active_tasks=_json_loads(row["active_tasks_json"], []),
            active_goals_completed=_json_loads(row["active_goals_completed_json"], {}),
            open_loops=_json_loads(row["open_loops_json"], []),
            open_loops_completed=_json_loads(row["open_loops_completed_json"], {}),
            pending_decisions=_json_loads(row["pending_decisions_json"], []),
            waiting_on=_json_loads(row["waiting_on_json"], {}),
            last_session_id=UUID(row["last_session_id"]) if row["last_session_id"] else None,
        )

    def _get_semantic_memory(self, profile_id: UUID) -> PersonalAssistantSemanticMemory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM semantic_memory WHERE profile_id = ?",
                (str(profile_id),),
            ).fetchone()
        if row is None:
            return PersonalAssistantSemanticMemory()
        return PersonalAssistantSemanticMemory(
            preferences=_json_loads(row["preferences_json"], []),
            triggers=_json_loads(row["triggers_json"], []),
            prefers_direct_language=bool(row["prefers_direct_language"]),
            dislikes_open_ended_questions=bool(row["dislikes_open_ended_questions"]),
            best_focus_time=row["best_focus_time"],
            sensitive_to_noise=bool(row["sensitive_to_noise"]),
        )

    def _get_procedural_memory(self, profile_id: UUID) -> PersonalAssistantProceduralMemory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM procedural_memory WHERE profile_id = ?",
                (str(profile_id),),
            ).fetchone()
        if row is None:
            return PersonalAssistantProceduralMemory()
        return PersonalAssistantProceduralMemory(
            successful_interventions=_json_loads(row["successful_interventions_json"], []),
            routines_that_worked=_json_loads(row["routines_that_worked_json"], []),
            effective_grouping_strategies=_json_loads(row["effective_grouping_strategies_json"], []),
            preferred_planning_structures=_json_loads(row["preferred_planning_structures_json"], []),
        )

    def _get_affective_state(self, profile_id: UUID) -> AffectiveState:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM affective_state WHERE profile_id = ?",
                (str(profile_id),),
            ).fetchone()
        if row is None:
            return AffectiveState()
        return AffectiveState(
            stress_level=row["stress_level"],
            energy_level=row["energy_level"],
            cognitive_load=row["cognitive_load"],
            social_energy=row["social_energy"],
            emotional_regulation=row["emotional_regulation"],
            executive_function=row["executive_function"],
        )

    def _get_episodic_memory(self, profile_id: UUID) -> list[PersonalAssistantEpisodicMemory]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodic_memory WHERE profile_id = ? ORDER BY occurred_on, created_at",
                (str(profile_id),),
            ).fetchall()
        episodes: list[PersonalAssistantEpisodicMemory] = []
        for row in rows:
            episodes.append(
                PersonalAssistantEpisodicMemory(
                    title=row["title"],
                    summary=row["summary"],
                    category=row["category"],
                    people=_json_loads(row["people_json"], []),
                    related_goals=_json_loads(row["related_goals_json"], []),
                    commitments=_json_loads(row["commitments_json"], []),
                    follow_ups=_json_loads(row["follow_ups_json"], []),
                    risks=_json_loads(row["risks_json"], []),
                    salience=row["salience"],
                    occurred_on=date.fromisoformat(row["occurred_on"]),
                )
            )
        return episodes

    def _upsert_working_memory(
        self,
        conn: sqlite3.Connection,
        profile_id: UUID,
        working: PersonalAssistantWorkingMemory,
        now: str,
    ) -> None:
        conn.execute(
            "INSERT INTO working_memory ("
            "profile_id, current_focus, active_goals_json, active_tasks_json, "
            "active_goals_completed_json, open_loops_json, open_loops_completed_json, "
            "pending_decisions_json, waiting_on_json, last_session_id, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(profile_id) DO UPDATE SET "
            "current_focus = excluded.current_focus, "
            "active_goals_json = excluded.active_goals_json, "
            "active_tasks_json = excluded.active_tasks_json, "
            "active_goals_completed_json = excluded.active_goals_completed_json, "
            "open_loops_json = excluded.open_loops_json, "
            "open_loops_completed_json = excluded.open_loops_completed_json, "
            "pending_decisions_json = excluded.pending_decisions_json, "
            "waiting_on_json = excluded.waiting_on_json, "
            "last_session_id = excluded.last_session_id, "
            "updated_at = excluded.updated_at",
            (
                str(profile_id),
                working.current_focus,
                _json_dumps(working.active_goals),
                _json_dumps(working.active_tasks),
                _json_dumps(working.active_goals_completed),
                _json_dumps(working.open_loops),
                _json_dumps(working.open_loops_completed),
                _json_dumps(working.pending_decisions),
                _json_dumps(working.waiting_on),
                str(working.last_session_id) if working.last_session_id else None,
                now,
            ),
        )

    def _upsert_semantic_memory(
        self,
        conn: sqlite3.Connection,
        profile_id: UUID,
        semantic: PersonalAssistantSemanticMemory,
        now: str,
    ) -> None:
        conn.execute(
            "INSERT INTO semantic_memory ("
            "profile_id, preferences_json, triggers_json, prefers_direct_language, "
            "dislikes_open_ended_questions, best_focus_time, sensitive_to_noise, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(profile_id) DO UPDATE SET "
            "preferences_json = excluded.preferences_json, "
            "triggers_json = excluded.triggers_json, "
            "prefers_direct_language = excluded.prefers_direct_language, "
            "dislikes_open_ended_questions = excluded.dislikes_open_ended_questions, "
            "best_focus_time = excluded.best_focus_time, "
            "sensitive_to_noise = excluded.sensitive_to_noise, "
            "updated_at = excluded.updated_at",
            (
                str(profile_id),
                _json_dumps(semantic.preferences),
                _json_dumps(semantic.triggers),
                int(semantic.prefers_direct_language),
                int(semantic.dislikes_open_ended_questions),
                semantic.best_focus_time,
                int(semantic.sensitive_to_noise),
                now,
            ),
        )

    def _upsert_procedural_memory(
        self,
        conn: sqlite3.Connection,
        profile_id: UUID,
        procedural: PersonalAssistantProceduralMemory,
        now: str,
    ) -> None:
        conn.execute(
            "INSERT INTO procedural_memory ("
            "profile_id, successful_interventions_json, routines_that_worked_json, "
            "effective_grouping_strategies_json, preferred_planning_structures_json, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(profile_id) DO UPDATE SET "
            "successful_interventions_json = excluded.successful_interventions_json, "
            "routines_that_worked_json = excluded.routines_that_worked_json, "
            "effective_grouping_strategies_json = excluded.effective_grouping_strategies_json, "
            "preferred_planning_structures_json = excluded.preferred_planning_structures_json, "
            "updated_at = excluded.updated_at",
            (
                str(profile_id),
                _json_dumps(procedural.successful_interventions),
                _json_dumps(procedural.routines_that_worked),
                _json_dumps(procedural.effective_grouping_strategies),
                _json_dumps(procedural.preferred_planning_structures),
                now,
            ),
        )

    def _upsert_affective_state(
        self,
        conn: sqlite3.Connection,
        profile_id: UUID,
        affective: AffectiveState,
        now: str,
    ) -> None:
        conn.execute(
            "INSERT INTO affective_state ("
            "profile_id, stress_level, energy_level, cognitive_load, social_energy, "
            "emotional_regulation, executive_function, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(profile_id) DO UPDATE SET "
            "stress_level = excluded.stress_level, "
            "energy_level = excluded.energy_level, "
            "cognitive_load = excluded.cognitive_load, "
            "social_energy = excluded.social_energy, "
            "emotional_regulation = excluded.emotional_regulation, "
            "executive_function = excluded.executive_function, "
            "updated_at = excluded.updated_at",
            (
                str(profile_id),
                affective.stress_level,
                affective.energy_level,
                affective.cognitive_load,
                affective.social_energy,
                affective.emotional_regulation,
                affective.executive_function,
                now,
            ),
        )

    def _replace_episodic_memory(
        self,
        conn: sqlite3.Connection,
        profile_id: UUID,
        episodes: list[PersonalAssistantEpisodicMemory],
        last_session_id: Optional[UUID],
        now: str,
    ) -> None:
        old_rows = conn.execute(
            "SELECT episode_id, title, summary, occurred_on FROM episodic_memory WHERE profile_id = ?",
            (str(profile_id),),
        ).fetchall()
        old_ids = {
            (row["title"], row["summary"], row["occurred_on"]): row["episode_id"]
            for row in old_rows
        }
        conn.execute("DELETE FROM episodic_memory WHERE profile_id = ?", (str(profile_id),))
        for episode in episodes:
            occurred_on = (episode.occurred_on or date.today()).isoformat()
            episode_id = old_ids.get((episode.title, episode.summary, occurred_on), str(uuid4()))
            conn.execute(
                "INSERT INTO episodic_memory ("
                "episode_id, profile_id, last_session_id, title, summary, category, people_json, "
                "related_goals_json, commitments_json, follow_ups_json, risks_json, salience, occurred_on, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    str(profile_id),
                    str(last_session_id) if last_session_id else None,
                    episode.title,
                    episode.summary,
                    episode.category,
                    _json_dumps(episode.people),
                    _json_dumps(episode.related_goals),
                    _json_dumps(episode.commitments),
                    _json_dumps(episode.follow_ups),
                    _json_dumps(episode.risks),
                    episode.salience,
                    occurred_on,
                    now,
                    now,
                ),
            )


def create_default_db(db_path: str | Path = DEFAULT_DB_PATH) -> PersonalAssistantDatabase:
    return PersonalAssistantDatabase(db_path)


def seed_default_profile(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    name: str = DEFAULT_PROFILE_NAME,
) -> PersonalAssistantDatabase:
    db = PersonalAssistantDatabase(db_path)
    db.set_profile_name(name)
    db.get_personal_assistant_memory()
    return db


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(raw: str, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ddc_review_item_from_row(row: sqlite3.Row) -> DDCReviewItem:
    return DDCReviewItem(
        review_id=row["review_id"],
        profile_id=UUID(row["profile_id"]),
        category=row["category"],
        risk=row["risk"],
        status=row["status"],
        source_task=row["source_task"],
        missing_context=row["missing_context"],
        proposed_memory=row["proposed_memory"],
        reason=row["reason"],
        session_id=UUID(row["session_id"]) if row["session_id"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _ddc_cycle_log_from_row(row: sqlite3.Row) -> DDCCycleLog:
    return DDCCycleLog(
        cycle_id=row["cycle_id"],
        review_id=row["review_id"],
        profile_id=UUID(row["profile_id"]),
        source_task=row["source_task"],
        category=row["category"],
        action=row["action"],
        promoted_memory=row["promoted_memory"],
        created_at=row["created_at"],
    )


def _planner_continuation_from_row(row: sqlite3.Row) -> PlannerContinuation:
    return PlannerContinuation(
        continuation_id=row["continuation_id"],
        profile_id=UUID(row["profile_id"]),
        original_user_task=row["original_user_task"],
        planner_output_json=row["planner_output_json"],
        blocking_questions_json=row["blocking_questions_json"],
        status=row["status"],
        session_id=UUID(row["session_id"]) if row["session_id"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _entity_review_item_from_row(row: sqlite3.Row) -> EntityReviewItem:
    return EntityReviewItem(
        review_id=row["review_id"],
        profile_id=UUID(row["profile_id"]),
        name=row["name"],
        entity_type=row["entity_type"],
        description=row["description"],
        aliases=_json_loads(row["aliases_json"], []),
        reason=row["reason"],
        risk=row["risk"],
        status=row["status"],
        source_task=row["source_task"],
        session_id=UUID(row["session_id"]) if row["session_id"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _context_entity_from_row(row: sqlite3.Row) -> ContextEntity:
    return ContextEntity(
        entity_id=row["entity_id"],
        profile_id=UUID(row["profile_id"]),
        name=row["name"],
        entity_type=row["entity_type"],
        description=row["description"],
        aliases=_json_loads(row["aliases_json"], []),
        source_task=row["source_task"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
