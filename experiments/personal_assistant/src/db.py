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
