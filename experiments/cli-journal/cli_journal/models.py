from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


EntityType = Literal["person", "org", "project", "tool", "place", "concept", "artifact", "entity"]
ThoughtType = Literal["idea", "work", "task", "health", "decision", "commitment", "risk", "extras"]
SemanticPredicate = Literal[
    "preference",
    "constraint",
    "work_style",
    "tool_usage",
    "relationship",
    "project_fact",
    "recurring_need",
    "goal",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class Entity:
    profile_id: str
    canonical_name: str
    type: EntityType = "entity"
    entity_id: str = field(default_factory=lambda: new_id("ent"))
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    first_seen: str = field(default_factory=utc_now_iso)
    last_referenced: str = field(default_factory=utc_now_iso)
    confidence_score: float = 0.6


@dataclass
class Thought:
    profile_id: str
    body: str
    thought_type: ThoughtType = "extras"
    thought_id: str = field(default_factory=lambda: new_id("thought"))
    thought: str | None = None
    tags: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class Episode:
    profile_id: str
    description: str
    event_type: str = "thought"
    episode_id: str = field(default_factory=lambda: new_id("ep"))
    occurred_at: str = field(default_factory=utc_now_iso)
    significance: str = ""
    thought_id: str | None = None
    thought: str | None = None
    tags: list[str] = field(default_factory=list)
    entity_refs: list[str] = field(default_factory=list)
    salience_score: float = 0.5
    consolidated_at: str | None = None


@dataclass
class SemanticFact:
    profile_id: str
    subject_entity_id: str
    predicate: SemanticPredicate
    value: str
    fact_id: str = field(default_factory=lambda: new_id("fact"))
    confidence_score: float = 0.7
    source_episode_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_confirmed_at: str | None = None


@dataclass
class SemanticFactHint:
    profile_id: str
    subject_entity_id: str
    predicate: SemanticPredicate
    value: str
    hint_id: str = field(default_factory=lambda: new_id("hint"))
    confidence_score: float = 0.6
    source_episode_refs: list[str] = field(default_factory=list)
    support_count: int = 1
    status: str = "pending"
    rationale: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class OrganizationJob:
    profile_id: str
    thought_id: str
    episode_id: str
    job_id: str = field(default_factory=lambda: new_id("org"))
    status: str = "pending"
    attempts: int = 0
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class JournalSession:
    profile_id: str
    session_id: str = field(default_factory=lambda: new_id("session"))
    name: str = "Journal chat"
    status: str = "active"
    started_at: str = field(default_factory=utc_now_iso)
    last_active_at: str = field(default_factory=utc_now_iso)
    rolling_summary: str = ""
    active_thought_ids: list[str] = field(default_factory=list)
    active_entity_ids: list[str] = field(default_factory=list)
    recent_queries: list[str] = field(default_factory=list)
    last_exchange: dict[str, str] | None = None
