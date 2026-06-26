from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_priming_config
from .db import DEFAULT_PROFILE_ID, JournalDatabase
from .models import Entity, Episode, SemanticFact, SemanticFactHint, Thought
from .runtime import configure_quiet_runtime


configure_quiet_runtime()


@dataclass
class PrimingHit:
    """A fast familiarity match returned by ChromaDB before deeper memory recall."""

    source_id: str
    memory_type: str
    document: str
    distance: float | None = None
    metadata: dict[str, Any] | None = None


class PrimingStore:
    """Local ChromaDB index used for Tulving-style perceptual priming.

    mem0 remains the durable episodic and semantic memory store. ChromaDB is
    only the fast recognition layer: it answers whether the current input is
    familiar and which records should shape the heavier recall step.
    """

    def __init__(self, *, path: str | None = None, collection: str | None = None) -> None:
        configure_quiet_runtime()
        config = load_priming_config(path=path, collection=collection)
        self.path = Path(config.path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        logging.getLogger("chromadb.telemetry").disabled = True
        logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
        warnings.filterwarnings(
            "ignore",
            message=".*model_fields.*",
            category=DeprecationWarning,
            module="chromadb.*",
        )

        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=config.collection)

    @classmethod
    def from_env(cls) -> "PrimingStore":
        """Create a priming store from environment-backed configuration."""

        return cls()

    def index_entity(self, entity: Entity) -> None:
        """Index an entity as natural language for semantic familiarity search."""

        self._upsert(
            source_id=entity.entity_id,
            memory_type="entity",
            document=render_entity_for_priming(entity),
            metadata={
                "profile_id": entity.profile_id,
                "entity_type": entity.type,
                "canonical_name": entity.canonical_name,
            },
        )

    def index_thought(self, thought: Thought) -> None:
        """Index a captured thought, including later organization labels."""

        self._upsert(
            source_id=thought.thought_id,
            memory_type="thought",
            document=render_thought_for_priming(thought),
            metadata={
                "profile_id": thought.profile_id,
                "thought_type": thought.thought_type,
                "thought": thought.thought or "",
                "tags": ", ".join(thought.tags),
                "entity_refs": ", ".join(thought.entity_refs),
                "created_at": thought.created_at,
            },
        )

    def index_episode(self, episode: Episode) -> None:
        """Index an episodic event as a compact, timestamped prose record."""

        self._upsert(
            source_id=episode.episode_id,
            memory_type="episode",
            document=render_episode_for_priming(episode),
            metadata={
                "profile_id": episode.profile_id,
                "event_type": episode.event_type,
                "thought_id": episode.thought_id or "",
                "thought": episode.thought or "",
                "tags": ", ".join(episode.tags),
                "entity_refs": ", ".join(episode.entity_refs),
                "occurred_at": episode.occurred_at,
            },
        )

    def index_semantic_fact(self, fact: SemanticFact) -> None:
        """Index a promoted semantic fact after it has been written to mem0."""

        self._upsert(
            source_id=fact.fact_id,
            memory_type="semantic_fact",
            document=render_semantic_fact_for_priming(fact),
            metadata={
                "profile_id": fact.profile_id,
                "subject_entity_id": fact.subject_entity_id,
                "predicate": fact.predicate,
                "source_episode_refs": ", ".join(fact.source_episode_refs),
                "created_at": fact.created_at,
            },
        )

    def index_promoted_hint(self, hint: SemanticFactHint) -> None:
        """Rebuild a semantic-fact priming entry from the local promoted hint."""

        self._upsert(
            source_id=hint.hint_id,
            memory_type="semantic_fact",
            document=render_semantic_hint_for_priming(hint),
            metadata={
                "profile_id": hint.profile_id,
                "subject_entity_id": hint.subject_entity_id,
                "predicate": hint.predicate,
                "source_episode_refs": ", ".join(hint.source_episode_refs),
                "created_at": hint.created_at,
            },
        )

    def search(self, query: str, *, profile_id: str = DEFAULT_PROFILE_ID, limit: int = 8) -> list[PrimingHit]:
        """Return familiar records for a query using ChromaDB vector search."""

        clean = " ".join(query.split())
        if not clean:
            return []
        count = self._collection.count()
        if count <= 0:
            return []
        result = self._collection.query(
            query_texts=[clean],
            n_results=min(limit, count),
            where={"profile_id": profile_id},
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        hits: list[PrimingHit] = []
        for source_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata if isinstance(metadata, dict) else {}
            hits.append(
                PrimingHit(
                    source_id=str(metadata.get("source_id") or source_id),
                    memory_type=str(metadata.get("memory_type") or "memory"),
                    document=str(document),
                    distance=float(distance) if isinstance(distance, (int, float)) else None,
                    metadata=metadata,
                )
            )
        return hits

    def rebuild(self, db: JournalDatabase, *, profile_id: str = DEFAULT_PROFILE_ID, limit: int = 5000) -> int:
        """Rebuild the priming index from local SQLite records.

        This is useful after manually editing the database, adding priming to an
        existing journal, or changing the prose renderers.
        """

        count = 0
        for entity in db.list_entities(profile_id, limit=limit):
            self.index_entity(entity)
            count += 1
        for thought in db.list_thoughts(profile_id, limit=limit):
            self.index_thought(thought)
            count += 1
        for episode in db.list_episodes(profile_id, limit=limit):
            self.index_episode(episode)
            count += 1
        for hint in db.list_semantic_fact_hints(profile_id, status="promoted", limit=limit):
            self.index_promoted_hint(hint)
            count += 1
        return count

    def _upsert(self, *, source_id: str, memory_type: str, document: str, metadata: dict[str, Any]) -> None:
        # Chroma metadata must be scalar, so list-like fields are rendered as
        # comma-separated strings while the full natural-language text remains
        # in the embedded document.
        clean_metadata = {
            **_clean_metadata(metadata),
            "source_id": source_id,
            "memory_type": memory_type,
        }
        self._collection.upsert(
            ids=[f"{memory_type}:{source_id}"],
            documents=[document],
            metadatas=[clean_metadata],
        )


def render_entity_for_priming(entity: Entity) -> str:
    """Render an entity into prose for embedding."""

    alias_text = f" Aliases: {', '.join(entity.aliases)}." if entity.aliases else ""
    description = _sentence(entity.description or "No description has been added yet.")
    return f"{entity.canonical_name} is a {entity.type} in the user's journal. {description}{alias_text}"


def render_thought_for_priming(thought: Thought) -> str:
    """Render a captured thought into prose for embedding."""

    label = f" about {thought.thought}" if thought.thought else ""
    tags = f" Tags: {', '.join(thought.tags)}." if thought.tags else ""
    return f"The user captured a {thought.thought_type} thought{label}: {thought.body}.{tags}"


def render_episode_for_priming(episode: Episode) -> str:
    """Render an episodic memory row into timestamped prose for embedding."""

    parts = [
        f"On {episode.occurred_at[:10]}, the user recorded a {episode.event_type} event: {episode.description}.",
        f"Significance: {_sentence(episode.significance)}" if episode.significance else "",
        f"Thought label: {episode.thought}." if episode.thought else "",
        f"Tags: {', '.join(episode.tags)}." if episode.tags else "",
    ]
    return " ".join(part for part in parts if part)


def render_semantic_fact_for_priming(fact: SemanticFact) -> str:
    """Render a promoted semantic fact into prose for embedding."""

    return (
        f"The journal has a semantic fact about {fact.subject_entity_id}: "
        f"{fact.predicate} is {fact.value}. Confidence: {fact.confidence_score:.2f}."
    )


def render_semantic_hint_for_priming(hint: SemanticFactHint) -> str:
    """Render a promoted local hint as a semantic fact for rebuilding Chroma."""

    return (
        f"The journal has a semantic fact about {hint.subject_entity_id}: "
        f"{hint.predicate} is {hint.value}. Support count: {hint.support_count}."
    )


def _clean_metadata(value: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            cleaned[key] = item
        else:
            cleaned[key] = str(item)
    return cleaned


def _sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."
