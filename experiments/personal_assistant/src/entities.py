from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


EntityRiskLevel = Literal["low", "medium", "high"]
EntityReviewStatus = Literal["pending", "approved", "rejected"]


class ContextEntity(BaseModel):
    entity_id: str = Field(default_factory=lambda: f"ent_{uuid4().hex[:12]}")
    profile_id: UUID
    name: str
    entity_type: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    source_task: str
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class EntityProposal(BaseModel):
    name: str
    entity_type: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    reason: str
    risk: EntityRiskLevel = "medium"


class EntityAnalysisResult(BaseModel):
    entities: list[EntityProposal] = Field(default_factory=list)


class EntityReviewItem(BaseModel):
    review_id: str = Field(default_factory=lambda: f"ent_review_{uuid4().hex[:12]}")
    profile_id: UUID
    name: str
    entity_type: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    reason: str
    risk: EntityRiskLevel = "medium"
    status: EntityReviewStatus = "pending"
    source_task: str
    session_id: Optional[UUID] = None
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class ContextEntityService:
    def __init__(self, db, knowledge_base_dir: str | Path | None = None) -> None:
        self.db = db
        self.exporter = KnowledgeBaseExporter(
            knowledge_base_dir or Path(db.db_path).parent / "knowledge_base"
        )

    def create_review_items(
        self,
        *,
        profile_id: UUID,
        source_task: str,
        proposals: list[EntityProposal],
        session_id: Optional[UUID] = None,
    ) -> list[EntityReviewItem]:
        existing_keys = self._known_entity_keys(profile_id)
        items: list[EntityReviewItem] = []
        for proposal in proposals:
            if not proposal.name.strip() or not proposal.description.strip():
                continue
            proposal_keys = _entity_keys(proposal.name, proposal.aliases)
            if proposal_keys & existing_keys:
                continue
            item = EntityReviewItem(
                profile_id=profile_id,
                name=proposal.name.strip(),
                entity_type=_normalize_type(proposal.entity_type),
                description=proposal.description.strip(),
                aliases=_clean_aliases(proposal.aliases),
                reason=proposal.reason.strip(),
                risk=proposal.risk,
                source_task=source_task,
                session_id=session_id,
            )
            items.append(item)
            existing_keys.update(proposal_keys)
        if items:
            self.db.insert_entity_review_items(items)
        return items

    def pending(self, profile_id: UUID) -> list[EntityReviewItem]:
        return self.db.list_entity_review_items(profile_id, status="pending")

    def show(self, profile_id: UUID, review_id: str) -> Optional[EntityReviewItem]:
        return self.db.get_entity_review_item(profile_id, review_id)

    def approve(self, profile_id: UUID, review_id: str) -> ContextEntity:
        item = self._require_pending(profile_id, review_id)
        entity = self.db.insert_context_entity(_entity_from_review(item))
        self.db.set_entity_review_status(profile_id, review_id, "approved")
        self.export_type(profile_id, entity.entity_type)
        self.db.increment_context_revision(profile_id)
        return entity

    def reject(self, profile_id: UUID, review_id: str) -> EntityReviewItem:
        self._require_pending(profile_id, review_id)
        return self.db.set_entity_review_status(profile_id, review_id, "rejected")

    def revise(
        self,
        profile_id: UUID,
        review_id: str,
        *,
        name: str,
        entity_type: str,
        description: str,
        aliases: list[str],
    ) -> ContextEntity:
        self._require_pending(profile_id, review_id)
        item = self.db.update_entity_review_item(
            profile_id,
            review_id,
            name=name.strip(),
            entity_type=_normalize_type(entity_type),
            description=description.strip(),
            aliases=_clean_aliases(aliases),
        )
        entity = self.db.insert_context_entity(_entity_from_review(item))
        self.db.set_entity_review_status(profile_id, review_id, "approved")
        self.export_type(profile_id, entity.entity_type)
        self.db.increment_context_revision(profile_id)
        return entity

    def approved(self, profile_id: UUID, *, limit: int = 100) -> list[ContextEntity]:
        return self.db.list_context_entities(profile_id, limit=limit)

    def export_type(self, profile_id: UUID, entity_type: str) -> None:
        entities = self.db.list_context_entities(profile_id, entity_type=entity_type, limit=1000)
        self.exporter.export_type(entity_type, entities)

    def export_all(self, profile_id: UUID) -> None:
        grouped: dict[str, list[ContextEntity]] = defaultdict(list)
        for entity in self.db.list_context_entities(profile_id, limit=1000):
            grouped[entity.entity_type].append(entity)
        self.exporter.export_all(grouped)

    def _require_pending(self, profile_id: UUID, review_id: str) -> EntityReviewItem:
        item = self.db.get_entity_review_item(profile_id, review_id)
        if item is None:
            raise ValueError(f"Unknown entity review item: {review_id}")
        if item.status != "pending":
            raise ValueError(f"Entity review item is not pending: {review_id}")
        return item

    def _known_entity_keys(self, profile_id: UUID) -> set[str]:
        keys: set[str] = set()
        for entity in self.db.list_context_entities(profile_id, limit=1000):
            keys.update(_entity_keys(entity.name, entity.aliases))
        for item in self.db.list_entity_review_items(profile_id, status="pending", limit=1000):
            keys.update(_entity_keys(item.name, item.aliases))
        return keys


class KnowledgeBaseExporter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def export_all(self, grouped_entities: dict[str, list[ContextEntity]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for entity_type, entities in grouped_entities.items():
            self.export_type(entity_type, entities)

    def export_type(self, entity_type: str, entities: list[ContextEntity]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        normalized_type = _normalize_type(entity_type)
        path = self.root / f"{normalized_type}.md"
        updated_at = _utc_now()
        lines = [
            "---",
            f"entity_type: {normalized_type}",
            "generated_from: sqlite",
            f"updated_at: {updated_at}",
            f"entity_count: {len(entities)}",
            "---",
            "",
            f"# {normalized_type.replace('_', ' ').title()}",
            "",
        ]
        for entity in sorted(entities, key=lambda item: item.name.lower()):
            lines.extend(_render_entity(entity))
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path


def _entity_from_review(item: EntityReviewItem) -> ContextEntity:
    return ContextEntity(
        profile_id=item.profile_id,
        name=item.name,
        entity_type=item.entity_type,
        description=item.description,
        aliases=item.aliases,
        source_task=item.source_task,
    )


def _render_entity(entity: ContextEntity) -> list[str]:
    aliases = ", ".join(entity.aliases)
    lines = [
        f"## {entity.name}",
        "",
        "```yaml",
        f"entity_id: {entity.entity_id}",
        f"name: {entity.name}",
        f"type: {entity.entity_type}",
        f"aliases: [{aliases}]",
        f"updated_at: {entity.updated_at}",
        f"source_task: {entity.source_task}",
        "```",
        "",
        entity.description,
        "",
    ]
    return lines


def _entity_keys(name: str, aliases: list[str]) -> set[str]:
    return {_normalize_key(value) for value in [name, *aliases] if _normalize_key(value)}


def _normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_type(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "entity"


def _clean_aliases(values: list[str]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = value.strip()
        key = _normalize_key(alias)
        if alias and key not in seen:
            aliases.append(alias)
            seen.add(key)
    return aliases
