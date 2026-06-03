from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from experiments.personal_assistant.src.ddc import DDCReviewItem
from experiments.personal_assistant.src.entities import ContextEntity


@dataclass
class _RenderedContext:
    revision: int
    content: str


class PersonalContextRenderer:
    """Render approved SQLite-backed personal context into LLM-ready Markdown."""

    def __init__(self, db) -> None:
        self.db = db
        self._cache: dict[str, _RenderedContext] = {}

    def render_if_stale(self, profile_id: UUID) -> str:
        cache_key = str(profile_id)
        revision = self.db.get_context_revision(profile_id)
        cached = self._cache.get(cache_key)
        if cached and cached.revision == revision:
            return cached.content

        content = self._render(profile_id)
        self._cache[cache_key] = _RenderedContext(revision=revision, content=content)
        return content

    def _render(self, profile_id: UUID) -> str:
        memory = self.db.get_personal_assistant_memory(profile_id)
        approved_items = self.db.list_ddc_review_items(profile_id, status="approved", limit=100)
        approved_entities = self.db.list_context_entities(profile_id, limit=50)
        grouped: dict[str, list[DDCReviewItem]] = defaultdict(list)
        for item in approved_items:
            grouped[item.category].append(item)

        lines: list[str] = ["# Personal Context", ""]
        lines.extend(_section("Boundaries", grouped.pop("boundary", [])))
        lines.extend(_section("Schedule Rules", grouped.pop("schedule_rule", [])))
        lines.extend(_section("Communication Rules", grouped.pop("communication_rule", [])))
        lines.extend(_section("Decision Rules", grouped.pop("decision_rule", [])))
        lines.extend(_section("Topics And Concepts", grouped.pop("topic_concept", [])))
        lines.extend(_section("People", grouped.pop("person", [])))
        lines.extend(_section("Projects", grouped.pop("project", [])))
        lines.extend(_section("Commitments", grouped.pop("commitment", [])))
        lines.extend(_section("Recurring Workflows", grouped.pop("recurring_workflow", [])))
        lines.extend(_section("Tool Procedures", grouped.pop("tool_procedure", [])))
        lines.extend(_section("Preferences", grouped.pop("preference", [])))
        lines.extend(_entity_section(approved_entities))

        if memory.working.pending_decisions or memory.working.waiting_on:
            lines.extend(["## Active Operating Context", ""])
            if memory.working.pending_decisions:
                lines.append("Pending decisions:")
                lines.extend(f"- {decision}" for decision in memory.working.pending_decisions)
            if memory.working.waiting_on:
                lines.append("Waiting on:")
                lines.extend(f"- {item} -> {blocker}" for item, blocker in memory.working.waiting_on.items())
            lines.append("")

        if (
            not approved_items
            and not approved_entities
            and not memory.working.pending_decisions
            and not memory.working.waiting_on
        ):
            lines.append("No approved demand-driven personal context yet.")

        return "\n".join(lines).strip()


def _section(title: str, items: list[DDCReviewItem]) -> list[str]:
    if not items:
        return []
    lines = [f"## {title}", ""]
    for item in items:
        lines.extend(
            [
                "---",
                f"type: {item.category}",
                f"id: {item.review_id}",
                f"risk: {item.risk}",
                "status: approved",
                "source: ddc_review",
                "---",
                item.proposed_memory,
                "",
            ]
        )
    return lines


def _entity_section(items: list[ContextEntity]) -> list[str]:
    if not items:
        return []
    lines = ["## Approved Entities", ""]
    for entity in items:
        aliases = ", ".join(entity.aliases)
        lines.extend(
            [
                "---",
                f"name: {entity.name}",
                f"type: {entity.entity_type}",
                f"aliases: [{aliases}]",
                "source: knowledge_base",
                "---",
                entity.description,
                "",
            ]
        )
    return lines
