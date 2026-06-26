from __future__ import annotations

import re

from .async_utils import run_async
from .db import DEFAULT_PROFILE_ID, JournalDatabase
from .llm import ConsolidationAbstractionClient, OrganizationResult, ThoughtOrganizerClient
from .mem0_store import JournalMem0Store
from .models import Entity, Episode, OrganizationJob, SemanticFact, SemanticFactHint, Thought, utc_now_iso
from .priming import PrimingStore
from .runtime import captured_output, quiet_third_party_output


TYPE_WORDS = {"person", "org", "project", "tool", "place", "concept", "artifact", "entity"}
MENTION_RE = re.compile(r"(?<![A-Za-z0-9._%+-])@([A-Za-z](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?)\b")

class JournalOrganizer:
    """Turns raw text into thoughts, episodic events, entity links, and semantic facts."""

    def __init__(
        self,
        db: JournalDatabase,
        profile_id: str = DEFAULT_PROFILE_ID,
        llm: ThoughtOrganizerClient | None = None,
        consolidator: ConsolidationAbstractionClient | None = None,
        memory: JournalMem0Store | None = None,
        priming: PrimingStore | None = None,
    ) -> None:
        self.db = db
        self.profile_id = profile_id
        self.llm = llm or ThoughtOrganizerClient()
        self.consolidator = consolidator or ConsolidationAbstractionClient()
        self._memory = memory
        self._priming = priming
        self.last_memory_error: str | None = None
        self.last_priming_error: str | None = None

    @property
    def memory(self) -> JournalMem0Store:
        if self._memory is None:
            self._memory = JournalMem0Store.from_env()
        return self._memory

    @property
    def priming(self) -> PrimingStore:
        if self._priming is None:
            self._priming = PrimingStore.from_env()
        return self._priming

    def capture(
        self,
        text: str,
        *,
        thought: str | None = None,
        thought_type: str | None = None,
        explicit_entities: list[str] | None = None,
    ) -> tuple[Thought, Episode]:
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("Thought cannot be empty.")

        entities = self._resolve_entities(clean, explicit_entities or [])
        tags: list[str] = []
        kind = thought_type or "extras"
        entity_refs = [entity.entity_id for entity in entities]

        captured = self.db.add_thought(
            Thought(
                profile_id=self.profile_id,
                thought_type=kind,
                body=clean,
                thought=thought,
                tags=tags,
                entity_refs=entity_refs,
            )
        )
        episode = self.db.add_episode(
            Episode(
                profile_id=self.profile_id,
                event_type=kind,
                description=clean,
                significance=self._significance(clean, kind, tags),
                thought_id=captured.thought_id,
                thought=thought,
                tags=tags,
                entity_refs=entity_refs,
                salience_score=self._salience(clean, kind),
            )
        )
        self.db.create_organization_job(
            OrganizationJob(
                profile_id=self.profile_id,
                thought_id=captured.thought_id,
                episode_id=episode.episode_id,
            )
        )
        self._index_priming(thought=captured, episode=episode, entities=entities)
        self.last_memory_error = None
        output = ""
        try:
            with quiet_third_party_output() as output_streams:
                run_async(self.memory.add_episode(episode, user_id=self.profile_id))
            output = captured_output(output_streams)
        except Exception as exc:
            output = captured_output(output_streams) if "output_streams" in locals() else ""
            self.last_memory_error = str(exc)
            self.db.add_log(
                profile_id=self.profile_id,
                level="warning",
                source="mem0.add_episode",
                message=str(exc),
                context={"episode_id": episode.episode_id, "thought_id": captured.thought_id},
            )
        self._log_third_party_output("mem0.add_episode.output", output)
        return captured, episode

    def organize_pending(self, *, limit: int = 10) -> int:
        """Process pending LLM organization jobs."""
        count = 0
        for job in self.db.list_organization_jobs(self.profile_id, status="pending", limit=limit):
            self.db.mark_organization_job_running(job)
            try:
                thought = self.db.get_thought(self.profile_id, job.thought_id)
                episode = self.db.get_episode(self.profile_id, job.episode_id)
                if thought is None or episode is None:
                    raise RuntimeError("Organization job points to a missing thought or episode.")
                result = self.llm.organize(thought.body)
                self._apply_organization(thought, episode, result)
                self.db.complete_organization_job(job.job_id)
                count += 1
            except Exception as exc:
                self.db.fail_organization_job(job.job_id, str(exc))
                raise
        return count

    def add_entity(
        self,
        name: str,
        entity_type: str = "entity",
        description: str = "",
        aliases: list[str] | None = None,
    ) -> Entity:
        normalized_type = entity_type if entity_type in TYPE_WORDS else "entity"
        entity = self.db.upsert_entity(
            Entity(
                profile_id=self.profile_id,
                canonical_name=name.strip(),
                type=normalized_type,
                description=description.strip(),
                aliases=[alias.strip() for alias in aliases or [] if alias.strip()],
                confidence_score=0.9,
            )
        )
        self._index_priming(entity=entity)
        return entity

    def answer(self, query: str, *, limit: int = 5) -> str:
        thought_hits = self.db.search_thoughts(self.profile_id, query, limit=limit)
        entity_hits = self._search_entities(query, limit=limit)
        memory_hits = []
        self.last_memory_error = None
        try:
            with quiet_third_party_output() as output_streams:
                memory_hits = run_async(self.memory.recall(query, user_id=self.profile_id, limit=limit))
            self._log_third_party_output("mem0.recall.output", captured_output(output_streams))
        except Exception as exc:
            self.last_memory_error = str(exc)
            self.db.add_log(
                profile_id=self.profile_id,
                level="warning",
                source="mem0.recall",
                message=str(exc),
                context={"query": query},
            )

        if not thought_hits and not memory_hits and not entity_hits:
            return "I do not have matching journal memory yet."

        lines = ["Here is what I found:"]
        if memory_hits:
            lines.append("")
            lines.append("Memory")
            for item in memory_hits:
                text = item.get("memory") or item.get("text") or item.get("content") or ""
                score = item.get("score")
                suffix = f" ({float(score):.1f})" if isinstance(score, (int, float)) else ""
                lines.append(f"- {str(text).strip()}{suffix}")
        if thought_hits:
            lines.append("")
            lines.append("Thoughts")
            for thought, score in thought_hits:
                label = f" [{thought.thought}]" if thought.thought else ""
                lines.append(f"- {thought.created_at[:10]}{label}: {thought.body} ({score:.1f})")
        if entity_hits:
            lines.append("")
            lines.append("Entities")
            for entity in entity_hits:
                description = f" - {entity.description}" if entity.description else ""
                lines.append(f"- {entity.canonical_name} ({entity.type}){description}")
        return "\n".join(lines)

    def consolidate(self, *, min_support: int = 3, limit: int = 200) -> list[SemanticFact]:
        episodes = self.db.list_episodes(self.profile_id, unconsolidated=True, limit=limit)
        if not episodes:
            return []
        episode_by_id = {episode.episode_id: episode for episode in episodes}
        grouping = self.consolidator.group_episodes(episodes=episodes)

        created: list[SemanticFact] = []
        consumed: list[str] = []
        for group in grouping.groups:
            items = [episode_by_id[episode_id] for episode_id in group.episode_ids if episode_id in episode_by_id]
            if len(items) < min_support:
                continue
            subject = (group.entity_refs or _entity_refs_from_episodes(items) or ["user"])[0]
            candidates = self._retrieve_candidate_semantic_facts(subject=subject, predicate=group.predicate_hint, episodes=items)
            abstraction = self.consolidator.abstract_semantic_fact(
                subject_entity_id=subject,
                predicate_hint=group.predicate_hint,
                episodes=items,
                candidate_facts=candidates,
            )
            if abstraction.action == "do_not_promote":
                continue
            if not abstraction.value.strip():
                continue
            hint = self._merge_or_create_hint(
                subject=subject,
                abstraction=abstraction,
                episodes=items,
            )
            if hint.support_count >= min_support and hint.status == "pending":
                promoted = self._promote_hint(hint)
                if promoted is not None:
                    created.append(promoted)
            consumed.extend(item.episode_id for item in items)

        self.db.mark_episodes_consolidated(self.profile_id, _unique(consumed))
        return created

    def _retrieve_candidate_semantic_facts(
        self,
        *,
        subject: str,
        predicate: str,
        episodes: list[Episode],
        limit: int = 12,
    ) -> list[dict]:
        query = " ".join(
            [
                subject,
                predicate,
                *[episode.description for episode in episodes[:5]],
                *[tag for episode in episodes[:5] for tag in episode.tags],
            ]
        )
        try:
            with quiet_third_party_output() as output_streams:
                result = run_async(
                    self.memory.recall(
                        query,
                        user_id=self.profile_id,
                        memory_type="semantic_fact",
                        limit=limit,
                    )
                )
            self._log_third_party_output("mem0.recall.output", captured_output(output_streams))
            return result
        except Exception as exc:
            self.last_memory_error = str(exc)
            self.db.add_log(
                profile_id=self.profile_id,
                level="warning",
                source="mem0.recall",
                message=str(exc),
                context={"operation": "candidate_semantic_facts"},
            )
            return []

    def _merge_or_create_hint(self, *, subject: str, abstraction, episodes: list[Episode]) -> SemanticFactHint:
        source_refs = _unique([episode.episode_id for episode in episodes])
        candidates = self._candidate_hints(subject=subject, predicate=abstraction.predicate)
        decision = self.consolidator.decide_hint_merge(
            abstraction=abstraction,
            candidate_hints=[_hint_to_candidate(hint) for hint in candidates],
        )
        if decision.action == "reject":
            return self.db.create_semantic_fact_hint(
                SemanticFactHint(
                    profile_id=self.profile_id,
                    subject_entity_id=subject,
                    predicate=abstraction.predicate,
                    value=abstraction.value.strip(),
                    confidence_score=decision.confidence_score,
                    source_episode_refs=source_refs,
                    support_count=len(source_refs),
                    status="rejected",
                    rationale=decision.rationale or abstraction.rationale,
                )
            )
        if decision.action == "merge_existing" and decision.target_hint_id:
            existing = self.db.get_semantic_fact_hint(self.profile_id, decision.target_hint_id)
            if existing is not None and existing.status == "pending":
                existing.value = (decision.merged_value or abstraction.value or existing.value).strip()
                existing.confidence_score = max(existing.confidence_score, decision.confidence_score, abstraction.confidence_score)
                existing.source_episode_refs = _unique([*existing.source_episode_refs, *source_refs])
                existing.support_count = len(existing.source_episode_refs)
                existing.rationale = decision.rationale or abstraction.rationale or existing.rationale
                return self.db.update_semantic_fact_hint(existing)
        return self.db.create_semantic_fact_hint(
            SemanticFactHint(
                profile_id=self.profile_id,
                subject_entity_id=subject,
                predicate=abstraction.predicate,
                value=abstraction.value.strip(),
                confidence_score=abstraction.confidence_score,
                source_episode_refs=source_refs,
                support_count=len(source_refs),
                status="pending",
                rationale=abstraction.rationale,
            )
        )

    def _candidate_hints(self, *, subject: str, predicate: str) -> list[SemanticFactHint]:
        scoped = self.db.list_semantic_fact_hints(
            self.profile_id,
            subject_entity_id=subject,
            predicate=predicate,
            limit=20,
        )
        broader = self.db.list_semantic_fact_hints(self.profile_id, limit=20)
        seen: set[str] = set()
        result: list[SemanticFactHint] = []
        for hint in [*scoped, *broader]:
            if hint.hint_id in seen:
                continue
            seen.add(hint.hint_id)
            result.append(hint)
        return result

    def _promote_hint(self, hint: SemanticFactHint) -> SemanticFact | None:
        fact = SemanticFact(
            profile_id=self.profile_id,
            subject_entity_id=hint.subject_entity_id,
            predicate=hint.predicate,
            value=hint.value,
            confidence_score=hint.confidence_score,
            source_episode_refs=hint.source_episode_refs,
            last_confirmed_at=utc_now_iso(),
        )
        self.last_memory_error = None
        try:
            with quiet_third_party_output() as output_streams:
                stored = run_async(self.memory.add_semantic_fact(fact, user_id=self.profile_id))
            self._log_third_party_output("mem0.add_semantic_fact.output", captured_output(output_streams))
        except Exception as exc:
            self.last_memory_error = str(exc)
            self.db.add_log(
                profile_id=self.profile_id,
                level="warning",
                source="mem0.add_semantic_fact",
                message=str(exc),
                context={"hint_id": hint.hint_id},
            )
            return None
        self.db.mark_semantic_fact_hint_promoted(self.profile_id, hint.hint_id)
        self._index_priming(fact=fact)
        return stored

    def _resolve_entities(self, text: str, explicit_entities: list[str]) -> list[Entity]:
        entities: list[Entity] = []

        for raw in explicit_entities:
            name, entity_type = _split_entity_arg(raw)
            entities.append(self.add_entity(name, entity_type))

        known = self.db.list_entities(self.profile_id, limit=1000)
        lowered = text.lower()
        for entity in known:
            names = [entity.canonical_name, *entity.aliases]
            if any(name.lower() in lowered for name in names if name):
                entities.append(self.db.upsert_entity(entity))

        for handle in _mention_handles(text):
            if any(_entity_has_name(entity, handle) for entity in entities):
                continue
            existing = self.db.find_entity_by_name(self.profile_id, handle)
            if existing is not None:
                entities.append(self.db.upsert_entity(_with_alias(existing, handle)))
                continue
            canonical_name = _canonical_name_from_handle(handle)
            existing = self.db.find_entity_by_name(self.profile_id, canonical_name)
            if existing is not None:
                entities.append(self.db.upsert_entity(_with_alias(existing, handle)))
                continue
            entities.append(self.add_entity(canonical_name, "person", aliases=[handle.lower()]))

        for name in _capitalized_candidates(text):
            if any(entity.canonical_name.lower() == name.lower() for entity in entities):
                continue
            existing = self.db.find_entity_by_name(self.profile_id, name)
            if existing is not None:
                entities.append(self.db.upsert_entity(existing))

        return _unique_entities(entities)

    def _search_entities(self, query: str, *, limit: int) -> list[Entity]:
        tokens = set(_tokens(query))
        results: list[Entity] = []
        for entity in self.db.list_entities(self.profile_id, limit=1000):
            fields = " ".join([entity.canonical_name, entity.type, entity.description, *entity.aliases]).lower()
            if any(token in fields for token in tokens):
                results.append(entity)
        return results[:limit]

    def _significance(self, text: str, thought_type: str, tags: list[str]) -> str:
        if not tags:
            return "Queued for LLM organization."
        if thought_type == "task":
            return "Captured as an action the user may need to revisit."
        if thought_type == "decision":
            return "Captured as a decision that may affect later work."
        if thought_type == "commitment":
            return "Captured as a commitment the user may need to honor."
        if thought_type == "risk":
            return "Captured as a risk or concern that may need follow-up."
        if "issue" in tags:
            return "Related to issue tracking or product work."
        return "Captured as a journal event that may be useful later."

    def _salience(self, text: str, thought_type: str) -> float:
        score = 0.45
        if thought_type in {"task", "decision", "commitment", "risk"}:
            score += 0.2
        if any(word in text.lower() for word in ["blocked", "urgent", "important", "deadline"]):
            score += 0.2
        return min(score, 0.95)

    def _apply_organization(self, thought: Thought, episode: Episode, result: OrganizationResult) -> None:
        self.db.update_thought_organization(
            self.profile_id,
            thought.thought_id,
            thought_type=result.thought_type,
            thought=result.thought,
            tags=result.tags,
        )
        self.db.update_episode_organization(
            self.profile_id,
            episode.episode_id,
            event_type=result.thought_type,
            thought=result.thought,
            tags=result.tags,
            significance=result.significance,
            salience_score=result.salience_score,
        )
        thought.thought_type = result.thought_type
        thought.thought = result.thought
        thought.tags = result.tags
        episode.event_type = result.thought_type
        episode.thought = result.thought
        episode.tags = result.tags
        episode.significance = result.significance
        episode.salience_score = result.salience_score
        self._index_priming(thought=thought, episode=episode)

    def _index_priming(
        self,
        *,
        thought: Thought | None = None,
        episode: Episode | None = None,
        entity: Entity | None = None,
        entities: list[Entity] | None = None,
        fact: SemanticFact | None = None,
    ) -> None:
        self.last_priming_error = None
        try:
            with quiet_third_party_output() as output_streams:
                store = self.priming
                if entity is not None:
                    store.index_entity(entity)
                for item in entities or []:
                    store.index_entity(item)
                if thought is not None:
                    store.index_thought(thought)
                if episode is not None:
                    store.index_episode(episode)
                if fact is not None:
                    store.index_semantic_fact(fact)
            self._log_third_party_output("priming.index.output", captured_output(output_streams))
        except Exception as exc:
            self.last_priming_error = str(exc)
            self.db.add_log(
                profile_id=self.profile_id,
                level="warning",
                source="priming.index",
                message=str(exc),
                context={},
            )

    def _log_third_party_output(self, source: str, output: str) -> None:
        if not output:
            return
        self.db.add_log(
            profile_id=self.profile_id,
            level="debug",
            source=source,
            message=output[:2000],
            context={},
        )

def _split_entity_arg(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        return raw.strip(), "entity"
    name, entity_type = raw.split(":", 1)
    return name.strip(), entity_type.strip().lower()


def _capitalized_candidates(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,2}\b", text)
    ignored = {"I", "The", "This", "That", "Need", "Remember", "Today", "Tomorrow"}
    return [candidate for candidate in candidates if candidate not in ignored]


def _mention_handles(text: str) -> list[str]:
    return _unique_strings(match.group(1) for match in MENTION_RE.finditer(text))


def _canonical_name_from_handle(handle: str) -> str:
    parts = [part for part in re.split(r"[_-]+", handle.strip()) if part]
    return " ".join(part[:1].upper() + part[1:] for part in parts)


def _entity_has_name(entity: Entity, name: str) -> bool:
    key = name.strip().lower()
    names = [entity.canonical_name, *entity.aliases]
    return any(value.lower() == key for value in names)


def _with_alias(entity: Entity, alias: str) -> Entity:
    clean = alias.strip().lower()
    if clean and all(existing.lower() != clean for existing in entity.aliases):
        entity.aliases = [*entity.aliases, clean]
    return entity


def _unique_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _unique_entities(entities: list[Entity]) -> list[Entity]:
    result: list[Entity] = []
    seen: set[str] = set()
    for entity in entities:
        if entity.entity_id in seen:
            continue
        seen.add(entity.entity_id)
        result.append(entity)
    return result


def _entity_refs_from_episodes(episodes: list[Episode]) -> list[str]:
    refs: list[str] = []
    for episode in episodes:
        refs.extend(episode.entity_refs)
    return _unique(refs)


def _hint_to_candidate(hint: SemanticFactHint) -> dict:
    return {
        "hint_id": hint.hint_id,
        "subject_entity_id": hint.subject_entity_id,
        "predicate": hint.predicate,
        "value": hint.value,
        "support_count": hint.support_count,
        "rationale": hint.rationale,
    }


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
