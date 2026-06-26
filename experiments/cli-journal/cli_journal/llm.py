from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent as StructuredLLMFunction

from .async_utils import run_async
from .config import ensure_gemini_provider_env, load_gemini_config
from .models import Episode, SemanticFact, SemanticPredicate


ThoughtType = Literal["idea", "work", "task", "health", "decision", "commitment", "risk", "extras"]
SemanticFactAbstractionAction = Literal["create_new", "update_existing", "do_not_promote"]


class OrganizationResult(BaseModel):
    """Validated structured output from Gemini's journal organizer."""

    thought_type: ThoughtType = Field(description="The best category for the captured thought.")
    thought: str | None = Field(default=None, description="Short topic label, or null if there is no useful label.")
    tags: list[str] = Field(min_length=1, max_length=6, description="One to six short lowercase tags.")
    significance: str = Field(min_length=1, max_length=240, description="One plain sentence explaining why this may matter later.")
    salience_score: float = Field(ge=0.0, le=1.0, description="Importance score from 0 to 1.")

    @field_validator("thought", mode="before")
    @classmethod
    def _clean_thought(cls, value):
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned[:80] or None

    @field_validator("tags", mode="before")
    @classmethod
    def _clean_tags(cls, value):
        if not isinstance(value, list):
            return ["general"]
        tags: list[str] = []
        for item in value:
            tag = str(item).strip().lower().replace(" ", "_")
            if tag and tag not in tags:
                tags.append(tag[:40])
        return tags[:6] or ["general"]

    @field_validator("significance", mode="before")
    @classmethod
    def _clean_significance(cls, value):
        cleaned = str(value or "").strip()
        return cleaned[:240] or "Organized by Gemini."


class ConsolidationGroup(BaseModel):
    """LLM-selected group of episodes that may support one semantic fact."""

    group_id: str = Field(default="", description="Short stable label for the group.")
    episode_ids: list[str] = Field(default_factory=list, description="Episode ids that support the same reusable signal.")
    entity_refs: list[str] = Field(default_factory=list, description="Entity ids referenced by the grouped episodes.")
    predicate_hint: SemanticPredicate = Field(default="recurring_need", description="Best initial predicate for retrieval.")
    rationale: str = Field(default="", max_length=240, description="Short operational reason for the group.")


class ConsolidationGroupingResult(BaseModel):
    groups: list[ConsolidationGroup] = Field(default_factory=list)


class SemanticFactAbstraction(BaseModel):
    """LLM decision for turning repeated episode evidence into semantic memory."""

    action: SemanticFactAbstractionAction = "create_new"
    target_fact_id: str | None = None
    predicate: SemanticPredicate
    value: str = Field(min_length=1, max_length=500)
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=240)


class SemanticFactHintMergeDecision(BaseModel):
    """LLM decision for merging a new semantic hint with existing pending hints."""

    action: Literal["merge_existing", "create_new", "reject"] = "create_new"
    target_hint_id: str | None = None
    merged_value: str | None = Field(default=None, max_length=500)
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=240)


class ThoughtOrganizerClient:
    """Client used by the background thought organizer. Gemini is the provider."""

    def __init__(
        self,
        *,
        model: str | None = None,
    ) -> None:
        config = load_gemini_config(model=model)
        ensure_gemini_provider_env(config)
        self._config = config
        self._function: StructuredLLMFunction | None = None

    def organize(self, text: str) -> OrganizationResult:
        if not self._config.api_key:
            raise RuntimeError("Set JOURNAL_GEMINI_API_KEY or GEMINI_API_KEY to run Gemini organization.")
        result = run_async(self._function_for().run(f"Thought:\n{text}"))
        return result.output

    def _function_for(self) -> StructuredLLMFunction:
        if self._function is None:
            self._function = StructuredLLMFunction(
                model=self._config.model,
                output_type=OrganizationResult,
                system_prompt=_ORGANIZER_SYSTEM_PROMPT,
            )
        return self._function


class ConsolidationAbstractionClient:
    """LLM-backed grouping and abstraction for semantic memory consolidation."""

    def __init__(self, *, model: str | None = None) -> None:
        config = load_gemini_config(model=model)
        ensure_gemini_provider_env(config)
        self._config = config
        self._group_function: StructuredLLMFunction | None = None
        self._abstraction_function: StructuredLLMFunction | None = None
        self._merge_function: StructuredLLMFunction | None = None

    def group_episodes(self, *, episodes: list[Episode]) -> ConsolidationGroupingResult:
        if not self._config.api_key:
            raise RuntimeError("Set JOURNAL_GEMINI_API_KEY or GEMINI_API_KEY to run consolidation.")
        evidence = "\n".join(
            (
                f"- id={episode.episode_id}; entities={','.join(episode.entity_refs) or 'none'}; "
                f"type={episode.event_type}; occurred_at={episode.occurred_at}; "
                f"description={episode.description}; significance={episode.significance}; "
                f"tags={','.join(episode.tags) or 'none'}"
            )
            for episode in episodes
        )
        result = run_async(
            self._group_function_for().run(
                f"Unconsolidated journal episodes:\n{evidence}\n\nReturn consolidation groups."
            )
        )
        return result.output

    def abstract_semantic_fact(
        self,
        *,
        subject_entity_id: str,
        predicate_hint: str,
        episodes: list[Episode],
        candidate_facts: list[dict[str, Any]],
    ) -> SemanticFactAbstraction:
        if not self._config.api_key:
            raise RuntimeError("Set JOURNAL_GEMINI_API_KEY or GEMINI_API_KEY to run consolidation.")
        evidence = "\n".join(
            (
                f"- id={episode.episode_id}; type={episode.event_type}; "
                f"occurred_at={episode.occurred_at}; description={episode.description}; "
                f"significance={episode.significance}; tags={','.join(episode.tags) or 'none'}"
            )
            for episode in episodes
        )
        candidates = "\n".join(_candidate_fact_line(item) for item in candidate_facts[:20]) or "none"
        prompt = (
            f"Subject entity id:\n{subject_entity_id}\n\n"
            f"Predicate hint from grouping step:\n{predicate_hint}\n\n"
            f"Candidate existing semantic facts from mem0 recall:\n{candidates}\n\n"
            f"Repeated episode evidence:\n{evidence}\n\n"
            "Return the semantic abstraction."
        )
        result = run_async(self._abstraction_function_for().run(prompt))
        return result.output

    def decide_hint_merge(
        self,
        *,
        abstraction: SemanticFactAbstraction,
        candidate_hints: list[dict[str, Any]],
    ) -> SemanticFactHintMergeDecision:
        if not self._config.api_key:
            raise RuntimeError("Set JOURNAL_GEMINI_API_KEY or GEMINI_API_KEY to run consolidation.")
        candidates = "\n".join(_candidate_hint_line(item) for item in candidate_hints[:20]) or "none"
        prompt = (
            "New semantic fact hint candidate:\n"
            f"- predicate={abstraction.predicate}; value={abstraction.value}; "
            f"confidence={abstraction.confidence_score:.2f}; rationale={abstraction.rationale}\n\n"
            f"Existing pending hints:\n{candidates}\n\n"
            "Decide whether the new hint semantically and contextually matches an existing pending hint."
        )
        result = run_async(self._merge_function_for().run(prompt))
        return result.output

    def _group_function_for(self) -> StructuredLLMFunction:
        if self._group_function is None:
            self._group_function = StructuredLLMFunction(
                model=self._config.model,
                output_type=ConsolidationGroupingResult,
                system_prompt=_CONSOLIDATION_GROUPING_PROMPT,
            )
        return self._group_function

    def _abstraction_function_for(self) -> StructuredLLMFunction:
        if self._abstraction_function is None:
            self._abstraction_function = StructuredLLMFunction(
                model=self._config.model,
                output_type=SemanticFactAbstraction,
                system_prompt=_CONSOLIDATION_ABSTRACTION_PROMPT,
            )
        return self._abstraction_function

    def _merge_function_for(self) -> StructuredLLMFunction:
        if self._merge_function is None:
            self._merge_function = StructuredLLMFunction(
                model=self._config.model,
                output_type=SemanticFactHintMergeDecision,
                system_prompt=_HINT_MERGE_PROMPT,
            )
        return self._merge_function


def _candidate_fact_line(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    fact_id = metadata.get("source_id") or item.get("id") or "unknown"
    memory_type = metadata.get("memory_type") or item.get("memory_type") or "memory"
    text = item.get("memory") or item.get("text") or item.get("content") or str(item)
    return f"- fact_id={fact_id}; type={memory_type}; {str(text).strip()}"


def _candidate_hint_line(item: dict[str, Any]) -> str:
    return (
        f"- hint_id={item.get('hint_id')}; subject={item.get('subject_entity_id')}; "
        f"predicate={item.get('predicate')}; support={item.get('support_count')}; "
        f"value={item.get('value')}; rationale={item.get('rationale')}"
    )


_ORGANIZER_SYSTEM_PROMPT = """Organize a captured journal thought.

Rules:
- thought_type must be one of idea, work, task, health, decision, commitment, risk, extras.
- thought should be a short topic label, or null if no label helps.
- tags should be 1 to 6 short lowercase labels.
- significance should be one plain sentence.
- salience_score should be a number from 0 to 1.
"""


_CONSOLIDATION_GROUPING_PROMPT = """You are a memory consolidation grouping process for a private CLI journal.

Given unconsolidated journal episodes, group only episodes that appear to support
the same durable, reusable semantic fact.

Rules:
- Return structured output only.
- A group should contain episode_ids that describe the same repeated signal.
- Do not group unrelated events just because they share a tag or entity.
- Use entity_refs from the supplied episodes; do not invent entity ids.
- predicate_hint must be exactly one of: preference, constraint, work_style,
  tool_usage, relationship, project_fact, recurring_need, goal.
- Leave isolated or weak one-off episodes out of all groups.
- Do not include private chain-of-thought. The rationale should be short.
"""


_CONSOLIDATION_ABSTRACTION_PROMPT = """You are a memory consolidation process for a private CLI journal.

Given repeated episode evidence, abstract it into one durable semantic fact,
update an existing candidate, or skip promotion.

Rules:
- Return structured output only.
- Preserve the subject entity. Do not invent entity ids.
- Set action=create_new when no candidate already captures the same meaning.
- Set action=update_existing when a candidate captures the same meaning and should be strengthened or rewritten.
- Set target_fact_id to the exact candidate fact_id when action=update_existing.
- Set action=do_not_promote when the evidence is contradictory, too vague, or not reusable.
- Prefer facts that remain useful without a timestamp.
- predicate must be exactly one of:
  preference, constraint, work_style, tool_usage, relationship, project_fact,
  recurring_need, goal.
- Use preference for likes, dislikes, defaults, and subjective choices.
- Use constraint for hard limits, rules, boundaries, policies, must/must-not facts.
- Use work_style for planning, deciding, executing, reviewing, or collaboration patterns.
- Use tool_usage for preferred tools and repeated ways of using them.
- Use relationship for how people, teams, projects, clients, tools, or systems relate.
- Use project_fact for stable facts about projects, organizations, artifacts, or initiatives.
- Use recurring_need for repeated situations where assistance is likely useful.
- Use goal for durable objectives or desired outcomes.
- The value should be a compact natural-language fact.
- Do not include private chain-of-thought. The rationale should be short.
"""


_HINT_MERGE_PROMPT = """You are a semantic memory hint merge process for a private CLI journal.

Decide whether a new semantic fact hint candidate matches an existing pending hint.
Use semantic meaning and context, not keyword overlap.

Rules:
- Return structured output only.
- action=merge_existing when an existing hint expresses the same durable fact or a clearly compatible version of it.
- action=create_new when the candidate is meaningfully different from all existing hints.
- action=reject when the candidate is too vague, contradictory, or not useful.
- target_hint_id is required when action=merge_existing.
- merged_value should be the best compact fact text when merging; otherwise it may be null.
- Do not include private chain-of-thought. The rationale should be short.
"""
