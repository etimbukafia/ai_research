from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent

from .async_utils import run_async
from .config import ensure_gemini_provider_env, load_gemini_config
from .db import DEFAULT_PROFILE_ID, JournalDatabase
from .mem0_store import JournalMem0Store
from .models import Entity, JournalSession, Thought
from .priming import PrimingHit, PrimingStore
from .runtime import captured_output, configure_quiet_runtime, quiet_third_party_output


configure_quiet_runtime()


ThoughtType = Literal["idea", "work", "task", "health", "decision", "commitment", "risk", "extras"]


class JournalAgentResponse(BaseModel):
    """Typed output returned by the interactive journal agent."""

    answer: str = Field(description="The conversational answer shown to the user.")
    should_save_thought: bool = Field(
        default=False,
        description="True when the user said something that should probably be captured as a thought.",
    )
    suggested_tags: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Short tags that would help organize the suggested thought.",
    )


@dataclass
class JournalAgentResult:
    answer: str
    response: JournalAgentResponse | None = None
    memory_error: str | None = None
    priming_error: str | None = None


class JournalAgent:
    """Conversational journal assistant backed by SQLite session state and mem0."""

    def __init__(
        self,
        db: JournalDatabase,
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
        runner: PydanticAgent | None = None,
        memory: JournalMem0Store | None = None,
        priming: PrimingStore | None = None,
    ) -> None:
        self.db = db
        self.profile_id = profile_id
        config = load_gemini_config()
        ensure_gemini_provider_env(config)
        self._config = config
        self._runner = runner
        self._memory = memory
        self._priming = priming

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

    def run(self, message: str, *, session: JournalSession, limit: int = 6) -> JournalAgentResult:
        """Answer a chat turn using session state, local journal data, and mem0 recall."""

        query = " ".join(message.split())
        if not query:
            raise ValueError("Message cannot be empty.")

        thought_hits = self.db.search_thoughts(self.profile_id, query, limit=limit)
        entity_hits = self._search_entities(query, limit=limit)
        priming_hits, priming_error = self._search_priming(query, limit=limit)
        memory_hits, memory_error = self._recall_mem0(_mem0_query(query, priming_hits), limit=limit)
        prompt = _agent_prompt(
            message=query,
            session=session,
            thought_hits=thought_hits,
            entity_hits=entity_hits,
            priming_hits=priming_hits,
            memory_hits=memory_hits,
        )
        if not self._config.api_key:
            raise RuntimeError("Set JOURNAL_GEMINI_API_KEY or GEMINI_API_KEY to run journal chat.")
        result = run_async(self._runner_for().run(prompt))
        response = result.output
        return JournalAgentResult(
            answer=response.answer,
            response=response,
            memory_error=memory_error,
            priming_error=priming_error,
        )

    def _runner_for(self) -> PydanticAgent:
        if self._runner is None:
            self._runner = PydanticAgent(
                model=self._config.model,
                output_type=JournalAgentResponse,
                system_prompt=_SYSTEM_PROMPT,
            )
        return self._runner

    def _recall_mem0(self, query: str, *, limit: int) -> tuple[list[dict[str, Any]], str | None]:
        try:
            with quiet_third_party_output() as output_streams:
                result = run_async(self.memory.recall(query, user_id=self.profile_id, limit=limit))
            self._log_third_party_output("agent.mem0_recall.output", captured_output(output_streams))
            return result, None
        except Exception as exc:
            return [], str(exc)

    def _search_priming(self, query: str, *, limit: int) -> tuple[list[PrimingHit], str | None]:
        try:
            with quiet_third_party_output() as output_streams:
                result = self.priming.search(query, profile_id=self.profile_id, limit=limit)
            self._log_third_party_output("agent.priming_search.output", captured_output(output_streams))
            return result, None
        except Exception as exc:
            return [], str(exc)

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

    def _search_entities(self, query: str, *, limit: int) -> list[Entity]:
        tokens = {token for token in query.lower().split() if len(token) > 2}
        results: list[Entity] = []
        for entity in self.db.list_entities(self.profile_id, limit=1000):
            fields = " ".join([entity.canonical_name, entity.type, entity.description, *entity.aliases]).lower()
            if any(token in fields for token in tokens):
                results.append(entity)
        return results[:limit]


def _agent_prompt(
    *,
    message: str,
    session: JournalSession,
    thought_hits: list[tuple[Thought, float]],
    entity_hits: list[Entity],
    priming_hits: list[PrimingHit],
    memory_hits: list[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "You are a private CLI journal assistant.",
            "Answer conversationally and use the provided memory. Be concise, direct, and practical.",
            "If the user is reflecting, help them make sense of the pattern. If they ask for recall, answer from the evidence. If evidence is weak, say so.",
            "Do not invent journal history. If a useful thought should be saved, suggest using /add.",
            _session_context(session),
            _priming_context(priming_hits),
            _memory_context(memory_hits),
            _thought_context(thought_hits),
            _entity_context(entity_hits),
            f"User message:\n{message}",
            "Assistant answer:",
        ]
    )


_SYSTEM_PROMPT = """You are a private CLI journal assistant.

Use the provided session memory, priming hits, local journal records, entities, and mem0 recall.
Answer conversationally, but keep the answer practical and concise.
Do not invent journal history.
If the user is reflecting, help them see a pattern.
If the user asks for recall, answer from the evidence.
If evidence is weak, say what is missing.
Set should_save_thought to true when the user gives a new thought, commitment, decision, risk, or task that should be captured.
"""


def _session_context(session: JournalSession) -> str:
    last_user = session.last_exchange.get("user") if session.last_exchange else "none"
    last_assistant = session.last_exchange.get("assistant") if session.last_exchange else "none"
    return "\n".join(
        [
            "Session memory:",
            f"- session_id: {session.session_id}",
            f"- rolling_summary: {session.rolling_summary or 'none'}",
            f"- active_thought_ids: {', '.join(session.active_thought_ids) or 'none'}",
            f"- active_entity_ids: {', '.join(session.active_entity_ids) or 'none'}",
            f"- recent_queries: {', '.join(session.recent_queries[-5:]) or 'none'}",
            f"- last_user: {last_user}",
            f"- last_assistant: {last_assistant}",
        ]
    )


def _memory_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return "mem0 recall:\n- none"
    lines = ["mem0 recall:"]
    for item in items:
        text = item.get("memory") or item.get("text") or item.get("content") or ""
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        memory_type = metadata.get("memory_type") or item.get("memory_type") or "memory"
        lines.append(f"- [{memory_type}] {str(text).strip()}")
    return "\n".join(lines)


def _priming_context(items: list[PrimingHit]) -> str:
    if not items:
        return "Priming memory:\n- none"
    lines = ["Priming memory:"]
    for item in items:
        distance = f" distance={item.distance:.3f}" if item.distance is not None else ""
        lines.append(f"- [{item.memory_type}] {item.document}{distance}")
    return "\n".join(lines)


def _thought_context(items: list[tuple[Thought, float]]) -> str:
    if not items:
        return "Local thoughts:\n- none"
    lines = ["Local thoughts:"]
    for thought, score in items:
        label = f" [{thought.thought}]" if thought.thought else ""
        tags = f" tags={', '.join(thought.tags)}" if thought.tags else ""
        lines.append(f"- {thought.created_at[:10]} {thought.thought_type}{label}: {thought.body} score={score:.1f}{tags}")
    return "\n".join(lines)


def _entity_context(items: list[Entity]) -> str:
    if not items:
        return "Entities:\n- none"
    lines = ["Entities:"]
    for entity in items:
        description = f" - {entity.description}" if entity.description else ""
        lines.append(f"- {entity.canonical_name} ({entity.type}){description}")
    return "\n".join(lines)


def _mem0_query(query: str, priming_hits: list[PrimingHit]) -> str:
    if not priming_hits:
        return query
    familiar_context = " ".join(hit.document for hit in priming_hits[:4])
    return f"{query}\n\nFamiliar context from priming:\n{familiar_context}"
