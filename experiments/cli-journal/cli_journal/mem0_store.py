from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from .config import load_mem0_config
from .models import Episode, SemanticFact
from .runtime import configure_quiet_runtime


configure_quiet_runtime()


class JournalMem0Store:
    """Durable mem0 store for journal episodic and semantic memory."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        agent_id: str | None = None,
        schema_version: int = 1,
    ) -> None:
        if client is None:
            configure_quiet_runtime()
            from mem0 import AsyncMemory

            client = AsyncMemory(config=_local_gemini_memory_config())
        config = load_mem0_config(agent_id=agent_id)
        self._client = client
        self.agent_id = config.agent_id
        self.schema_version = schema_version

    @classmethod
    def from_env(cls) -> "JournalMem0Store":
        config = load_mem0_config()
        if config.api_key:
            configure_quiet_runtime()
            from mem0 import MemoryClient

            return cls(MemoryClient(api_key=config.api_key), agent_id=config.agent_id)
        return cls(agent_id=config.agent_id)

    async def add_episode(self, episode: Episode, *, user_id: str) -> Episode:
        await self._add_memory(
            _episode_text(episode),
            user_id=user_id,
            memory_type="episode",
            source_id=episode.episode_id,
            metadata=episode.__dict__,
        )
        return episode

    async def add_semantic_fact(self, fact: SemanticFact, *, user_id: str) -> SemanticFact:
        await self._add_memory(
            _semantic_fact_text(fact),
            user_id=user_id,
            memory_type="semantic_fact",
            source_id=fact.fact_id,
            metadata=fact.__dict__,
        )
        return fact

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {
            "user_id": user_id,
            "agent_id": self.agent_id,
        }
        if memory_type is not None:
            filters["metadata"] = {"memory_type": memory_type}
        response = await self._call("search", query=query, filters=filters, top_k=limit)
        values = response.get("results", response) if isinstance(response, dict) else response
        return values[:limit] if isinstance(values, list) else []

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._call("close")

    async def _add_memory(
        self,
        content: str,
        *,
        user_id: str,
        memory_type: str,
        source_id: str,
        metadata: dict[str, Any],
    ) -> None:
        await self._call(
            "add",
            messages=[{"role": "user", "content": content[:4000]}],
            user_id=user_id,
            agent_id=self.agent_id,
            metadata=_clean_metadata(
                {
                    **metadata,
                    "memory_type": memory_type,
                    "source_id": source_id,
                    "agent_id": self.agent_id,
                    "schema_version": self.schema_version,
                }
            ),
        )

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._client, method_name)
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await asyncio.to_thread(method, *args, **kwargs)


def _episode_text(episode: Episode) -> str:
    return "\n".join(
        part
        for part in [
            f"Episode: {episode.event_type}",
            episode.description,
            f"Thought label: {episode.thought}" if episode.thought else "",
            f"Tags: {', '.join(episode.tags)}" if episode.tags else "",
            f"Significance: {episode.significance}" if episode.significance else "",
            f"Occurred at: {episode.occurred_at}",
        ]
        if part
    )


def _semantic_fact_text(fact: SemanticFact) -> str:
    return f"Semantic fact: {fact.subject_entity_id}.{fact.predicate} = {fact.value}"


def _local_gemini_memory_config() -> Any:
    config = load_mem0_config()
    if not config.gemini_api_key:
        raise RuntimeError(
            "Set JOURNAL_MEM0_GEMINI_API_KEY, JOURNAL_GEMINI_API_KEY, or GEMINI_API_KEY "
            "to use local mem0 with Gemini."
        )

    from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig

    return MemoryConfig(
        llm=LlmConfig(
            provider="gemini",
            config={
                "api_key": config.gemini_api_key,
                "model": config.gemini_llm_model,
            },
        ),
        embedder=EmbedderConfig(
            provider="gemini",
            config={
                "api_key": config.gemini_api_key,
                "model": config.gemini_embedding_model,
                "embedding_dims": config.gemini_embedding_dims,
            },
        ),
        vector_store=VectorStoreConfig(
            provider="chroma",
            config={
                "collection_name": "cli_journal_mem0",
                "path": str(Path(config.vector_path).expanduser()),
            },
        ),
    )


def _clean_metadata(value: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Return metadata values accepted by Chroma-backed mem0 stores."""

    cleaned: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        if item is None:
            continue
        cleaned[str(key)] = _metadata_scalar(item)
    return cleaned


def _metadata_scalar(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(_metadata_scalar(item)) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return str(value)
