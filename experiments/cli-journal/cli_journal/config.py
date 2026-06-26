from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MEM0_GEMINI_LLM_MODEL = "gemini-2.0-flash"
DEFAULT_MEM0_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_MEM0_GEMINI_EMBEDDING_DIMS = 768
DEFAULT_MEM0_VECTOR_PATH = "~/.cli-journal/mem0-chroma"
DEFAULT_MEM0_AGENT_ID = "cli-journal"
DEFAULT_PRIMING_PATH = "~/.cli-journal/chroma"


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini settings used by the background thought organizer."""

    api_key: str | None
    model: str


@dataclass(frozen=True)
class Mem0Config:
    """mem0 settings used for durable episodic and semantic memory."""

    api_key: str | None
    agent_id: str
    gemini_api_key: str | None
    gemini_llm_model: str
    gemini_embedding_model: str
    gemini_embedding_dims: int
    vector_path: str


@dataclass(frozen=True)
class PrimingConfig:
    """ChromaDB settings used for the local familiarity index."""

    path: str
    collection: str


def load_gemini_config(
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> GeminiConfig:
    """Load Gemini config from explicit args first, then environment variables."""

    return GeminiConfig(
        api_key=api_key or _env_first("JOURNAL_GEMINI_API_KEY", "GEMINI_API_KEY"),
        model=model or _env_first("JOURNAL_GEMINI_MODEL", "GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
    )


def load_mem0_config(*, api_key: str | None = None, agent_id: str | None = None) -> Mem0Config:
    """Load mem0 config from explicit args first, then environment variables."""

    return Mem0Config(
        api_key=api_key or _env_first("MEM0_API_KEY", "JOURNAL_MEM0_API_KEY"),
        agent_id=agent_id or _env_first("JOURNAL_MEM0_AGENT_ID") or DEFAULT_MEM0_AGENT_ID,
        gemini_api_key=_env_first("JOURNAL_MEM0_GEMINI_API_KEY", "JOURNAL_GEMINI_API_KEY", "GEMINI_API_KEY"),
        gemini_llm_model=_env_first(
            "JOURNAL_MEM0_GEMINI_LLM_MODEL",
            "JOURNAL_GEMINI_MODEL",
            "GEMINI_MODEL",
        )
        or DEFAULT_MEM0_GEMINI_LLM_MODEL,
        gemini_embedding_model=(
            _env_first("JOURNAL_MEM0_GEMINI_EMBEDDING_MODEL") or DEFAULT_MEM0_GEMINI_EMBEDDING_MODEL
        ),
        gemini_embedding_dims=_env_int("JOURNAL_MEM0_GEMINI_EMBEDDING_DIMS")
        or DEFAULT_MEM0_GEMINI_EMBEDDING_DIMS,
        vector_path=_env_first("JOURNAL_MEM0_VECTOR_PATH") or DEFAULT_MEM0_VECTOR_PATH,
    )


def load_priming_config(*, path: str | None = None, collection: str | None = None) -> PrimingConfig:
    """Load ChromaDB config for the priming memory layer."""

    return PrimingConfig(
        path=path or _env_first("JOURNAL_PRIMING_PATH") or DEFAULT_PRIMING_PATH,
        collection=collection or _env_first("JOURNAL_PRIMING_COLLECTION") or "journal_priming",
    )


def ensure_gemini_provider_env(config: GeminiConfig) -> None:
    """Expose journal-specific Gemini config through provider env variables."""

    if config.api_key and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = config.api_key


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    return int(value)
