"""Construction of the local lab dependency graph."""

from __future__ import annotations

from dataclasses import dataclass

from .config import LabConfig
from .policies import PolicyEngine
from .retrieval import HybridRetriever, build_retrievers
from .storage import SourceStore
from .tools import ToolRegistry


@dataclass
class LabComponents:
    store: SourceStore
    retriever: HybridRetriever
    policy_engine: PolicyEngine
    registry: ToolRegistry


def build_components(config: LabConfig, condition: str, account_scope: str | None, case_time: str) -> LabComponents:
    store = SourceStore(config.source_db)
    retriever = build_retrievers(
        use_minilm=config.use_minilm,
        model_name=config.embedding_model,
    )[condition]
    policy_engine = PolicyEngine(
        store,
        account_scope=account_scope,
        case_time=case_time,
    )
    registry = ToolRegistry(
        store,
        retriever,
        policy_engine,
        review_dir=config.review_dir,
        account_scope=account_scope,
        case_time=case_time,
    )
    return LabComponents(store, retriever, policy_engine, registry)

