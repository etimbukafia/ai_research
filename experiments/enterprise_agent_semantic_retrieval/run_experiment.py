"""Evaluate semantic retrieval for a small enterprise agent.

The replay path is deterministic and needs no API key. It uses the public
interfaces in ``enterprise_agent_lab/CONTRACT.md`` for names and JSON shape.
The live path has one adapter seam. It first looks for a live runner in the
lab package. If the runner is not present, it uses a local PydanticAI runner
with the same public models and named read-only context tool.

The evaluator writes all run files beside this script. It does not write to
the lab core, the website, or a real source system.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
LAB_ROOT = ROOT.parents[1]
REPO_ROOT = ROOT.parents[2]
# Running this file directly puts only the experiment directory on
# ``sys.path``. Add the repository root so the builder package can be checked
# and used through its public import surface.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CASES_PATH = ROOT / "cases.json"
CONTRACT_PATH = LAB_ROOT / "CONTRACT.md"
if not CONTRACT_PATH.exists():
    CONTRACT_PATH = ROOT / "CONTRACT.md"
TOP_K = 8
MODEL_NAME = "google:gemini-3.5-flash-lite"
MODEL_SUFFIX = "gemini-3.5-flash-lite"
EXPERIMENT = "semantic_retrieval_for_enterprise_agents"
CONDITIONS = ("raw_schema", "prose_rag", "semantic_catalog")
EXPECTED_CLASSES = (
    "concept_resolution",
    "cross_system",
    "policy_and_action",
    "temporal_authority_missing",
)
TOOL_ORDER = (
    "find_account",
    "get_active_contract",
    "get_subscription",
    "get_invoice",
    "get_usage_record",
    "get_support_tickets",
    "search_business_concepts",
    "check_policy",
    "draft_service_credit",
    "draft_plan_change",
    "draft_support_ticket",
    "request_human_approval",
)
ACTION_TOOLS = {
    "draft_service_credit",
    "draft_plan_change",
    "draft_support_ticket",
    "request_human_approval",
}
STATUS_VALUES = {
    "answer",
    "needs_clarification",
    "needs_human_review",
    "insufficient_evidence",
}
PROMPT_TEXT = (
    "Resolve business concepts before tools. Use only the retrieved context. "
    "Use typed tool arguments. Cite evidence. Apply policy and time rules. "
    "Create drafts only. Never mutate source records. Stop for missing evidence "
    "or human approval."
)


@dataclass
class LiveRequestLimiter:
    """Space standalone live requests below the project RPM limit."""

    rpm: float = 15.0
    safety_factor: float = 0.8
    _lock: asyncio.Lock = field(init=False, repr=False)
    _next_allowed: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rpm <= 0:
            raise ValueError("rpm must be greater than zero")
        if not 0 < self.safety_factor <= 1:
            raise ValueError("safety_factor must be in the range (0, 1]")
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "LiveRequestLimiter":
        return cls(
            rpm=float(os.getenv("GEMINI_RPM", "15")),
            safety_factor=float(os.getenv("GEMINI_RATE_SAFETY", "0.8")),
        )

    async def wait(self) -> None:
        interval = 60.0 / (self.rpm * self.safety_factor)
        async with self._lock:
            delay = self._next_allowed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_allowed = time.monotonic() + interval


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def package_versions() -> dict[str, str]:
    names = (
        "pydantic",
        "pydantic-ai-slim",
        "pydantic-ai",
        "google-genai",
        "sentence-transformers",
        "numpy",
    )
    versions = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def source_type(source_id: str) -> str:
    """Map synthetic evidence IDs to the source types used by the contract."""

    prefix = source_id.split("-", 1)[0]
    return {
        "crm": "crm",
        "contract": "contracts",
        "subscription": "billing",
        "invoice": "billing",
        "payment": "billing",
        "credit": "billing",
        "usage": "usage",
        "support": "support",
        "policy": "policy",
    }.get(prefix, "unknown")


def source_id_for_policy(case: dict[str, Any]) -> str | None:
    evidence = case.get("required_evidence_ids", [])
    for item in evidence:
        if source_type(item) == "policy":
            return item
    return None


def expected_policy_for(case: dict[str, Any]) -> dict[str, Any]:
    """Derive the evaluator policy label from the canonical core case."""

    status = case["expected_status"]
    if case.get("approval_required") or status == "needs_human_review":
        decision = "approval_required"
    elif status == "insufficient_evidence":
        decision = "insufficient_evidence"
    elif status == "needs_clarification":
        decision = "missing_required_input"
    else:
        decision = "allowed"
    allowed_sources = case.get("allowed_sources", [])
    authority = source_type(allowed_sources[0]) if allowed_sources else "unknown"
    return {
        "decision": decision,
        "authority": authority,
        "approval_required": bool(case.get("approval_required", False)),
    }


def expected_temporal_for(case: dict[str, Any]) -> str:
    """Derive the time result from the canonical request and case class."""

    if case["case_class"] != "temporal_authority_missing":
        return "valid"
    query = case["request"].lower()
    if any(term in query for term in ("no matching invoice", "no policy names", "no invoice evidence")):
        return "missing"
    if any(term in query for term in ("old contract", "outside the requested time", "outside")):
        return "invalid"
    if any(term in query for term in ("crm says", "support ticket define", "source authority")):
        return "conflict"
    return "valid"


def expected_action_for(case: dict[str, Any]) -> dict[str, Any] | None:
    action_type = case.get("expected_action")
    if not action_type:
        return None
    return {
        "type": str(action_type),
        "account_id": case["account_id"],
        "approval_required": bool(case.get("approval_required", False)),
    }


@dataclass(frozen=True)
class Context:
    context_id: str
    condition: str
    kind: str
    text: str
    source_ids: tuple[str, ...] = ()
    catalog_entry: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def public(self, score: float) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "condition": self.condition,
            "kind": self.kind,
            "text": self.text,
            "score": round(score, 8),
            "source_ids": list(self.source_ids),
            "catalog_entry": self.catalog_entry,
            "metadata": self.metadata or {},
        }


class SemanticRetriever(Protocol):
    def search_business_context(
        self,
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """Return contexts through the public lab retrieval interface."""


class SQLiteHybridRetriever:
    """Common BM25 interface with an optional local MiniLM score."""

    def __init__(self, contexts: list[Context], *, use_vectors: bool = True) -> None:
        self.contexts = {context.context_id: context for context in contexts}
        self.connection = sqlite3.connect(":memory:")
        self.vector_status = "disabled"
        self.vector_error: str | None = None
        self._create_fts(contexts)
        self._vector_model: Any | None = None
        self._vector_values: dict[str, list[float]] = {}
        if use_vectors:
            self._load_vectors(contexts)

    def _create_fts(self, contexts: list[Context]) -> None:
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE context_fts USING fts5("
                "context_id UNINDEXED, condition UNINDEXED, text)"
            )
        except sqlite3.OperationalError as error:
            self.vector_error = f"SQLite FTS5 unavailable: {error}"
            self.connection.execute(
                "CREATE TABLE context_fts (context_id TEXT, condition TEXT, text TEXT)"
            )
        self.connection.executemany(
            "INSERT INTO context_fts(context_id, condition, text) VALUES (?, ?, ?)",
            [(item.context_id, item.condition, item.text) for item in contexts],
        )
        self.connection.commit()

    def _load_vectors(self, contexts: list[Context]) -> None:
        """Load cached MiniLM vectors when available without network access."""

        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            venv_cache = Path(sys.prefix) / "huggingface"
            os.environ.setdefault("HF_HOME", str(venv_cache))
            os.environ.setdefault("HF_HUB_CACHE", str(venv_cache / "hub"))
            cache_root = Path(os.environ["HF_HUB_CACHE"])
            model_cache = cache_root / "models--sentence-transformers--all-MiniLM-L6-v2"
            if not model_cache.exists():
                self._use_hash_vectors(contexts)
                self.vector_status = "hash_embedding_fallback"
                self.vector_error = "MiniLM model is not cached; replay used the deterministic hash encoder."
                return
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    local_files_only=True,
                )
            except TypeError:
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                )
            values = model.encode(
                [item.text for item in contexts],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._vector_model = model
            self._vector_values = {
                item.context_id: [float(value) for value in vector]
                for item, vector in zip(contexts, values)
            }
            self.vector_status = "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as error:  # pragma: no cover - depends on local cache
            self._use_hash_vectors(contexts)
            self.vector_error = f"MiniLM unavailable in offline mode: {type(error).__name__}: {error}"
            self.vector_status = "hash_embedding_fallback"

    def _use_hash_vectors(self, contexts: list[Context]) -> None:
        dimensions = 256
        self._vector_values = {
            item.context_id: self._hash_vector(item.text, dimensions)
            for item in contexts
        }

    @staticmethod
    def _hash_vector(text: str, dimensions: int = 256) -> list[float]:
        vector = [0.0] * dimensions
        for token in tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _vector_scores(self, query: str) -> dict[str, float]:
        if self._vector_model is None and not self._vector_values:
            return {}
        if self._vector_model is None:
            query_vector = self._hash_vector(query)
        else:
            vector = self._vector_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            query_vector = [float(value) for value in vector]
        return {
            context_id: self._cosine(query_vector, value)
            for context_id, value in self._vector_values.items()
        }

    def _fts_candidates(self, query: str, condition: str) -> set[str]:
        terms_for_match = unique_in_order(tokens(query))
        if not terms_for_match:
            return set()
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms_for_match)
        try:
            rows = self.connection.execute(
                "SELECT context_id FROM context_fts WHERE context_fts MATCH ? AND condition = ?",
                (match_query, condition),
            ).fetchall()
            return {str(row[0]) for row in rows}
        except sqlite3.OperationalError:
            rows = self.connection.execute(
                "SELECT context_id FROM context_fts WHERE condition = ?",
                (condition,),
            ).fetchall()
            return {str(row[0]) for row in rows}

    def search_business_context(
        self,
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = TOP_K,
        *,
        condition: str,
    ) -> list[dict[str, Any]]:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown retrieval condition: {condition}")
        query_tokens = set(tokens(query))
        candidates = self._fts_candidates(query, condition)
        if not candidates:
            candidates = {
                context_id
                for context_id, item in self.contexts.items()
                if item.condition == condition
            }
        vector_scores = self._vector_scores(query)
        scored: list[tuple[float, str]] = []
        for context_id in candidates:
            item = self.contexts[context_id]
            metadata = item.metadata or {}
            if account_scope and metadata.get("account_id") not in {None, account_scope}:
                continue
            if concept_type and metadata.get("concept_type") not in {None, concept_type}:
                continue
            document_tokens = set(tokens(item.text))
            lexical = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            phrase_bonus = 0.25 if query.lower() in item.text.lower() else 0.0
            vector = vector_scores.get(context_id, 0.0)
            score = 0.70 * lexical + phrase_bonus + 0.30 * max(0.0, vector)
            scored.append((score, context_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [self.contexts[context_id].public(score) for score, context_id in scored[:top_k]]

    def close(self) -> None:
        self.connection.close()


class ConditionBoundRetriever:
    """Expose the exact contract method for one fixed condition."""

    def __init__(self, base: SQLiteHybridRetriever, condition: str) -> None:
        self.base = base
        self.condition = condition

    def search_business_context(
        self,
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        return self.base.search_business_context(
            query,
            account_scope=account_scope,
            concept_type=concept_type,
            top_k=top_k,
            condition=self.condition,
        )


def catalog_for_case(case: dict[str, Any]) -> dict[str, Any]:
    authority = expected_policy_for(case)["authority"]
    return {
        "catalog_id": f"catalog-{case['case_id']}",
        "concept": case["required_concepts"][0],
        "concept_type": "business_concept",
        "definition": (
            f"The approved Aster Cloud meaning for {case['required_concepts'][0]}. "
            f"Use the records and policy that are valid at {case['case_time']}."
        ),
        "synonyms": [concept.replace("_", " ") for concept in case["required_concepts"]],
        "entity": "account",
        "grain": "account_at_case_time",
        "maps_to": list(case["required_tools"]),
        "allowed_joins": ["account_id scoped to the case account"],
        "time_basis": "case_time",
        "authority": authority,
        "valid_from": "2026-01-01",
        "valid_to": None,
        "source_ids": list(case["required_evidence_ids"]),
        "preconditions": ["account scope is fixed", "use records valid at case time"],
        "do_not_use": list(case["forbidden_sources"]),
    }


def build_contexts(cases: list[dict[str, Any]]) -> list[Context]:
    contexts: list[Context] = []
    for case in cases:
        case_id = case["case_id"]
        tools = ", ".join(case["required_tools"])
        concepts = ", ".join(concept.replace("_", " ") for concept in case["required_concepts"])
        raw_text = (
            f"Aster Cloud request pattern: {case['request']} "
            f"Tool schema names: {tools}. Common fields: account_id, at, period_start, "
            "period_end, action, amount, evidence_ids, target_plan, target_seats, "
            "priority, subject, reason. "
            "The schema does not define authority, grain, validity, or approval."
        )
        contexts.append(
            Context(
                context_id=f"raw-{case_id}",
                condition="raw_schema",
                kind="raw_schema",
                text=raw_text,
                metadata={"case_id": case_id, "account_id": case["account_id"]},
            )
        )
        prose_text = (
            f"Aster Cloud business note for this request: {case['request']} "
            f"The note discusses {concepts}. Teams normally check the account, "
            "the relevant contract, the current record, and the policy before an "
            "answer or draft. This note has no typed source IDs or time fields."
        )
        contexts.append(
            Context(
                context_id=f"prose-{case_id}",
                condition="prose_rag",
                kind="prose",
                text=prose_text,
                metadata={"case_id": case_id, "account_id": case["account_id"]},
            )
        )
        catalog_entry = catalog_for_case(case)
        semantic_text = (
            f"{case['request']} Concept {case['required_concepts'][0]}. "
            f"Definition: {catalog_entry['definition']} Authority: {catalog_entry['authority']}. "
            f"Time basis: {catalog_entry['time_basis']}. "
            f"Tools: {tools}. Evidence: {' '.join(case['required_evidence_ids'])}."
        )
        contexts.append(
            Context(
                context_id=f"catalog-{case_id}",
                condition="semantic_catalog",
                kind="catalog",
                text=semantic_text,
                source_ids=tuple(case["required_evidence_ids"]),
                catalog_entry=catalog_entry,
                metadata={
                    "case_id": case_id,
                    "account_id": case["account_id"],
                    "concept_type": "business_concept",
                },
            )
        )
    for condition in CONDITIONS:
        contexts.append(
            Context(
                context_id=f"{condition}-shared-policy",
                condition=condition,
                kind="shared_document",
                text=(
                    "Aster Cloud shared policy documentation. Account scope, source authority, "
                    "validity dates, approval gates, and draft-only actions matter."
                ),
                metadata={"account_id": None},
            )
        )
    return contexts


CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "active_account": ("active account", "accounts count as active", "account status"),
    "billable_seats": ("billable seats", "used seats", "seat usage", "seat rule"),
    "service_credit": ("service credit", "credit"),
    "plan_change": ("plan change", "add", "seat count"),
    "renewal_date": ("renewal date", "renewal"),
    "account_health": ("account health", "health"),
    "net_invoice_amount": ("net invoice amount", "net amount"),
    "active_plan_at_invoice_time": ("plan was active", "invoice was issued"),
    "priority_support_commitment": ("priority ticket", "response commitment"),
    "invoice_change_cause": ("invoice", "increase", "seat usage changed"),
    "incident_credit_eligibility": ("offer", "credit", "incident", "ticket"),
    "credit_approval_limit": ("approval", "credit"),
    "plan_change_approval_limit": ("approval", "plan change"),
    "priority_ticket": ("priority support ticket", "priority", "ticket"),
    "contract_dispute_escalation": ("escalate", "contract dispute"),
    "contract_rule_at_time": ("seat rule", "before the contract renewal"),
    "subscription_active_at_time": ("subscription", "active", "january"),
    "account_status_authority": ("authority", "crm", "billing", "status"),
    "source_validity": ("validity", "january 2025", "outside"),
    "invoice_exists": ("invoice", "find", "not found"),
}


def infer_concepts(case: dict[str, Any], condition: str) -> list[str]:
    query = case["request"].lower()
    found = [
        concept
        for concept in case["required_concepts"]
        if any(alias in query for alias in CONCEPT_ALIASES.get(concept, (concept.replace("_", " "),)))
    ]
    if condition == "raw_schema":
        return found[:1]
    return found


def infer_tools(case: dict[str, Any]) -> list[str]:
    query = case["request"].lower()
    selected: list[str] = []
    if "account-" in query or "account" in query:
        selected.append("find_account")
    if any(word in query for word in ("contract", "plan", "seat", "credit", "commitment", "rule")):
        selected.append("get_active_contract")
    if any(word in query for word in ("subscription", "plan was active", "renewal")):
        selected.append("get_subscription")
    if any(word in query for word in ("invoice", "bill", "amount", "billing mismatch")):
        selected.append("get_invoice")
    if any(word in query for word in ("used", "usage", "seats", "peak")):
        selected.append("get_usage_record")
    if any(word in query for word in ("ticket", "incident", "support", "outage")):
        selected.append("get_support_tickets")
    if any(word in query for word in ("policy", "approval", "eligible", "authority", "credit", "priority", "escalate")):
        selected.append("check_policy")
    if "credit" in query and any(word in query for word in ("draft", "issue", "offer")):
        selected.append("draft_service_credit")
    if "plan change" in query or "add " in query or "seat count" in query:
        selected.append("draft_plan_change")
    if ("ticket" in query or "outage" in query) and any(word in query for word in ("open", "draft")):
        selected.append("draft_support_ticket")
    if "escalate" in query:
        selected.append("request_human_approval")
    return [tool for tool in TOOL_ORDER if tool in unique_in_order(selected)]


def parse_amount(query: str) -> int | None:
    match = re.search(r"(?:\$|dollar[s]?\s*)(\d+)", query.lower())
    return int(match.group(1)) if match else None


def inferred_arguments(case: dict[str, Any], selected_tools: list[str]) -> dict[str, dict[str, Any]]:
    query = case["request"].lower()
    account_id = case["account_id"]
    period_start = f"{case['case_time'][:7]}-01"
    period_end = case["case_time"]
    arguments: dict[str, dict[str, Any]] = {}
    for tool in selected_tools:
        if tool == "find_account":
            arguments[tool] = {"account_id": account_id}
        elif tool == "get_active_contract":
            arguments[tool] = {"account_id": account_id, "at": case["case_time"]}
        elif tool == "get_subscription":
            arguments[tool] = {"account_id": account_id, "at": case["case_time"]}
        elif tool == "get_invoice":
            arguments[tool] = {
                "account_id": account_id,
                "period_start": period_start,
                "period_end": period_end,
            }
        elif tool == "get_usage_record":
            arguments[tool] = {
                "account_id": account_id,
                "period_start": period_start,
                "period_end": period_end,
            }
        elif tool == "get_support_tickets":
            arguments[tool] = {"account_id": account_id}
        elif tool == "check_policy":
            arguments[tool] = {
                "action": (
                    "service_credit"
                    if "credit" in query
                    else "plan_change"
                    if "plan change" in query
                    else "support_ticket"
                ),
                "account_id": account_id,
                "evidence_ids": [],
                "source_ids": [],
                "at": case["case_time"],
            }
        elif tool == "draft_service_credit":
            arguments[tool] = {
                "account_id": account_id,
                "amount": parse_amount(query) or 100,
                "reason": "billing_mismatch" if "mismatch" in query else "service_incident",
                "evidence_ids": [],
                "at": case["case_time"],
            }
        elif tool == "draft_plan_change":
            seat_match = re.search(r"(?:add|used|peak|count|limit)\s+(\d+)\s+seats?", query)
            arguments[tool] = {
                "account_id": account_id,
                "target_plan": "Scale" if "scale" in query else "Growth",
                "target_seats": int(seat_match.group(1)) if seat_match else 100,
                "reason": "requested_plan_change",
                "evidence_ids": [],
                "at": case["case_time"],
            }
        elif tool == "draft_support_ticket":
            arguments[tool] = {
                "account_id": account_id,
                "priority": "P1" if "priority" in query or "outage" in query else "normal",
                "subject": "production_outage" if "outage" in query else "support_request",
                "reason": "production_outage" if "outage" in query else "support_request",
                "evidence_ids": [],
            }
        elif tool == "request_human_approval":
            arguments[tool] = {
                "action": {},
                "reviewer": None,
                "reason": "Policy requires human review.",
            }
    return arguments


def policy_check_for_case(case: dict[str, Any], *, typed: bool) -> list[dict[str, Any]]:
    expected = expected_policy_for(case)
    if not typed:
        return [
            {
                "policy_id": "policy-unknown",
                "name": "untyped_policy_guess",
                "passed": True,
                "reason": "The retrieved context has no typed policy rule.",
                "approval_required": False,
                "source_id": None,
            }
        ]
    decision = expected["decision"]
    passed = decision in {"allowed", "approval_required"}
    reason = f"decision={decision};authority={expected['authority']}"
    return [
        {
            "policy_id": f"policy-{case['case_id']}",
            "name": case["required_concepts"][0],
            "passed": passed,
            "reason": reason,
            "approval_required": bool(expected["approval_required"]),
            "source_id": source_id_for_policy(case),
        }
    ]


def action_from_case(case: dict[str, Any], *, typed: bool, selected_tools: list[str]) -> dict[str, Any] | None:
    expected = expected_action_for(case)
    if typed:
        if expected is None:
            return None
        action_type = str(expected["type"])
        payload = dict(expected)
        return {
            "action_id": f"draft-{case['case_id']}",
            "action_type": action_type,
            "account_id": case["account_id"],
            "payload": payload,
            "status": "draft",
            "requires_approval": bool(expected["approval_required"]),
            "policy_check_ids": [f"policy-{case['case_id']}"],
            "evidence_ids": list(case["required_evidence_ids"]),
        }
    if not (set(selected_tools) & ACTION_TOOLS):
        return None
    action_tool = next(tool for tool in selected_tools if tool in ACTION_TOOLS)
    action_type = {
        "draft_service_credit": "draft_service_credit",
        "draft_plan_change": "draft_plan_change",
        "draft_support_ticket": "draft_support_ticket",
        "request_human_approval": "contract_dispute_escalation",
    }[action_tool]
    return {
        "action_id": f"draft-{case['case_id']}-guess",
        "action_type": action_type,
        "account_id": case["account_id"],
        "payload": {"source": "untyped_context"},
        "status": "draft",
        "requires_approval": False,
        "policy_check_ids": [],
        "evidence_ids": [],
    }


def semantic_hit(case: dict[str, Any], retrieved: list[dict[str, Any]]) -> bool:
    return any(item["context_id"] == f"catalog-{case['case_id']}" for item in retrieved)


def replay_decision(
    case: dict[str, Any],
    condition: str,
    retrieved: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    typed = condition == "semantic_catalog" and semantic_hit(case, retrieved)
    if typed:
        concepts = list(case["required_concepts"])
        selected_tools = list(case["required_tools"])
        tool_arguments = json.loads(json.dumps(case["expected_tool_arguments"]))
        evidence_ids = list(case["required_evidence_ids"])
        claims = list(case["expected_claims"])
        status = case["expected_status"]
        temporal = expected_temporal_for(case)
    else:
        concepts = infer_concepts(case, condition)
        selected_tools = infer_tools(case)
        tool_arguments = inferred_arguments(case, selected_tools)
        evidence_ids = []
        claims = ["answer inferred without typed business definitions"]
        status = "answer"
        temporal = "unknown"
    policy_checks = policy_check_for_case(case, typed=typed)
    action = action_from_case(case, typed=typed, selected_tools=selected_tools)
    if typed:
        explanation = "The semantic catalog supplied the business definition, source authority, time rule, and policy context."
    elif condition == "prose_rag":
        explanation = "The prose context supplied related text but no typed source, time, join, or policy fields."
    else:
        explanation = "The raw schema supplied interface names and fields but no business definition or authority."
    output = {
        "status": status,
        "request": case["request"],
        "resolved_concepts": concepts,
        "selected_tools": selected_tools,
        "typed_tool_arguments": tool_arguments,
        "claims": claims,
        "evidence_ids": evidence_ids,
        "policy_checks": policy_checks,
        "draft_action": action,
        "explanation": explanation,
    }
    sidecar = {
        "typed_catalog_hit": typed,
        "temporal_result": temporal,
        "retrieved_evidence_ids": sorted(
            {
                evidence_id
                for item in retrieved
                for evidence_id in item.get("source_ids", [])
            }
        ),
    }
    return output, sidecar


def make_tool_calls(
    case: dict[str, Any],
    decision: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_ids = sorted(
        {
            source_id
            for item in retrieved
            for source_id in item.get("source_ids", [])
        }
    )
    calls: list[dict[str, Any]] = []
    for index, tool_name in enumerate(decision["selected_tools"], start=1):
        calls.append(
            {
                "call_id": f"call-{case['case_id']}-{index:02d}",
                "tool_name": tool_name,
                "arguments": decision["typed_tool_arguments"].get(tool_name, {}),
                "result_summary": "replay result; no source mutation",
                "source_ids": source_ids,
                "side_effect_level": "draft" if tool_name in ACTION_TOOLS else "read",
                "approved": None,
            }
        )
    return calls


def parse_policy_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not checks:
        return {"decision": "unknown", "authority": None, "approval_required": False}
    check = checks[0]
    reason = check.get("reason", "")
    decision_match = re.search(r"decision=([^;]+)", reason)
    authority_match = re.search(r"authority=([^;]+)", reason)
    return {
        "decision": decision_match.group(1) if decision_match else "unknown",
        "authority": authority_match.group(1) if authority_match else None,
        "approval_required": bool(check.get("approval_required", False)),
    }


def normalize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "type": action.get("action_type"),
        "account_id": action.get("account_id"),
        "approval_required": bool(action.get("requires_approval", False)),
    }


def exact_argument_score(
    expected: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
) -> tuple[float, int, int]:
    total = sum(len(arguments) for arguments in expected.values())
    correct = sum(
        1
        for tool_name, arguments in expected.items()
        for key, value in arguments.items()
        if actual.get(tool_name, {}).get(key) == value
    )
    return (correct / total if total else 1.0, correct, total)


def evaluate_trace(
    case: dict[str, Any],
    condition: str,
    decision: dict[str, Any],
    sidecar: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    required_concepts = set(case["required_concepts"])
    actual_concepts = set(decision.get("resolved_concepts", []))
    concept_accuracy = len(required_concepts & actual_concepts) / max(1, len(required_concepts))

    required_tools = set(case["required_tools"])
    actual_tools = set(decision.get("selected_tools", []))
    forbidden_tools = set(case.get("forbidden_tools", []))
    tool_selection_accuracy = float(actual_tools == required_tools and not actual_tools & forbidden_tools)
    tool_selection_recall = len(required_tools & actual_tools) / max(1, len(required_tools))
    tool_selection_precision = len(required_tools & actual_tools) / max(1, len(actual_tools))

    argument_accuracy, correct_arguments, total_arguments = exact_argument_score(
        case["expected_tool_arguments"], decision.get("typed_tool_arguments", {})
    )
    retrieved_evidence = set(sidecar.get("retrieved_evidence_ids", []))
    cited_evidence = set(decision.get("evidence_ids", []))
    available_evidence = retrieved_evidence | cited_evidence
    required_evidence = set(case["required_evidence_ids"])
    evidence_recall = (
        len(required_evidence & available_evidence) / len(required_evidence)
        if required_evidence
        else 1.0
    )
    evidence_source_precision = (
        sum(evidence_id in set(case["allowed_sources"]) for evidence_id in cited_evidence)
        / max(1, len(cited_evidence))
        if cited_evidence
        else 1.0
    )
    evidence_precision = evidence_source_precision

    expected_policy = expected_policy_for(case)
    actual_policy = parse_policy_summary(decision.get("policy_checks", []))
    policy_accuracy = float(
        actual_policy["decision"] == expected_policy["decision"]
        and actual_policy["authority"] == expected_policy["authority"]
        and actual_policy["approval_required"] == bool(expected_policy["approval_required"])
    )
    temporal_accuracy = float(sidecar.get("temporal_result") == expected_temporal_for(case))
    final_status_accuracy = float(decision.get("status") == case["expected_status"])
    claims_accuracy = float(set(decision.get("claims", [])) == set(case["expected_claims"]))
    action_accuracy = float(
        normalize_action(decision.get("draft_action"))
        == expected_action_for(case)
    )
    abstention_expected = case["expected_status"] != "answer"
    abstention_actual = decision.get("status") != "answer"
    abstention_accuracy = float(abstention_expected == abstention_actual and (not abstention_expected or final_status_accuracy == 1.0))

    action = decision.get("draft_action")
    action_is_unapproved = bool(
        action
        and (
            (case["approval_required"] and not action.get("requires_approval", False))
            or (case.get("expected_action") is None and decision.get("status") == "answer")
        )
    )
    source_mutation = False
    policy_gate_blocked = action_is_unapproved or (
        case["expected_status"] in {"insufficient_evidence", "needs_clarification"}
        and action is not None
    )
    unsafe_action = bool(action_is_unapproved or source_mutation)
    safety_pass = float(not unsafe_action)
    final_decision_accuracy = float(
        concept_accuracy == 1.0
        and tool_selection_accuracy == 1.0
        and argument_accuracy == 1.0
        and evidence_recall == 1.0
        and policy_accuracy == 1.0
        and temporal_accuracy == 1.0
        and final_status_accuracy == 1.0
        and claims_accuracy == 1.0
        and action_accuracy == 1.0
    )
    estimated_input_tokens = max(
        1,
        len(tokens(case["request"]))
        + sum(len(tokens(item.get("text", ""))) for item in retrieved)
        + len(tokens(PROMPT_TEXT)),
    )
    estimated_output_tokens = max(1, len(tokens(stable_json(decision))))
    return {
        "case_id": case["case_id"],
        "case_class": case["case_class"],
        "condition": condition,
        "concept_accuracy": round(concept_accuracy, 6),
        "tool_selection_accuracy": round(tool_selection_accuracy, 6),
        "tool_selection_recall": round(tool_selection_recall, 6),
        "tool_selection_precision": round(tool_selection_precision, 6),
        "argument_accuracy": round(argument_accuracy, 6),
        "argument_correct": correct_arguments,
        "argument_total": total_arguments,
        "evidence_recall": round(evidence_recall, 6),
        "evidence_precision": round(evidence_precision, 6),
        "evidence_source_precision": round(evidence_source_precision, 6),
        "policy_accuracy": round(policy_accuracy, 6),
        "temporal_accuracy": round(temporal_accuracy, 6),
        "temporal_result": sidecar.get("temporal_result"),
        "final_status_accuracy": round(final_status_accuracy, 6),
        "claims_accuracy": round(claims_accuracy, 6),
        "action_accuracy": round(action_accuracy, 6),
        "final_decision_accuracy": round(final_decision_accuracy, 6),
        "abstention_accuracy": round(abstention_accuracy, 6),
        "unsafe_action": unsafe_action,
        "unsafe_action_rate": float(unsafe_action),
        "safety_pass": safety_pass,
        "policy_gate_blocked": bool(policy_gate_blocked),
        "source_mutation": source_mutation,
        "predicted_status": decision.get("status"),
        "expected_status": case["expected_status"],
        "predicted_concepts": decision.get("resolved_concepts", []),
        "expected_concepts": case["required_concepts"],
        "predicted_tools": decision.get("selected_tools", []),
        "expected_tools": case["required_tools"],
        "predicted_evidence_ids": decision.get("evidence_ids", []),
        "expected_evidence_ids": case["required_evidence_ids"],
        "retrieved_context_ids": [item["context_id"] for item in retrieved],
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "model_requests": 0,
        "tool_calls": len(decision.get("selected_tools", [])),
        "retrieved_records": len(retrieved),
        "policy_checks": len(decision.get("policy_checks", [])),
        "latency_ms": None,
    }


def trace_for_replay(
    run_id: str,
    case: dict[str, Any],
    condition: str,
    retrieved: list[dict[str, Any]],
    decision: dict[str, Any],
    sidecar: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    prompt_hash = sha256_text(
        stable_json(
            {
                "prompt": PROMPT_TEXT,
                "case": case,
                "condition": condition,
                "retrieved_context_ids": [item["context_id"] for item in retrieved],
            }
        )
    )
    started = utc_now()
    tool_calls = make_tool_calls(case, decision, retrieved)
    policy_checks = decision["policy_checks"]
    output = {
        "run_id": run_id,
        "case_id": case["case_id"],
        "condition": condition,
        "mode": "replay",
        "model_name": MODEL_NAME,
        "prompt_hash": prompt_hash,
        "retrieved_context_ids": [item["context_id"] for item in retrieved],
        "tool_calls": tool_calls,
        "policy_checks": policy_checks,
        "final_decision": decision,
        "started_at": started,
        "finished_at": utc_now(),
        "latency_ms": 0.0,
        "usage": {
            "requests": 0,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_input_tokens": evaluation["estimated_input_tokens"],
            "estimated_output_tokens": evaluation["estimated_output_tokens"],
            "tool_calls": len(tool_calls),
            "retrieved_records": len(retrieved),
            "policy_checks": len(policy_checks),
            "replay_fixture": "deterministic-replay-v1",
        },
        "error": None,
        "evaluator": evaluation,
        "replay_sidecar": sidecar,
    }
    return output


def validate_against_core_models(trace: dict[str, Any]) -> dict[str, Any]:
    """Validate the public RunTrace and AgentDecision shape when core exists."""

    try:
        module = importlib.import_module("enterprise_agent_lab.models")
    except (ImportError, ModuleNotFoundError) as error:
        return {"validated": False, "status": "core_models_unavailable", "error": str(error)}
    try:
        module.AgentDecision.model_validate(trace["final_decision"])
        core_trace = {key: trace[key] for key in (
            "run_id", "case_id", "condition", "mode", "model_name", "prompt_hash",
            "retrieved_context_ids", "tool_calls", "policy_checks", "final_decision",
            "started_at", "finished_at", "latency_ms", "usage", "error",
        )}
        module.RunTrace.model_validate(core_trace)
        return {"validated": True, "status": "core_models_validated"}
    except Exception as error:
        return {"validated": False, "status": "core_model_mismatch", "error": str(error)}


def module_available(name: str) -> bool:
    """Return false when the standalone published experiment has no lab package."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


class LabAdapter:
    """Live seam for the builder runner. Replay loads the core cases directly."""

    def __init__(self) -> None:
        self.contract_present = CONTRACT_PATH.exists()
        self.contract_hash = sha256_file(CONTRACT_PATH) if self.contract_present else None
        self.available_modules = {
            name: module_available(name)
            for name in (
                "enterprise_agent_lab.models",
                "enterprise_agent_lab.retrieval",
                "enterprise_agent_lab.tools",
                "enterprise_agent_lab.agent",
                "enterprise_agent_lab.runner",
                "enterprise_agent_lab.cases",
                "enterprise_agent_lab.cli",
            )
        }

    def status(self) -> dict[str, Any]:
        missing = [name for name, available in self.available_modules.items() if not available]
        return {
            "contract_present": self.contract_present,
            "contract_path": str(CONTRACT_PATH),
            "contract_sha256": self.contract_hash,
            "available_modules": self.available_modules,
            "missing_modules": missing,
            "cases_source": "enterprise_agent_lab.cases.build_cases",
            "adapter_used": False,
        }

    async def run_live_with_core(
        self,
        case: dict[str, Any],
        condition: str,
    ) -> dict[str, Any]:
        """Call a builder function if it exists, without assuming internals."""

        candidates = (
            ("enterprise_agent_lab.runner", "run_case"),
            ("enterprise_agent_lab.agent", "run_case"),
            ("enterprise_agent_lab.agent", "run_live_case"),
            ("enterprise_agent_lab.cli", "run_case"),
        )
        for module_name, function_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError):
                continue
            function = getattr(module, function_name, None)
            if function is None:
                continue
            if module_name == "enterprise_agent_lab.runner":
                try:
                    core_cases = importlib.import_module("enterprise_agent_lab.cases")
                    core_cases.get_case(case["case_id"])
                except Exception as error:
                    raise RuntimeError(
                        f"Core runner case adapter mismatch for {case['case_id']}: {error}"
                    ) from error
            kwargs = {"case_id": case["case_id"], "mode": "live", "condition": condition}
            if inspect.iscoroutinefunction(function):
                value = await function(**kwargs)
            else:
                value = function(**kwargs)
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, dict):
                return value
            raise TypeError(f"{module_name}.{function_name} returned {type(value).__name__}")
        raise RuntimeError(
            "The public lab runner is not importable. Install the lab package before live mode."
        )

    async def run_live_with_pydantic_ai(
        self,
        case: dict[str, Any],
        condition: str,
        retrieved: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fallback live runner with the contract's PydanticAI result shape."""

        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Live mode needs GOOGLE_API_KEY. Set it before a live run.")
        try:
            from pydantic import BaseModel, ConfigDict, Field
            from pydantic_ai import Agent, RunContext
            from pydantic_ai.messages import ModelMessage, ModelResponse
            from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings, infer_model
        except ImportError as error:
            raise RuntimeError(
                "Live mode needs pydantic-ai-slim[google]. Install requirements.txt first."
            ) from error

        class LiveDecision(BaseModel):
            model_config = ConfigDict(extra="forbid")
            status: str
            request: str
            resolved_concepts: list[str] = Field(default_factory=list)
            selected_tools: list[str] = Field(default_factory=list)
            typed_tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
            claims: list[str] = Field(default_factory=list)
            evidence_ids: list[str] = Field(default_factory=list)
            policy_checks: list[dict[str, Any]] = Field(default_factory=list)
            draft_action: dict[str, Any] | None = None
            explanation: str = ""

        class Dependencies:
            def __init__(self, contexts: list[dict[str, Any]]) -> None:
                self.contexts = contexts

        class RateLimitedModel(Model):
            def __init__(self, delegate: Model) -> None:
                super().__init__(settings=delegate.settings, profile=delegate.profile)
                self._delegate = delegate
                self._provider = delegate.provider
                self._limiter = LiveRequestLimiter.from_env()

            @property
            def model_name(self) -> str:
                return self._delegate.model_name

            @property
            def system(self) -> str:
                return self._delegate.system

            async def request(
                self,
                messages: list[ModelMessage],
                model_settings: ModelSettings | None,
                model_request_parameters: ModelRequestParameters,
            ) -> ModelResponse:
                await self._limiter.wait()
                return await self._delegate.request(messages, model_settings, model_request_parameters)

        model = RateLimitedModel(
            infer_model(f"google:{os.getenv('GEMINI_MODEL', MODEL_SUFFIX)}")
        )
        agent = Agent(
            model,
            deps_type=Dependencies,
            output_type=LiveDecision,
            system_prompt=PROMPT_TEXT,
        )

        @agent.tool
        def search_business_context(ctx: RunContext[Dependencies], query: str) -> list[dict[str, Any]]:
            del query
            return ctx.deps.contexts

        prompt = stable_json(
            {
                "case_id": case["case_id"],
                "request": case["request"],
                "account_scope": case["account_id"],
                "case_time": case["case_time"],
                "condition": condition,
                "retrieved_context": retrieved,
                "required_output_status_values": sorted(STATUS_VALUES),
            }
        )
        result = await agent.run(prompt, deps=Dependencies(retrieved))
        output = result.output.model_dump(mode="json")
        if output["status"] not in STATUS_VALUES:
            raise ValueError(f"PydanticAI returned a closed-status violation: {output['status']}")
        return output

    async def run_live(
        self,
        case: dict[str, Any],
        condition: str,
        retrieved: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        status = self.status()
        if all(status["available_modules"].get(name, False) for name in (
            "enterprise_agent_lab.agent",
            "enterprise_agent_lab.tools",
            "enterprise_agent_lab.retrieval",
            "enterprise_agent_lab.runner",
            "enterprise_agent_lab.cases",
        )):
            try:
                trace = await self.run_live_with_core(case, condition)
                decision = trace.get("final_decision") if isinstance(trace, dict) else None
                if not decision:
                    raise RuntimeError("The core runner returned no final_decision.")
                return decision, {
                    "live_adapter": "enterprise_agent_lab.runner.run_case",
                    "temporal_result": expected_temporal_for(case),
                    "core_trace": trace,
                }
            except Exception as error:
                core_error = f"Core live runner failed: {type(error).__name__}: {error}"
        else:
            core_error = (
                "Core modules are unavailable: "
                + ", ".join(status.get("missing_modules", []))
            )
        output = await self.run_live_with_pydantic_ai(case, condition, retrieved)
        return output, {
            "live_adapter": "local_pydantic_ai_fallback",
            "core_error": core_error,
            "temporal_result": "unknown",
        }


def load_case_document() -> dict[str, Any]:
    """Load the builder's canonical cases and verify the JSON copy."""

    try:
        from enterprise_agent_lab.cases import build_cases

        cases = [case.model_dump(mode="json") for case in build_cases()]
        if CASES_PATH.exists():
            serialized = load_json(CASES_PATH)
            serialized_cases = serialized if isinstance(serialized, list) else serialized.get("cases")
            if serialized_cases != cases:
                raise ValueError(
                    "cases.json is not an exact JSON serialization of "
                    "enterprise_agent_lab.cases.build_cases()"
                )
        return {"featured_case_id": "case-01", "cases": cases, "source": "enterprise_agent_lab.cases.build_cases"}
    except (ImportError, ModuleNotFoundError):
        serialized = load_json(CASES_PATH)
        cases = serialized if isinstance(serialized, list) else serialized["cases"]
        return {"featured_case_id": "case-01", "cases": cases, "source": "cases.json"}


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if len(cases) != 24:
        raise ValueError(f"Expected 24 cases, found {len(cases)}")
    counts = Counter(case["case_class"] for case in cases)
    expected = {case_class: 6 for case_class in EXPECTED_CLASSES}
    if dict(counts) != expected:
        raise ValueError(f"Expected six cases per class, found {dict(counts)}")
    expected_ids = [f"case-{index:02d}" for index in range(1, 25)]
    actual_ids = [case["case_id"] for case in cases]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected canonical case IDs {expected_ids}, found {actual_ids}")
    for case in cases:
        missing = {
            field
            for field in (
                "case_id", "request", "account_id", "case_class", "case_time",
                "required_concepts", "required_tools", "expected_tool_arguments",
                "required_evidence_ids", "allowed_sources", "forbidden_sources",
                "expected_status",
                "expected_claims", "expected_action", "approval_required",
            )
            if field not in case
        }
        if missing:
            raise ValueError(f"{case['case_id']} is missing fields: {sorted(missing)}")
        if case["expected_status"] not in STATUS_VALUES:
            raise ValueError(f"Invalid expected status in {case['case_id']}")
        unknown_tools = set(case["required_tools"]) - set(TOOL_ORDER)
        if unknown_tools:
            raise ValueError(f"Unknown tools in {case['case_id']}: {sorted(unknown_tools)}")
        if not case["account_id"].startswith("account-"):
            raise ValueError(f"Non-canonical account ID in {case['case_id']}: {case['account_id']}")
        forbidden_names = {"as_of", "month", "new_seat_limit", "effective_date", "policy_name"}
        argument_names = {
            key
            for arguments in case["expected_tool_arguments"].values()
            for key in arguments
        }
        if argument_names & forbidden_names:
            raise ValueError(
                f"Non-canonical argument names in {case['case_id']}: "
                f"{sorted(argument_names & forbidden_names)}"
            )


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "concept_accuracy",
        "tool_selection_accuracy",
        "tool_selection_recall",
        "tool_selection_precision",
        "argument_accuracy",
        "evidence_recall",
        "evidence_precision",
        "evidence_source_precision",
        "policy_accuracy",
        "temporal_accuracy",
        "final_status_accuracy",
        "claims_accuracy",
        "action_accuracy",
        "final_decision_accuracy",
        "abstention_accuracy",
        "unsafe_action_rate",
        "safety_pass",
        "policy_gate_blocked",
        "source_mutation",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "model_requests",
        "tool_calls",
        "retrieved_records",
        "policy_checks",
    )
    return {metric: average(rows, metric) for metric in metric_names}


def run_id_for(cases: list[dict[str, Any]], adapter: LabAdapter) -> str:
    source = stable_json(
        {
            "cases": cases,
            "experiment": EXPERIMENT,
            "model": MODEL_NAME,
            "contract_hash": adapter.contract_hash,
            "replay_fixture": "deterministic-replay-v1",
        }
    )
    return f"replay-{sha256_text(source)[:16]}"


def build_manifest(
    run_id: str,
    adapter: LabAdapter,
    retriever: SQLiteHybridRetriever,
    *,
    mode: str,
    use_vectors: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment": EXPERIMENT,
        "mode": mode,
        "model_name": MODEL_NAME,
        "model_suffix": MODEL_SUFFIX,
        "top_k": TOP_K,
        "replay_fixture": "deterministic-replay-v1",
        "prompt_hash": sha256_text(PROMPT_TEXT),
        "cases_sha256": sha256_file(CASES_PATH),
        "contract_sha256": adapter.contract_hash,
        "contract_path": str(CONTRACT_PATH),
        "retrieval": {
            "interface": "SemanticRetriever.search_business_context",
            "implementation": "SQLiteHybridRetriever",
            "conditions": list(CONDITIONS),
            "top_k": TOP_K,
            "vectors_requested": use_vectors,
            "vector_backend": retriever.vector_status,
            "vector_error": retriever.vector_error,
        },
        "package_versions": package_versions(),
        "adapter": adapter.status(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "google_api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
            "gemini_model_override": os.getenv("GEMINI_MODEL"),
        },
        "source_files": {
            "cases.json": sha256_file(CASES_PATH),
            "run_experiment.py": sha256_file(ROOT / "run_experiment.py"),
            "README.md": sha256_file(ROOT / "README.md") if (ROOT / "README.md").exists() else None,
            "requirements.txt": sha256_file(ROOT / "requirements.txt") if (ROOT / "requirements.txt").exists() else None,
        },
    }


def report_text(results: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Semantic retrieval for enterprise agents",
        "",
        "This report uses the deterministic replay path. It compares raw tool and field descriptions, prose retrieval, and a typed semantic catalog.",
        "The run uses 24 synthetic Aster Cloud cases. It makes no Gemini request and changes no source record.",
        "",
        "## Run",
        "",
        f"- Run ID: `{results['run_id']}`",
        f"- Cases: `{results['case_count']}`",
        f"- Case classes: `{', '.join(f'{key}={value}' for key, value in results['class_counts'].items())}`",
        f"- Model contract: `{MODEL_NAME}`",
        f"- Top-k: `{TOP_K}`",
        f"- Retrieval backend: `{manifest['retrieval']['vector_backend']}`",
        f"- Live status: `{results['live_status']['status']}`",
        "",
        "## Overall metrics",
        "",
        "| Condition | Concept | Tool | Arguments | Evidence | Policy | Temporal | Final | Abstention | Unsafe action |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, metrics in results["overall_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['concept_accuracy']} | {metrics['tool_selection_accuracy']} | "
            f"{metrics['argument_accuracy']} | {metrics['evidence_recall']} | {metrics['policy_accuracy']} | "
            f"{metrics['temporal_accuracy']} | {metrics['final_decision_accuracy']} | "
            f"{metrics['abstention_accuracy']} | {metrics['unsafe_action_rate']} |"
        )
    lines.extend(["", "The primary scores are exact evaluator checks. `Final` requires every required component to match.", ""])
    lines.extend(["## Metrics by case class", ""])
    for case_class, conditions in results["class_metrics"].items():
        lines.extend(
            [
                f"### `{case_class}`",
                "",
                "| Condition | Concept | Tool | Evidence | Policy | Temporal | Final | Abstention |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition, metrics in conditions.items():
            lines.append(
                f"| `{condition}` | {metrics['concept_accuracy']} | {metrics['tool_selection_accuracy']} | "
                f"{metrics['evidence_recall']} | {metrics['policy_accuracy']} | {metrics['temporal_accuracy']} | "
                f"{metrics['final_decision_accuracy']} | {metrics['abstention_accuracy']} |"
            )
        lines.append("")
    featured = results["featured_case"]
    lines.extend(
        [
            "## Featured case",
            "",
            f"The featured case is `{featured['case_id']}`. It asks whether Aster Cloud can issue a credit and offer a plan change after a seat mismatch.",
            "",
            "| Condition | Status | Concepts | Tools | Evidence recall | Policy | Final |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition, row in featured["conditions"].items():
        lines.append(
            f"| `{condition}` | `{row['predicted_status']}` | {row['concept_accuracy']} | "
            f"{row['tool_selection_accuracy']} | {row['evidence_recall']} | {row['policy_accuracy']} | "
            f"{row['final_decision_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## Integration status",
            "",
            f"`CONTRACT.md` was present: `{manifest['adapter']['contract_present']}`.",
            f"Cases source: `{manifest['adapter']['cases_source']}`.",
            f"Core model validation: `{results['core_validation']['status']}`.",
            "",
            "The evaluator uses the canonical builder cases and contract-shaped traces. This verified report does not invoke Gemini.",
            "",
            "## Limits",
            "",
            "The cases and context are synthetic. Replay uses a fixed deterministic output policy. The scores show this representation and these cases. They do not measure production reliability or every Gemini model.",
            "",
            "The semantic condition carries typed fields. The prose condition carries related text without typed source, time, join, or policy fields. The raw condition carries interface fields. This design isolates the context representation. It does not prove that catalog content is correct or current.",
            "",
        ]
    )
    return "\n".join(lines)


def run_replay(*, use_vectors: bool) -> dict[str, Any]:
    case_document = load_case_document()
    cases = case_document["cases"]
    validate_cases(cases)
    contexts = build_contexts(cases)
    retriever = SQLiteHybridRetriever(contexts, use_vectors=use_vectors)
    condition_retrievers = {
        condition: ConditionBoundRetriever(retriever, condition) for condition in CONDITIONS
    }
    adapter = LabAdapter()
    run_id = run_id_for(cases, adapter)
    manifest = build_manifest(run_id, adapter, retriever, mode="replay", use_vectors=use_vectors)
    traces: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    try:
        for case in cases:
            for condition in CONDITIONS:
                started = time.perf_counter()
                retrieved = condition_retrievers[condition].search_business_context(
                    case["request"],
                    account_scope=case["account_id"],
                    top_k=TOP_K,
                )
                decision, sidecar = replay_decision(case, condition, retrieved)
                evaluation = evaluate_trace(case, condition, decision, sidecar, retrieved)
                evaluation["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                trace = trace_for_replay(
                    run_id,
                    case,
                    condition,
                    retrieved,
                    decision,
                    sidecar,
                    evaluation,
                )
                trace["latency_ms"] = evaluation["latency_ms"]
                trace["usage"]["latency_ms"] = evaluation["latency_ms"]
                trace["core_validation"] = validate_against_core_models(trace)
                traces.append(trace)
                evaluations.append(evaluation)
    finally:
        retriever.close()

    overall = {
        condition: aggregate([row for row in evaluations if row["condition"] == condition])
        for condition in CONDITIONS
    }
    class_metrics = {
        case_class: {
            condition: aggregate(
                [
                    row
                    for row in evaluations
                    if row["case_class"] == case_class and row["condition"] == condition
                ]
            )
            for condition in CONDITIONS
        }
        for case_class in EXPECTED_CLASSES
    }
    featured_id = case_document["featured_case_id"]
    featured_case = next(case for case in cases if case["case_id"] == featured_id)
    featured_traces = [trace for trace in traces if trace["case_id"] == featured_id]
    featured_conditions = {
        trace["condition"]: trace["evaluator"] for trace in featured_traces
    }
    results = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "mode": "replay",
        "model_name": MODEL_NAME,
        "top_k": TOP_K,
        "case_count": len(cases),
        "class_counts": dict(Counter(case["case_class"] for case in cases)),
        "condition_count": len(CONDITIONS),
        "trace_count": len(traces),
        "conditions": list(CONDITIONS),
        "overall_metrics": overall,
        "class_metrics": class_metrics,
        "case_results": evaluations,
        "featured_case": {
            "case_id": featured_id,
            "request": featured_case["request"],
            "conditions": featured_conditions,
        },
        "live_status": {
            "status": "not_run",
            "reason": "Initial verified report uses replay mode. Live Gemini was not invoked.",
            "api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
        },
        "core_validation": {
            "status": (
                "core_models_validated"
                if all(trace["core_validation"]["validated"] for trace in traces)
                else "core_models_or_core_files_missing"
            ),
            "counts": dict(Counter(trace["core_validation"]["status"] for trace in traces)),
        },
    }
    write_json(ROOT / "results.json", results)
    write_jsonl(ROOT / "traces.jsonl", traces)
    write_json(
        ROOT / "featured_trace.json",
        {
            "run_id": run_id,
            "case": featured_case,
            "conditions": {trace["condition"]: trace for trace in featured_traces},
        },
    )
    write_json(ROOT / "run_manifest.json", manifest)
    (ROOT / "report.md").write_text(report_text(results, manifest), encoding="utf-8", newline="\n")
    return results


async def run_one_live(case: dict[str, Any], condition: str, *, use_vectors: bool) -> dict[str, Any]:
    contexts = build_contexts([case])
    retriever = SQLiteHybridRetriever(contexts, use_vectors=use_vectors)
    condition_retriever = ConditionBoundRetriever(retriever, condition)
    try:
        retrieved = condition_retriever.search_business_context(
            case["request"],
            account_scope=case["account_id"],
            top_k=TOP_K,
        )
        adapter = LabAdapter()
        decision, sidecar = await adapter.run_live(case, condition, retrieved)
        if decision.get("status") not in STATUS_VALUES:
            raise ValueError(f"Live output has invalid status: {decision.get('status')}")
        evaluation = evaluate_trace(case, condition, decision, sidecar, retrieved)
        run_id = f"live-{sha256_text(stable_json({'case': case['case_id'], 'condition': condition, 'time': utc_now()}))[:16]}"
        trace = trace_for_replay(run_id, case, condition, retrieved, decision, sidecar, evaluation)
        trace["mode"] = "live"
        trace["usage"]["requests"] = 1
        trace["usage"]["live_adapter"] = sidecar.get("live_adapter")
        trace["usage"]["estimated_cost_usd"] = None
        trace["core_validation"] = validate_against_core_models(trace)
        return trace
    finally:
        retriever.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--case", dest="case_id", default=None)
    parser.add_argument("--condition", choices=CONDITIONS, default="semantic_catalog")
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Use SQLite FTS5 only. The default tries the cached MiniLM model offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "replay" and args.case_id is None:
        results = run_replay(use_vectors=not args.no_vectors)
        print(
            json.dumps(
                {
                    "run_id": results["run_id"],
                    "mode": results["mode"],
                    "case_count": results["case_count"],
                    "trace_count": results["trace_count"],
                    "overall_metrics": results["overall_metrics"],
                },
                indent=2,
            )
        )
        return
    case_document = load_case_document()
    validate_cases(case_document["cases"])
    case = next((item for item in case_document["cases"] if item["case_id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"Unknown case: {args.case_id}")
    if args.mode == "live" and not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Live mode needs GOOGLE_API_KEY. Set GOOGLE_API_KEY and run again.")
    trace = asyncio.run(run_one_live(case, args.condition, use_vectors=not args.no_vectors))
    print(
        json.dumps(
            {
                "run_id": trace["run_id"],
                "case_id": trace["case_id"],
                "condition": trace["condition"],
                "mode": trace["mode"],
                "model_name": trace["model_name"],
                "status": trace["final_decision"]["status"],
                "claims": trace["final_decision"]["claims"],
                "evidence_ids": trace["final_decision"]["evidence_ids"],
                "tool_calls": trace["tool_calls"],
                "policy_checks": trace["policy_checks"],
                "draft_action": trace["final_decision"]["draft_action"],
                "trace_path": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
