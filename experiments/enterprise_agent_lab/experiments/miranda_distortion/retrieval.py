"""Small local context retriever for the three replay configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RetrievedContext:
    context_id: str
    source_id: str
    kind: str
    text: str
    score: float


def _contract() -> dict[str, Any]:
    return json.loads((EXPERIMENT_DIR / "semantic_contract.json").read_text(encoding="utf-8"))


def _concept_text(concept: dict[str, Any]) -> str:
    names = ", ".join(str(item) for item in concept.get("names", []))
    distinct = ", ".join(str(item) for item in concept.get("related_but_distinct", []))
    return (
        f"Canonical concept {concept['id']}; names: {names}; "
        f"authority: {concept.get('source_authority')}; "
        f"validity: {concept.get('validity_rule')}; "
        f"distinct concepts: {distinct or 'none'}."
    )


def retrieve_context(case: dict[str, Any], condition: str) -> list[RetrievedContext]:
    """Return deterministic context for a case.

    The raw-record condition returns only a compact record index.  Retrieval
    adds prose policy context.  Semantic adds the versioned contract entries.
    No embeddings or external service are required.
    """

    if condition not in {"raw-record", "retrieval", "semantic"}:
        raise ValueError(f"Unknown condition: {condition}")

    request = str(case.get("request", ""))
    input_data = case.get("input", {})
    context: list[RetrievedContext] = [
        RetrievedContext(
            context_id="raw-record-index",
            source_id="northstar-record-index",
            kind="raw_records",
            text=(
                f"Northstar record index for request: {request}. "
                f"Employee={input_data.get('employee_id')}; "
                f"role_query={input_data.get('role_query')}; "
                f"scope={input_data.get('scope_id')}."
            ),
            score=1.0,
        )
    ]
    if condition in {"retrieval", "semantic"}:
        context.append(
            RetrievedContext(
                context_id="retrieved-policy-summary",
                source_id="policy-retrieval-summary",
                kind="policy_text",
                text=(
                    "Retrieved policy text: HR owns worker type; IAM owns role and active access; "
                    "the approvals system owns approval type and status; Security owns access policy. "
                    "A ticket is not proof of every approval. Use records valid at the case time."
                ),
                score=0.95,
            )
        )
    if condition == "semantic":
        contract = _contract()
        concepts = contract.get("canonical_concepts", [])
        query = f"{request} {input_data.get('role_query', '')}".casefold()
        selected: list[dict[str, Any]] = []
        for concept in concepts:
            haystack = " ".join(
                [str(concept.get("id", "")), *[str(item) for item in concept.get("names", [])]]
            ).casefold()
            if any(token in haystack for token in query.split() if len(token) > 3):
                selected.append(concept)
        # The core rules are always included.  This keeps the semantic
        # condition deterministic and makes the intervention inspectable.
        selected_ids = {str(item.get("id")) for item in selected}
        for concept_id in (
            "role.finance.report_viewer",
            "role.finance.production_export",
            "scope.finance.production",
            "approval.manager",
            "approval.security",
            "approval.sponsor",
        ):
            match = next((item for item in concepts if item.get("id") == concept_id), None)
            if match is not None and concept_id not in selected_ids:
                selected.append(match)
        for concept in selected:
            context.append(
                RetrievedContext(
                    context_id=f"contract:{concept['id']}",
                    source_id="semantic-contract-v1",
                    kind="semantic_contract",
                    text=_concept_text(concept),
                    score=1.0,
                )
            )
        context.append(
            RetrievedContext(
                context_id="contract:rules",
                source_id="semantic-contract-v1",
                kind="semantic_contract_rules",
                text="; ".join(str(rule["statement"]) for rule in contract.get("rules", [])),
                score=1.0,
            )
        )
    return context
