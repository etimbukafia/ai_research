"""Typed deterministic answers and the live-generation prompt."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from compression import evidence_packet, full_context, packet_text


class Answer(BaseModel):
    """The answer contract shared by deterministic and live evaluation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answer", "insufficient_evidence", "clarification"]
    answer: str
    value: float | int | None = None
    unit: str | None = None
    product_id: str | None = None
    firmware_version: str | None = None
    region: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    evidence_path: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = ""


def compose_deterministic(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> Answer:
    selected_id = retrieval.get("selected_document_id")
    retrieval_status = retrieval.get("answer_status", "insufficient_evidence")
    if selected_id is None:
        reason = case.get("abstention_reason") or "No valid evidence path was found."
        return Answer(
            status=retrieval_status,
            answer=(
                "I cannot provide a supported pressure limit. "
                f"{reason}"
            ),
            explanation=reason,
        )

    document = documents_by_id[selected_id]
    fact = next(
        fact for fact in document["facts"] if fact["name"] == case["required_fact"]["name"]
    )
    source_ids = [selected_id]
    expected_id = case.get("applicable_document_id")
    unsupported_claims: list[str] = []
    if expected_id is not None and selected_id != expected_id:
        unsupported_claims.append(
            f"Source {selected_id} does not satisfy the requested applicability rule."
        )
    return Answer(
        status="answer",
        answer=f"The pressure limit is {fact['value']} {fact['unit']} (source {selected_id}).",
        value=fact["value"],
        unit=fact["unit"],
        product_id=document["product_ids"][0],
        firmware_version=document["firmware_versions"][0],
        region=document["regions"][0],
        source_ids=source_ids,
        evidence_path=retrieval.get("selected_evidence_path", []),
        unsupported_claims=unsupported_claims,
        explanation=(
            f"The deterministic composer copied pressure_limit from {selected_id}."
        ),
    )


def live_contexts(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    candidate_ids = [
        item["document_id"] for item in retrieval.get("candidate_documents", [])
    ]
    contexts = {"full_documents": full_context(candidate_ids, documents_by_id)}
    selected_id = retrieval.get("selected_document_id")
    if selected_id is not None:
        packet = evidence_packet(
            case,
            documents_by_id[selected_id],
            retrieval.get("selected_evidence_path", []),
        )
        contexts["evidence_packet"] = packet_text(packet)
    else:
        contexts["evidence_packet"] = "NO EVIDENCE PACKET: no valid evidence path exists."
    return contexts


def live_system_prompt(context_form: str) -> str:
    return (
        "You answer one enterprise product question from supplied evidence. "
        "Use only the evidence. Preserve the numeric value and unit. "
        "If the evidence does not contain one valid path for the requested "
        "product, firmware, region, time, status, and authority, use status "
        "insufficient_evidence or clarification. Return only JSON that matches "
        f"this schema: {json.dumps(Answer.model_json_schema(), sort_keys=True)}. "
        f"The context form is {context_form}."
    )


def live_user_prompt(case: dict[str, Any], context: str) -> str:
    return (
        f"Question: {case['question']}\n"
        f"Question time: {case['question_time']}\n"
        f"Required scope: {json.dumps(case['required_scope'], sort_keys=True)}\n"
        f"Required fact: {json.dumps(case['required_fact'], sort_keys=True)}\n"
        f"Evidence:\n{context}"
    )
