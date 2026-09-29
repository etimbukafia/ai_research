"""Stable Pydantic contracts for the enterprise agent lab."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LabModel(BaseModel):
    """Base model that rejects fields which the lab does not know."""

    model_config = ConfigDict(extra="forbid")


class Account(LabModel):
    account_id: str
    name: str
    status: str
    region: str
    owner: str
    valid_from: str
    valid_to: str | None = None
    source_id: str


class Contract(LabModel):
    contract_id: str
    account_id: str
    plan: str
    seat_limit: int
    seat_billing_rule: str
    effective_from: str
    effective_to: str | None = None
    renewal_date: str
    service_credit_allowed: bool
    source_id: str
    authority: str


class Subscription(LabModel):
    subscription_id: str
    account_id: str
    contract_id: str
    plan: str
    status: str
    seats: int
    started_at: str
    ended_at: str | None = None
    source_id: str


class Invoice(LabModel):
    invoice_id: str
    account_id: str
    subscription_id: str
    period_start: str
    period_end: str
    amount: float
    currency: str
    seat_count: int
    status: str
    source_id: str


class UsageRecord(LabModel):
    usage_id: str
    account_id: str
    subscription_id: str
    period_start: str
    period_end: str
    peak_seats: int
    active_users: int
    source_id: str


class SupportTicket(LabModel):
    ticket_id: str
    account_id: str
    subject: str
    priority: str
    status: str
    created_at: str
    source_id: str


class PolicyRule(LabModel):
    policy_id: str
    name: str
    action: str
    condition: str
    threshold: float | None = None
    approval_required: bool
    authority: str
    valid_from: str
    valid_to: str | None = None
    source_id: str


class CatalogEntry(LabModel):
    catalog_id: str
    concept: str
    concept_type: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    entity: str
    grain: str
    maps_to: list[str] = Field(default_factory=list)
    allowed_joins: list[str] = Field(default_factory=list)
    time_basis: str
    authority: str
    valid_from: str
    valid_to: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    do_not_use: list[str] = Field(default_factory=list)


class EvidenceRef(LabModel):
    evidence_id: str
    source_type: str
    source_id: str
    excerpt: str
    authority: str
    valid_from: str | None = None
    valid_to: str | None = None


class PolicyCheck(LabModel):
    policy_id: str
    name: str
    passed: bool
    reason: str
    approval_required: bool = False
    source_id: str | None = None


class ToolCallRecord(LabModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str
    source_ids: list[str] = Field(default_factory=list)
    side_effect_level: str
    approved: bool | None = None


class DraftAction(LabModel):
    action_id: str
    action_type: str
    account_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"
    requires_approval: bool = False
    policy_check_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ApprovalRequest(LabModel):
    request_id: str
    action_id: str
    reviewer: str | None = None
    status: Literal["requested", "approved", "rejected"] = "requested"
    reason: str


AgentStatus = Literal[
    "answer",
    "needs_clarification",
    "needs_human_review",
    "insufficient_evidence",
]


class AgentDecision(LabModel):
    status: AgentStatus
    request: str
    resolved_concepts: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    typed_tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    claims: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    policy_checks: list[PolicyCheck] = Field(default_factory=list)
    draft_action: DraftAction | None = None
    explanation: str = ""


class RunTrace(LabModel):
    run_id: str
    case_id: str | None = None
    condition: str
    mode: str
    model_name: str
    prompt_hash: str
    retrieved_context_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    policy_checks: list[PolicyCheck] = Field(default_factory=list)
    final_decision: AgentDecision | None = None
    started_at: str
    finished_at: str | None = None
    latency_ms: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationCase(LabModel):
    case_id: str
    request: str
    account_id: str | None = None
    case_class: str
    case_time: str
    required_concepts: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    expected_tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required_evidence_ids: list[str] = Field(default_factory=list)
    allowed_sources: list[str] = Field(default_factory=list)
    forbidden_sources: list[str] = Field(default_factory=list)
    expected_status: AgentStatus
    expected_claims: list[str] = Field(default_factory=list)
    expected_action: str | None = None
    approval_required: bool = False


class RunUsage(LabModel):
    requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int = 0
    retrieved_records: int = 0
    policy_checks: int = 0
    errors: int = 0

