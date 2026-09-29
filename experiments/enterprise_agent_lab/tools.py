"""Typed read and draft-only tools for Aster Cloud."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .models import (
    Account,
    ApprovalRequest,
    Contract,
    DraftAction,
    EvidenceRef,
    Invoice,
    LabModel,
    PolicyCheck,
    SupportTicket,
    Subscription,
    UsageRecord,
)
from .policies import PolicyDenied, PolicyEngine, ScopeDenied
from .retrieval import RetrievedContext, SemanticRetriever
from .storage import SourceStore


class ToolError(RuntimeError):
    """Base error for a named tool."""


class ToolNotFound(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class FindAccountInput(LabModel):
    account_id: str | None = None
    name: str | None = None


class FindAccountOutput(LabModel):
    accounts: list[Account] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GetActiveContractInput(LabModel):
    account_id: str
    at: str | None = None


class GetActiveContractOutput(LabModel):
    contract: Contract | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GetSubscriptionInput(LabModel):
    account_id: str
    subscription_id: str | None = None
    at: str | None = None


class GetSubscriptionOutput(LabModel):
    subscriptions: list[Subscription] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GetInvoiceInput(LabModel):
    account_id: str
    invoice_id: str | None = None
    period_start: str | None = None
    period_end: str | None = None


class GetInvoiceOutput(LabModel):
    invoices: list[Invoice] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GetUsageRecordInput(LabModel):
    account_id: str
    usage_id: str | None = None
    period_start: str | None = None
    period_end: str | None = None


class GetUsageRecordOutput(LabModel):
    records: list[UsageRecord] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GetSupportTicketsInput(LabModel):
    account_id: str
    priority: str | None = None
    status: str | None = None


class GetSupportTicketsOutput(LabModel):
    tickets: list[SupportTicket] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class SearchBusinessConceptsInput(LabModel):
    query: str
    account_scope: str | None = None
    concept_type: str | None = None
    top_k: int = Field(default=8, ge=1, le=8)


class SearchBusinessConceptsOutput(LabModel):
    contexts: list[RetrievedContext] = Field(default_factory=list)


class CheckPolicyInput(LabModel):
    action: str
    account_id: str
    amount: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    at: str | None = None
    reason: str | None = None
    priority: str | None = None


class CheckPolicyOutput(LabModel):
    checks: list[PolicyCheck] = Field(default_factory=list)
    approval_required: bool = False


class DraftServiceCreditInput(LabModel):
    account_id: str
    amount: float = Field(gt=0)
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    at: str | None = None


class DraftPlanChangeInput(LabModel):
    account_id: str
    target_plan: str
    target_seats: int = Field(gt=0)
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    at: str | None = None


class DraftSupportTicketInput(LabModel):
    account_id: str
    subject: str
    reason: str
    priority: str = "normal"
    evidence_ids: list[str] = Field(default_factory=list)


class DraftActionOutput(LabModel):
    action: DraftAction
    checks: list[PolicyCheck] = Field(default_factory=list)


class RequestHumanApprovalInput(LabModel):
    action: DraftAction
    reviewer: str | None = None
    reason: str = "Policy requires human review."


class RequestHumanApprovalOutput(LabModel):
    request: ApprovalRequest


@dataclass(frozen=True)
class ToolSpec:
    name: str
    purpose: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    required_concepts: tuple[str, ...]
    side_effect_level: str
    approval_requirement: str
    handler: Callable[..., BaseModel]


def _evidence(
    source_type: str,
    source_id: str,
    excerpt: str,
    authority: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=source_id,
        source_type=source_type,
        source_id=source_id,
        excerpt=excerpt,
        authority=authority,
        valid_from=valid_from,
        valid_to=valid_to,
    )


class ToolRegistry:
    """Registry that validates every call before it reaches a local handler."""

    def __init__(
        self,
        store: SourceStore,
        retriever: SemanticRetriever,
        policy_engine: PolicyEngine,
        *,
        review_dir: str | Path | None = None,
        account_scope: str | None = None,
        case_time: str = "2026-05-31",
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.policy_engine = policy_engine
        self.review_dir = Path(review_dir) if review_dir else Path("reviews")
        self.account_scope = account_scope
        self.case_time = case_time
        self.call_records: list[dict[str, Any]] = []
        self._specs = self._make_specs()

    def _make_specs(self) -> dict[str, ToolSpec]:
        return {
            "find_account": ToolSpec(
                "find_account", "Find an account in CRM.", FindAccountInput, FindAccountOutput,
                ("active_account",), "none", "none", self._find_account
            ),
            "get_active_contract": ToolSpec(
                "get_active_contract", "Read the contract valid at a time.",
                GetActiveContractInput, GetActiveContractOutput, ("active_contract",),
                "none", "none", self._get_active_contract
            ),
            "get_subscription": ToolSpec(
                "get_subscription", "Read a subscription.",
                GetSubscriptionInput, GetSubscriptionOutput, ("subscription_status",),
                "none", "none", self._get_subscription
            ),
            "get_invoice": ToolSpec(
                "get_invoice", "Read billing invoices.",
                GetInvoiceInput, GetInvoiceOutput, ("invoice_period",),
                "none", "none", self._get_invoice
            ),
            "get_usage_record": ToolSpec(
                "get_usage_record", "Read usage for an account period.",
                GetUsageRecordInput, GetUsageRecordOutput, ("usage_peak",),
                "none", "none", self._get_usage_record
            ),
            "get_support_tickets": ToolSpec(
                "get_support_tickets", "Read support tickets.",
                GetSupportTicketsInput, GetSupportTicketsOutput, ("priority_ticket",),
                "none", "none", self._get_support_tickets
            ),
            "search_business_concepts": ToolSpec(
                "search_business_concepts", "Retrieve business context.",
                SearchBusinessConceptsInput, SearchBusinessConceptsOutput,
                ("source_authority",), "none", "none", self._search_business_concepts
            ),
            "check_policy": ToolSpec(
                "check_policy", "Check a policy before a draft action.",
                CheckPolicyInput, CheckPolicyOutput, ("service_credit",),
                "none", "policy", self._check_policy
            ),
            "draft_service_credit": ToolSpec(
                "draft_service_credit", "Create a local service-credit draft.",
                DraftServiceCreditInput, DraftActionOutput, ("service_credit",),
                "draft", "policy", self._draft_service_credit
            ),
            "draft_plan_change": ToolSpec(
                "draft_plan_change", "Create a local plan-change draft.",
                DraftPlanChangeInput, DraftActionOutput, ("approved_plan_change",),
                "draft", "policy", self._draft_plan_change
            ),
            "draft_support_ticket": ToolSpec(
                "draft_support_ticket", "Create a local support-ticket draft.",
                DraftSupportTicketInput, DraftActionOutput, ("priority_ticket",),
                "draft", "policy", self._draft_support_ticket
            ),
            "request_human_approval": ToolSpec(
                "request_human_approval", "Write a local review request.",
                RequestHumanApprovalInput, RequestHumanApprovalOutput, (),
                "local_write", "approval", self._request_human_approval
            ),
        }

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def _scope(self, account_id: str | None) -> None:
        check = self.policy_engine.check_scope(account_id)
        if not check.passed:
            raise ScopeDenied(check.reason)

    def _record(self, name: str, arguments: BaseModel, result: BaseModel, side_effect: str) -> None:
        data = result.model_dump(mode="json")
        source_ids: set[str] = set()
        for key in ("source_id", "source_ids", "evidence_ids"):
            value = data.get(key)
            if isinstance(value, str):
                source_ids.add(value)
            elif isinstance(value, list):
                source_ids.update(str(item) for item in value)
        for item in data.get("evidence", []) if isinstance(data.get("evidence"), list) else []:
            if isinstance(item, dict) and item.get("source_id"):
                source_ids.add(str(item["source_id"]))
        self.call_records.append(
            {
                "call_id": f"call-{len(self.call_records) + 1:03d}",
                "tool_name": name,
                "arguments": arguments.model_dump(mode="json"),
                "result_summary": f"{name} returned {len(json.dumps(data))} JSON characters",
                "source_ids": sorted(source_ids),
                "side_effect_level": side_effect,
                "approved": None,
            }
        )

    def call(self, name: str, arguments: BaseModel | dict[str, Any]) -> BaseModel:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolNotFound(f"Unknown tool: {name}")
        try:
            parsed = (
                arguments
                if isinstance(arguments, spec.input_model)
                else spec.input_model.model_validate(arguments)
            )
        except Exception as exc:
            raise ToolValidationError(f"{name} arguments are invalid: {exc}") from exc
        result = spec.handler(parsed)
        self._record(name, parsed, result, spec.side_effect_level)
        return result

    def _find_account(self, arguments: FindAccountInput) -> FindAccountOutput:
        accounts = self.store.find_account(account_id=arguments.account_id, name=arguments.name)
        accounts = [account for account in accounts if not self.account_scope or account.account_id == self.account_scope]
        return FindAccountOutput(
            accounts=accounts,
            evidence=[
                _evidence("crm", account.source_id, f"{account.name} has status {account.status}.", "crm")
                for account in accounts
            ],
        )

    def _get_active_contract(self, arguments: GetActiveContractInput) -> GetActiveContractOutput:
        self._scope(arguments.account_id)
        contract = self.store.get_active_contract(arguments.account_id, arguments.at or self.case_time)
        return GetActiveContractOutput(
            contract=contract,
            evidence=(
                [
                    _evidence(
                        "contract",
                        contract.source_id,
                        f"{contract.plan} allows {contract.seat_limit} seats.",
                        contract.authority,
                        contract.effective_from,
                        contract.effective_to,
                    )
                ]
                if contract
                else []
            ),
        )

    def _get_subscription(self, arguments: GetSubscriptionInput) -> GetSubscriptionOutput:
        self._scope(arguments.account_id)
        subscriptions = self.store.get_subscription(
            arguments.account_id, arguments.subscription_id, arguments.at or self.case_time
        )
        return GetSubscriptionOutput(
            subscriptions=subscriptions,
            evidence=[
                _evidence("subscription", item.source_id, f"{item.plan} subscription is {item.status}.", "billing")
                for item in subscriptions
            ],
        )

    def _get_invoice(self, arguments: GetInvoiceInput) -> GetInvoiceOutput:
        self._scope(arguments.account_id)
        invoices = self.store.get_invoice(
            arguments.account_id, arguments.invoice_id, arguments.period_start, arguments.period_end
        )
        return GetInvoiceOutput(
            invoices=invoices,
            evidence=[
                _evidence("invoice", item.source_id, f"Invoice amount is {item.amount:.2f} {item.currency}.", "billing",
                          item.period_start, item.period_end)
                for item in invoices
            ],
        )

    def _get_usage_record(self, arguments: GetUsageRecordInput) -> GetUsageRecordOutput:
        self._scope(arguments.account_id)
        records = self.store.get_usage_record(
            arguments.account_id, arguments.usage_id, arguments.period_start, arguments.period_end
        )
        return GetUsageRecordOutput(
            records=records,
            evidence=[
                _evidence("usage", item.source_id, f"Peak seats were {item.peak_seats}.", "usage",
                          item.period_start, item.period_end)
                for item in records
            ],
        )

    def _get_support_tickets(self, arguments: GetSupportTicketsInput) -> GetSupportTicketsOutput:
        self._scope(arguments.account_id)
        tickets = self.store.get_support_tickets(arguments.account_id, arguments.priority, arguments.status)
        return GetSupportTicketsOutput(
            tickets=tickets,
            evidence=[
                _evidence("support_ticket", item.source_id, f"{item.priority} ticket: {item.subject}.", "support")
                for item in tickets
            ],
        )

    def _search_business_concepts(self, arguments: SearchBusinessConceptsInput) -> SearchBusinessConceptsOutput:
        contexts = self.retriever.search_business_context(
            arguments.query, arguments.account_scope or self.account_scope,
            arguments.concept_type, arguments.top_k
        )
        return SearchBusinessConceptsOutput(contexts=contexts)

    def _check_policy(self, arguments: CheckPolicyInput) -> CheckPolicyOutput:
        self._scope(arguments.account_id)
        checks, approval_required = self.policy_engine.check_action(
            arguments.action,
            arguments.account_id,
            amount=arguments.amount,
            evidence=arguments.evidence_ids,
            source_ids=arguments.source_ids,
            at=arguments.at,
            reason=arguments.reason,
            priority=arguments.priority,
        )
        return CheckPolicyOutput(checks=checks, approval_required=approval_required)

    def _draft_service_credit(self, arguments: DraftServiceCreditInput) -> DraftActionOutput:
        self._scope(arguments.account_id)
        checks, approval_required = self.policy_engine.check_action(
            "service_credit", arguments.account_id, amount=arguments.amount,
            evidence=arguments.evidence_ids, at=arguments.at
        )
        PolicyEngine.assert_allowed(checks)
        action = DraftAction(
            action_id=f"draft-{uuid.uuid4().hex[:10]}",
            action_type="service_credit",
            account_id=arguments.account_id,
            payload={"amount": arguments.amount, "reason": arguments.reason},
            status="pending_approval" if approval_required else "draft",
            requires_approval=approval_required,
            policy_check_ids=[check.policy_id for check in checks],
            evidence_ids=arguments.evidence_ids,
        )
        return DraftActionOutput(action=action, checks=checks)

    def _draft_plan_change(self, arguments: DraftPlanChangeInput) -> DraftActionOutput:
        self._scope(arguments.account_id)
        checks, approval_required = self.policy_engine.check_action(
            "plan_change", arguments.account_id, evidence=arguments.evidence_ids, at=arguments.at
        )
        PolicyEngine.assert_allowed(checks)
        action = DraftAction(
            action_id=f"draft-{uuid.uuid4().hex[:10]}",
            action_type="plan_change",
            account_id=arguments.account_id,
            payload={
                "target_plan": arguments.target_plan,
                "target_seats": arguments.target_seats,
                "reason": arguments.reason,
            },
            status="pending_approval",
            requires_approval=approval_required,
            policy_check_ids=[check.policy_id for check in checks],
            evidence_ids=arguments.evidence_ids,
        )
        return DraftActionOutput(action=action, checks=checks)

    def _draft_support_ticket(self, arguments: DraftSupportTicketInput) -> DraftActionOutput:
        self._scope(arguments.account_id)
        checks, approval_required = self.policy_engine.check_action(
            "support_ticket", arguments.account_id, evidence=arguments.evidence_ids,
            reason=arguments.reason, priority=arguments.priority
        )
        PolicyEngine.assert_allowed(checks)
        action = DraftAction(
            action_id=f"draft-{uuid.uuid4().hex[:10]}",
            action_type="support_ticket",
            account_id=arguments.account_id,
            payload={
                "subject": arguments.subject,
                "reason": arguments.reason,
                "priority": arguments.priority,
            },
            status="pending_approval" if approval_required else "draft",
            requires_approval=approval_required,
            policy_check_ids=[check.policy_id for check in checks],
            evidence_ids=arguments.evidence_ids,
        )
        return DraftActionOutput(action=action, checks=checks)

    def _request_human_approval(
        self, arguments: RequestHumanApprovalInput
    ) -> RequestHumanApprovalOutput:
        request = ApprovalRequest(
            request_id=f"approval-{uuid.uuid4().hex[:10]}",
            action_id=arguments.action.action_id,
            reviewer=arguments.reviewer,
            reason=arguments.reason,
        )
        self.review_dir.mkdir(parents=True, exist_ok=True)
        path = self.review_dir / f"{request.request_id}.json"
        path.write_text(
            json.dumps({"request": request.model_dump(mode="json"), "action": arguments.action.model_dump(mode="json")},
                       indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return RequestHumanApprovalOutput(request=request)

