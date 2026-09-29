"""PydanticAI Gemini adapter for live runs."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings, infer_model

from .config import LabConfig
from .models import AgentDecision, DraftAction
from .prompts import SYSTEM_PROMPT
from .rate_limit import RequestRateLimiter
from .retrieval import SemanticRetriever
from .tools import (
    CheckPolicyOutput,
    DraftActionOutput,
    FindAccountOutput,
    GetActiveContractOutput,
    GetInvoiceOutput,
    GetSubscriptionOutput,
    GetSupportTicketsOutput,
    GetUsageRecordOutput,
    RequestHumanApprovalOutput,
    SearchBusinessConceptsOutput,
    ToolRegistry,
)


@dataclass
class AgentDependencies:
    registry: ToolRegistry
    retriever: SemanticRetriever
    account_scope: str | None
    case_time: str


class RateLimitedModel(Model):
    """Delegate to a PydanticAI model while spacing API requests."""

    def __init__(self, delegate: Model, limiter: RequestRateLimiter) -> None:
        super().__init__(settings=delegate.settings, profile=delegate.profile)
        self._delegate = delegate
        self._provider = delegate.provider
        self._limiter = limiter

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


def _tool_functions():
    def find_account(
        ctx: RunContext[AgentDependencies], account_id: str | None = None, name: str | None = None
    ) -> FindAccountOutput:
        return ctx.deps.registry.call("find_account", {"account_id": account_id, "name": name})  # type: ignore[return-value]

    def get_active_contract(
        ctx: RunContext[AgentDependencies], account_id: str, at: str | None = None
    ) -> GetActiveContractOutput:
        return ctx.deps.registry.call("get_active_contract", {"account_id": account_id, "at": at})  # type: ignore[return-value]

    def get_subscription(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        subscription_id: str | None = None,
        at: str | None = None,
    ) -> GetSubscriptionOutput:
        return ctx.deps.registry.call(
            "get_subscription",
            {"account_id": account_id, "subscription_id": subscription_id, "at": at},
        )  # type: ignore[return-value]

    def get_invoice(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        invoice_id: str | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> GetInvoiceOutput:
        return ctx.deps.registry.call(
            "get_invoice",
            {
                "account_id": account_id,
                "invoice_id": invoice_id,
                "period_start": period_start,
                "period_end": period_end,
            },
        )  # type: ignore[return-value]

    def get_usage_record(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        usage_id: str | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> GetUsageRecordOutput:
        return ctx.deps.registry.call(
            "get_usage_record",
            {
                "account_id": account_id,
                "usage_id": usage_id,
                "period_start": period_start,
                "period_end": period_end,
            },
        )  # type: ignore[return-value]

    def get_support_tickets(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        priority: str | None = None,
        status: str | None = None,
    ) -> GetSupportTicketsOutput:
        return ctx.deps.registry.call(
            "get_support_tickets",
            {"account_id": account_id, "priority": priority, "status": status},
        )  # type: ignore[return-value]

    def search_business_concepts(
        ctx: RunContext[AgentDependencies],
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = 8,
    ) -> SearchBusinessConceptsOutput:
        return ctx.deps.registry.call(
            "search_business_concepts",
            {
                "query": query,
                "account_scope": account_scope,
                "concept_type": concept_type,
                "top_k": top_k,
            },
        )  # type: ignore[return-value]

    def check_policy(
        ctx: RunContext[AgentDependencies],
        action: str,
        account_id: str,
        amount: float | None = None,
        evidence_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        at: str | None = None,
        reason: str | None = None,
        priority: str | None = None,
    ) -> CheckPolicyOutput:
        return ctx.deps.registry.call(
            "check_policy",
            {
                "action": action,
                "account_id": account_id,
                "amount": amount,
                "evidence_ids": evidence_ids or [],
                "source_ids": source_ids or [],
                "at": at,
                "reason": reason,
                "priority": priority,
            },
        )  # type: ignore[return-value]

    def draft_service_credit(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        amount: float,
        reason: str,
        evidence_ids: list[str] | None = None,
        at: str | None = None,
    ) -> DraftActionOutput:
        return ctx.deps.registry.call(
            "draft_service_credit",
            {
                "account_id": account_id,
                "amount": amount,
                "reason": reason,
                "evidence_ids": evidence_ids or [],
                "at": at,
            },
        )  # type: ignore[return-value]

    def draft_plan_change(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        target_plan: str,
        target_seats: int,
        reason: str,
        evidence_ids: list[str] | None = None,
        at: str | None = None,
    ) -> DraftActionOutput:
        return ctx.deps.registry.call(
            "draft_plan_change",
            {
                "account_id": account_id,
                "target_plan": target_plan,
                "target_seats": target_seats,
                "reason": reason,
                "evidence_ids": evidence_ids or [],
                "at": at,
            },
        )  # type: ignore[return-value]

    def draft_support_ticket(
        ctx: RunContext[AgentDependencies],
        account_id: str,
        subject: str,
        reason: str,
        priority: str = "normal",
        evidence_ids: list[str] | None = None,
    ) -> DraftActionOutput:
        return ctx.deps.registry.call(
            "draft_support_ticket",
            {
                "account_id": account_id,
                "subject": subject,
                "reason": reason,
                "priority": priority,
                "evidence_ids": evidence_ids or [],
            },
        )  # type: ignore[return-value]

    def request_human_approval(
        ctx: RunContext[AgentDependencies],
        action: DraftAction,
        reviewer: str | None = None,
        reason: str = "Policy requires human review.",
    ) -> RequestHumanApprovalOutput:
        return ctx.deps.registry.call(
            "request_human_approval",
            {"action": action, "reviewer": reviewer, "reason": reason},
        )  # type: ignore[return-value]

    return [
        find_account,
        get_active_contract,
        get_subscription,
        get_invoice,
        get_usage_record,
        get_support_tickets,
        search_business_concepts,
        check_policy,
        draft_service_credit,
        draft_plan_change,
        draft_support_ticket,
        request_human_approval,
    ]


def create_live_agent(config: LabConfig):
    """Create the PydanticAI agent only after live configuration is valid."""

    config.require_live()
    from pydantic_ai import Agent

    model = RateLimitedModel(infer_model(config.model_name), RequestRateLimiter.from_env())
    return Agent(
        model,
        deps_type=AgentDependencies,
        output_type=AgentDecision,
        instructions=SYSTEM_PROMPT,
        tools=_tool_functions(),
        retries=1,
    )
