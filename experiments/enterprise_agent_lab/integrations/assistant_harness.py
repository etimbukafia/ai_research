"""Assistant-harness adapters for the enterprise lab.

The enterprise lab owns its records and retriever. The assistant harness owns
the conversation boundary, tool allow-list, session state, citation checks,
and read-assistant safety decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant_harness import (
    AssistantHarness,
    AssistantResponse,
    DeterministicAssistantProvider,
    PermissionDecision,
    PrincipalContext,
    ReadAssistantPermissionBroker,
    SkillDefinition,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    Citation,
    indirect_injection_matches,
)
from pydantic import BaseModel, ConfigDict, Field

from ..retrieval import RetrievedContext, SemanticRetriever


class EnterpriseSearchInput(BaseModel):
    """Arguments accepted by the harness-facing semantic search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    concept_type: str | None = None
    top_k: int = Field(default=8, ge=1, le=8)


BUSINESS_EVIDENCE_SKILL = SkillDefinition(
    skill_id="business_evidence",
    description="Review accessible business definitions and evidence.",
    trigger_terms=[
        "account",
        "billing",
        "billable",
        "contract",
        "credit",
        "customer",
        "invoice",
        "policy",
        "plan",
        "renew",
        "seat",
        "subscription",
        "ticket",
        "usage",
    ],
    required_sections=["summary", "evidence", "sources", "next_step"],
    allowed_tools=["search_business_concepts"],
)


class EnterprisePermissionBroker(ReadAssistantPermissionBroker):
    """Keep the harness principal inside the configured tenant."""

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def authorize(self, *, principal, skill, tool, arguments):
        if principal.tenant_id != self.tenant_id:
            return PermissionDecision(
                allowed=False,
                principal_id=principal.principal_id,
                tenant_id=principal.tenant_id,
                tool_name=tool.name,
                reason_code="tenant_not_allowed",
            )
        return super().authorize(
            principal=principal,
            skill=skill,
            tool=tool,
            arguments=arguments,
        )


def _citation(context: RetrievedContext) -> Citation:
    """Convert one enterprise context into a harness citation."""

    source_path = context.source_ids[0] if context.source_ids else context.kind
    metadata = {
        "condition": context.condition,
        "kind": context.kind,
    }
    if context.catalog_entry is not None:
        metadata["concept"] = context.catalog_entry.concept
        metadata["authority"] = context.catalog_entry.authority
    return Citation(
        citation_id=context.context_id,
        source_path=source_path,
        chunk_id=context.context_id,
        text=context.text,
        score=max(0.0, context.score),
        metadata=metadata,
    )


def _search_result(
    *,
    query: str,
    contexts: list[RetrievedContext],
) -> ToolResult:
    citations = [_citation(context) for context in contexts]
    injection_flags = [
        marker
        for context in contexts
        for marker in indirect_injection_matches(context.text)
    ]
    return ToolResult(
        tool_name="search_business_concepts",
        query=query,
        citations=citations,
        confidence=0.9 if contexts else 0.0,
        no_evidence=not contexts,
        injection_flags=list(dict.fromkeys(injection_flags)),
        metadata={
            "condition": contexts[0].condition if contexts else "none",
            "retrieved_count": str(len(contexts)),
        },
    )


def _search_handler(
    retriever: SemanticRetriever,
    account_scope: str | None,
):
    def handle(principal: PrincipalContext, arguments: BaseModel) -> ToolResult:
        del principal
        parsed = (
            arguments
            if isinstance(arguments, EnterpriseSearchInput)
            else EnterpriseSearchInput.model_validate(arguments)
        )
        contexts = retriever.search_business_context(
            parsed.query,
            account_scope=account_scope,
            concept_type=parsed.concept_type,
            top_k=parsed.top_k,
        )
        return _search_result(query=parsed.query, contexts=contexts)

    return handle


@dataclass(frozen=True)
class EnterpriseReadAssistant:
    """A principal-bound read assistant backed by the enterprise retriever."""

    harness: AssistantHarness
    principal: PrincipalContext

    def respond(self, message: str) -> AssistantResponse:
        """Run one message through the shared assistant harness."""

        return self.harness.respond(self.principal, message)


def build_enterprise_read_assistant(
    retriever: SemanticRetriever,
    *,
    account_scope: str | None,
    principal_id: str = "portfolio-user",
    tenant_id: str = "aster-cloud",
    session_id: str = "enterprise-session",
) -> EnterpriseReadAssistant:
    """Build a harness assistant for one trusted account scope.

    The account scope comes from application code. It does not come from the
    user message or from retrieved text.
    """

    principal = PrincipalContext(
        principal_id=principal_id,
        tenant_id=tenant_id,
        session_id=session_id,
    )
    search_tool = ToolDefinition(
        name="search_business_concepts",
        description="Retrieve typed business definitions and accessible evidence.",
        argument_model=EnterpriseSearchInput,
        handler=_search_handler(retriever, account_scope),
        read_only=True,
    )
    harness = AssistantHarness(
        skills=[BUSINESS_EVIDENCE_SKILL.model_copy(deep=True)],
        tools=ToolRegistry([search_tool]),
        provider=DeterministicAssistantProvider(),
        permission_broker=EnterprisePermissionBroker(tenant_id),
    )
    return EnterpriseReadAssistant(harness=harness, principal=principal)

