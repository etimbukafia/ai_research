"""Typed mock tools for the software-access request service."""

from __future__ import annotations

from typing import Any

from enterprise_agent_harness import (
    AgentLifecycleStatus,
    EvidenceRef,
    ExecutionContext,
    PolicyDefinition,
    PolicyEffect,
    PolicyRule,
    RiskLevel,
    ToolDefinition,
    ToolKind,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
)
from pydantic import BaseModel, Field

from .data_adapter import BusinessData


class ResolvePrincipalInput(BaseModel):
    principal_id: str
    tenant_id: str


class ResolvePrincipalOutput(BaseModel):
    found: bool
    eligible: bool
    principal_id: str
    tenant_id: str
    role: str | None = None
    team_id: str | None = None
    status: str | None = None
    reason: str


class ProductSearchInput(BaseModel):
    query: str
    tenant_id: str


class ProductSearchOutput(BaseModel):
    found: bool
    ambiguous: bool
    products: list[dict[str, Any]] = Field(default_factory=list)
    reason: str


class PolicyInput(BaseModel):
    product_id: str
    action: str
    tenant_id: str


class PolicyOutput(BaseModel):
    found: bool
    product_id: str
    action: str
    allows: bool
    requires_approval: bool
    policy_id: str | None = None
    policy_version: str | None = None
    reason: str


class BudgetInput(BaseModel):
    tenant_id: str
    team_id: str
    product_id: str


class BudgetOutput(BaseModel):
    found: bool
    available: bool
    remaining: int = 0
    limit: int = 0
    used: int = 0
    reason: str


class AccessRequestInput(BaseModel):
    requester_id: str
    tenant_id: str
    team_id: str
    product_id: str
    idempotency_key: str
    justification: str


class AccessRequestOutput(BaseModel):
    created: bool
    existing: bool
    request_id: str
    status: str
    reason: str


class GrantAccessInput(BaseModel):
    requester_id: str
    tenant_id: str
    team_id: str
    product_id: str
    idempotency_key: str
    justification: str


class GrantAccessOutput(BaseModel):
    granted: bool
    grant_id: str
    reason: str


class HandoffInput(BaseModel):
    requester_id: str
    tenant_id: str
    reason: str


class HandoffOutput(BaseModel):
    recorded: bool
    handoff_id: str
    reason: str


class AccessRequestApp:
    """Own business data, side effects, and per-case handler evidence."""

    def __init__(self, data: BusinessData) -> None:
        self.data = data
        self.requests: dict[str, dict[str, Any]] = {}
        self.handler_calls: list[dict[str, Any]] = []
        self._handoff_count = 0

    def reset_case(self) -> None:
        self.handler_calls = []

    def tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        tools = [
            ToolDefinition(
                "resolve_principal",
                "1.0.0",
                "Resolve one principal inside the execution tenant.",
                ResolvePrincipalInput,
                ResolvePrincipalOutput,
                self.resolve_principal,
                kind=ToolKind.READ,
                risk_level=RiskLevel.LOW,
                required_permissions=("identity.read",),
            ),
            ToolDefinition(
                "search_products",
                "1.0.0",
                "Find products in the requested tenant and report ambiguity.",
                ProductSearchInput,
                ProductSearchOutput,
                self.search_products,
                kind=ToolKind.READ,
                risk_level=RiskLevel.LOW,
                required_permissions=("catalog.read",),
            ),
            ToolDefinition(
                "get_current_policy",
                "1.0.0",
                "Resolve the current policy for one product and action.",
                PolicyInput,
                PolicyOutput,
                self.get_current_policy,
                kind=ToolKind.READ,
                risk_level=RiskLevel.LOW,
                required_permissions=("policy.read",),
            ),
            ToolDefinition(
                "check_budget",
                "1.0.0",
                "Check the current team budget for one product.",
                BudgetInput,
                BudgetOutput,
                self.check_budget,
                kind=ToolKind.READ,
                risk_level=RiskLevel.LOW,
                required_permissions=("budget.read",),
            ),
            ToolDefinition(
                "create_access_request",
                "1.0.0",
                "Create one pending access request with an idempotency key.",
                AccessRequestInput,
                AccessRequestOutput,
                self.create_access_request,
                kind=ToolKind.WRITE,
                risk_level=RiskLevel.MEDIUM,
                required_permissions=("access.write",),
                idempotency_required=True,
                sensitive_argument_fields=("justification",),
            ),
            ToolDefinition(
                "grant_access",
                "1.0.0",
                "Grant access after an exact human approval.",
                GrantAccessInput,
                GrantAccessOutput,
                self.grant_access,
                kind=ToolKind.ACTION,
                risk_level=RiskLevel.HIGH,
                required_permissions=("access.grant",),
                requires_approval=True,
                idempotency_required=True,
                sensitive_argument_fields=("justification",),
            ),
            ToolDefinition(
                "record_handoff",
                "1.0.0",
                "Record a human review handoff without granting access.",
                HandoffInput,
                HandoffOutput,
                self.record_handoff,
                kind=ToolKind.ACTION,
                risk_level=RiskLevel.LOW,
                required_permissions=("handoff.write",),
            ),
        ]
        for tool in tools:
            registry.register(tool)
        return registry

    def resolve_principal(
        self,
        context: ExecutionContext,
        arguments: ResolvePrincipalInput,
    ) -> ToolResult:
        self._record("resolve_principal", context, arguments.model_dump(mode="json"))
        record = self.data.principal(arguments.principal_id)
        allowed = record is not None and record["tenant_id"] == context.principal.tenant_id
        eligible = bool(allowed and record["status"] == "active" and record["role"] == "finance_admin")
        evidence_id = f"principal:{arguments.principal_id}:source"
        output = ResolvePrincipalOutput(
            found=record is not None,
            eligible=eligible,
            principal_id=arguments.principal_id,
            tenant_id=record["tenant_id"] if record else arguments.tenant_id,
            role=record.get("role") if record else None,
            team_id=record.get("team_id") if record else None,
            status=record.get("status") if record else None,
            reason=(
                "Principal is active and has the finance admin role."
                if eligible
                else "Principal is missing, outside the tenant, inactive, or lacks the required role."
            ),
        )
        return ToolResult(
            tool_id="resolve_principal",
            status=ToolResultStatus.SUCCEEDED,
            output=output,
            evidence=[EvidenceRef(evidence_id=evidence_id, source="principals.json", kind="source_record")],
        )

    def search_products(
        self,
        context: ExecutionContext,
        arguments: ProductSearchInput,
    ) -> ToolResult:
        self._record("search_products", context, arguments.model_dump(mode="json"))
        if arguments.tenant_id != context.principal.tenant_id:
            output = ProductSearchOutput(
                found=False,
                ambiguous=False,
                reason="The requested tenant differs from the principal tenant.",
            )
            return ToolResult(
                tool_id="search_products",
                status=ToolResultStatus.RESTRICTED,
                restricted=True,
                output=output,
                evidence=[EvidenceRef(evidence_id="tenant-boundary:product-search", source="application_policy", kind="policy")],
            )
        products = self.data.products_for(arguments.tenant_id, arguments.query)
        output = ProductSearchOutput(
            found=bool(products),
            ambiguous=len(products) > 1,
            products=products,
            reason=(
                "One product matched."
                if len(products) == 1
                else "The product name is ambiguous or no product matched."
            ),
        )
        return ToolResult(
            tool_id="search_products",
            status=ToolResultStatus.SUCCEEDED,
            output=output,
            evidence=[EvidenceRef(evidence_id="products:search", source="products.json", kind="source_record")],
        )

    def get_current_policy(
        self,
        context: ExecutionContext,
        arguments: PolicyInput,
    ) -> ToolResult:
        self._record("get_current_policy", context, arguments.model_dump(mode="json"))
        policy = self.data.current_policy(
            arguments.product_id,
            arguments.action,
            arguments.tenant_id,
        )
        if arguments.tenant_id != context.principal.tenant_id:
            policy = None
        output = PolicyOutput(
            found=policy is not None,
            product_id=arguments.product_id,
            action=arguments.action,
            allows=bool(policy and policy["effect"] == "allow"),
            requires_approval=bool(policy and policy["requires_approval"]),
            policy_id=policy.get("policy_id") if policy else None,
            policy_version=policy.get("version") if policy else None,
            reason=(policy["reason"] if policy else "No current policy applies."),
        )
        return ToolResult(
            tool_id="get_current_policy",
            status=ToolResultStatus.SUCCEEDED,
            output=output,
            evidence=[EvidenceRef(evidence_id=f"policy:{arguments.product_id}:{arguments.action}", source="policies.json", kind="policy_record")],
        )

    def check_budget(self, context: ExecutionContext, arguments: BudgetInput) -> ToolResult:
        self._record("check_budget", context, arguments.model_dump(mode="json"))
        budget = self.data.budget(arguments.tenant_id, arguments.team_id, arguments.product_id)
        if arguments.tenant_id != context.principal.tenant_id:
            budget = None
        remaining = (budget["limit"] - budget["used"]) if budget else 0
        output = BudgetOutput(
            found=budget is not None,
            available=bool(budget and remaining > 0),
            remaining=remaining,
            limit=budget["limit"] if budget else 0,
            used=budget["used"] if budget else 0,
            reason=(
                "Budget has capacity."
                if budget and remaining > 0
                else "No budget record exists or the budget has no capacity."
            ),
        )
        return ToolResult(
            tool_id="check_budget",
            status=ToolResultStatus.SUCCEEDED,
            output=output,
            evidence=[EvidenceRef(evidence_id=f"budget:{arguments.team_id}:{arguments.product_id}", source="budgets.json", kind="source_record")],
        )

    def create_access_request(
        self,
        context: ExecutionContext,
        arguments: AccessRequestInput,
    ) -> ToolResult:
        self._record("create_access_request", context, arguments.model_dump(mode="json"))
        principal = self.data.principal(arguments.requester_id)
        product = self.data.product(arguments.product_id)
        policy = self.data.current_policy(arguments.product_id, "create_access_request", arguments.tenant_id)
        budget = self.data.budget(arguments.tenant_id, arguments.team_id, arguments.product_id)
        valid = bool(
            principal
            and principal["tenant_id"] == context.principal.tenant_id
            and principal["status"] == "active"
            and principal["role"] == "finance_admin"
            and product
            and product["tenant_id"] == context.principal.tenant_id
            and policy
            and policy["effect"] == "allow"
            and budget
            and budget["limit"] > budget["used"]
        )
        existing = self.requests.get(arguments.idempotency_key)
        if existing is not None:
            output = AccessRequestOutput(
                created=False,
                existing=True,
                request_id=existing["request_id"],
                status=existing["status"],
                reason="The idempotency key already identifies a pending request.",
            )
        elif valid:
            request = {
                "request_id": arguments.idempotency_key,
                "requester_id": arguments.requester_id,
                "tenant_id": arguments.tenant_id,
                "team_id": arguments.team_id,
                "product_id": arguments.product_id,
                "status": "pending_approval",
            }
            self.requests[arguments.idempotency_key] = request
            output = AccessRequestOutput(
                created=True,
                existing=False,
                request_id=arguments.idempotency_key,
                status="pending_approval",
                reason="The access request was created and awaits the policy-defined review path.",
            )
        else:
            output = AccessRequestOutput(
                created=False,
                existing=False,
                request_id=arguments.idempotency_key,
                status="rejected",
                reason="The application data or current policy does not allow this request.",
            )
        return ToolResult(
            tool_id="create_access_request",
            status=ToolResultStatus.SUCCEEDED,
            output=output,
            evidence=[EvidenceRef(evidence_id=f"request:{arguments.idempotency_key}", source="requests", kind="application_state")],
        )

    def grant_access(self, context: ExecutionContext, arguments: GrantAccessInput) -> ToolResult:
        self._record("grant_access", context, arguments.model_dump(mode="json"))
        return ToolResult(
            tool_id="grant_access",
            status=ToolResultStatus.SUCCEEDED,
            output=GrantAccessOutput(
                granted=True,
                grant_id=f"grant-{arguments.idempotency_key}",
                reason="Mock grant handler ran.",
            ),
            evidence=[EvidenceRef(evidence_id=f"grant:{arguments.idempotency_key}", source="mock_grant_handler", kind="side_effect")],
        )

    def record_handoff(self, context: ExecutionContext, arguments: HandoffInput) -> ToolResult:
        self._record("record_handoff", context, arguments.model_dump(mode="json"))
        self._handoff_count += 1
        return ToolResult(
            tool_id="record_handoff",
            status=ToolResultStatus.SUCCEEDED,
            output=HandoffOutput(
                recorded=True,
                handoff_id=f"handoff-{self._handoff_count}",
                reason="Human review handoff recorded.",
            ),
            evidence=[EvidenceRef(evidence_id=f"handoff:{self._handoff_count}", source="handoff_log", kind="application_state")],
        )

    def _record(self, tool_id: str, context: ExecutionContext, arguments: dict[str, Any]) -> None:
        self.handler_calls.append(
            {
                "tool_id": tool_id,
                "execution_id": context.execution_id,
                "principal_tenant_id": context.principal.tenant_id,
                "arguments": arguments,
            }
        )


def access_policy(tools: ToolRegistry) -> PolicyDefinition:
    """Allow registered tools after the runtime checks identity and scope."""

    return PolicyDefinition(
        policy_id="software-access-tools",
        version="1.0.0",
        description="Allow the application tools in the local environment.",
        owner_id="production-ai-pillars",
        default_effect=PolicyEffect.DENY,
        lifecycle=AgentLifecycleStatus.ACTIVE,
        rules=[
            PolicyRule(
                rule_id="allow-registered-tools",
                effect=PolicyEffect.ALLOW,
                tool_ids=list(tools.names()),
                environments=["development"],
            )
        ],
    )


def tenant_boundary(*, principal: Any, arguments: dict[str, Any], **_: Any) -> bool:
    """Stop a tool before its handler when its tenant differs from identity."""

    requested_tenant = arguments.get("tenant_id")
    return requested_tenant is None or requested_tenant == principal.tenant_id
