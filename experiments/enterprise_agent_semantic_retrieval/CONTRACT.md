# Enterprise Agent Lab contract

This file is the public interface between `enterprise_agent_lab` and its
article evaluator. The builder may change internal code. These names and
fields must remain stable.

## Package and commands

The package is importable from the repository root:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli demo
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli run --case case-01 --mode replay
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli run --case case-01 --mode live
```

`demo` and `run --mode replay` do not need a network or `GOOGLE_API_KEY`.
`run --mode live` uses Gemini and fails with a clear configuration error when
`GOOGLE_API_KEY` is absent.

The live model name is `google:gemini-3.7-flash` by default. `GEMINI_MODEL`
may override the model suffix, but it must not be a `latest` alias. The API
key variable is `GOOGLE_API_KEY`.

## Public models

All models are Pydantic v2 models exported from
`enterprise_agent_lab.models`.

### Source models

Each source model uses the fields below. Dates are ISO `YYYY-MM-DD` strings.
Nullable end dates use `null`.

```text
Account
  account_id: str
  name: str
  status: str
  region: str
  owner: str
  valid_from: str
  valid_to: str | None
  source_id: str

Contract
  contract_id: str
  account_id: str
  plan: str
  seat_limit: int
  seat_billing_rule: str
  effective_from: str
  effective_to: str | None
  renewal_date: str
  service_credit_allowed: bool
  source_id: str
  authority: str

Subscription
  subscription_id: str
  account_id: str
  contract_id: str
  plan: str
  status: str
  seats: int
  started_at: str
  ended_at: str | None
  source_id: str

Invoice
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

UsageRecord
  usage_id: str
  account_id: str
  subscription_id: str
  period_start: str
  period_end: str
  peak_seats: int
  active_users: int
  source_id: str

SupportTicket
  ticket_id: str
  account_id: str
  subject: str
  priority: str
  status: str
  created_at: str
  source_id: str

PolicyRule
  policy_id: str
  name: str
  action: str
  condition: str
  threshold: float | None
  approval_required: bool
  authority: str
  valid_from: str
  valid_to: str | None
  source_id: str
```

### Semantic and result models

```text
CatalogEntry
  catalog_id: str
  concept: str
  concept_type: str
  definition: str
  synonyms: list[str]
  entity: str
  grain: str
  maps_to: list[str]
  allowed_joins: list[str]
  time_basis: str
  authority: str
  valid_from: str
  valid_to: str | None
  source_ids: list[str]
  preconditions: list[str]
  do_not_use: list[str]

EvidenceRef
  evidence_id: str
  source_type: str
  source_id: str
  excerpt: str
  authority: str
  valid_from: str | None
  valid_to: str | None

PolicyCheck
  policy_id: str
  name: str
  passed: bool
  reason: str
  approval_required: bool
  source_id: str | None

ToolCallRecord
  call_id: str
  tool_name: str
  arguments: dict[str, object]
  result_summary: str
  source_ids: list[str]
  side_effect_level: str
  approved: bool | None

DraftAction
  action_id: str
  action_type: str
  account_id: str
  payload: dict[str, object]
  status: str
  requires_approval: bool
  policy_check_ids: list[str]
  evidence_ids: list[str]

AgentDecision
  status: Literal[answer, needs_clarification, needs_human_review, insufficient_evidence]
  request: str
  resolved_concepts: list[str]
  selected_tools: list[str]
  typed_tool_arguments: dict[str, dict[str, object]]
  claims: list[str]
  evidence_ids: list[str]
  policy_checks: list[PolicyCheck]
  draft_action: DraftAction | None
  explanation: str

RunTrace
  run_id: str
  case_id: str | None
  condition: str
  mode: str
  model_name: str
  prompt_hash: str
  retrieved_context_ids: list[str]
  tool_calls: list[ToolCallRecord]
  policy_checks: list[PolicyCheck]
  final_decision: AgentDecision | None
  started_at: str
  finished_at: str | None
  latency_ms: float | None
  usage: dict[str, object]
  error: str | None
```

The status values are closed. A policy failure must not become `answer`.
Draft actions are local records. They never change source records.

## Tool names and registry interface

The public registry is `ToolRegistry` from `enterprise_agent_lab.tools`.
The registry exposes:

```python
registry.call(name: str, arguments: BaseModel | dict) -> BaseModel
registry.specs() -> list[ToolSpec]
registry.names() -> list[str]
```

The stable tool names are:

Read tools:

```text
find_account
get_active_contract
get_subscription
get_invoice
get_usage_record
get_support_tickets
search_business_concepts
check_policy
```

Draft-only action tools:

```text
draft_service_credit
draft_plan_change
draft_support_ticket
request_human_approval
```

Every tool has a typed Pydantic input model and output model. The registry
rejects an unknown tool, invalid arguments, an account outside the configured
scope, and an action that fails the policy gate.

## Retrieval interface

The common interface is `SemanticRetriever` from
`enterprise_agent_lab.retrieval`:

```python
search_business_context(
    query: str,
    account_scope: str | None = None,
    concept_type: str | None = None,
    top_k: int = 8,
) -> list[RetrievedContext]
```

`RetrievedContext` has:

```text
context_id: str
condition: str
kind: str
text: str
score: float
source_ids: list[str]
catalog_entry: CatalogEntry | None
metadata: dict[str, object]
```

The implementation provides BM25 and vector scores through the same method.
`HybridRetriever` merges them with deterministic tie ordering. The supported
conditions are `raw_schema`, `prose_rag`, and `semantic_catalog`.

## Replay and live result formats

`run_case(case_id, mode="replay", condition="semantic_catalog")` returns a
`RunTrace` and writes the same JSON object to the selected run directory.
Replay uses checked-in deterministic outputs and needs no API key.

The CLI prints a JSON object with these top-level keys:

```text
run_id
case_id
condition
mode
model_name
status
claims
evidence_ids
tool_calls
policy_checks
draft_action
trace_path
```

Live mode uses PydanticAI's Google provider with the model name
`google:<GEMINI_MODEL>`, validates `AgentDecision`, and records model and
usage metadata in `RunTrace`. Missing `GOOGLE_API_KEY` raises a clear
`ConfigurationError` before any tool or write action runs.
