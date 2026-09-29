# Enterprise Agents Need a Semantic Map of the Business

An autonomous enterprise agent can invoke tools with syntactic perfection, parse structured JSON schemas without error, and cleanly execute complex state machines, yet still deliver a catastrophic business decision.

Consider a routine billing inquiry submitted to our fictional SaaS provider, Aster Cloud:

> The Acme Robotics account used 150 seats in May, but the invoice shows 100. Can we issue a service credit and offer a plan change?

To an ungrounded model, this prompt seems like an invitation to fetch usage figures, grab an invoice, calculate a difference of 50 seats, and issue a credit draft. But beneath those plain words lie tangled enterprise definitions:
- What constitutes a "used seat" under Acme's signed contract? Is it peak concurrent concurrency, active monthly provisioning, or assigned directory entries?
- Which contract terms governed Aster's billing relationship during the month of May?
- Does Aster Cloud's credit policy permit automated credits above $500 without managerial sign-off?
- Does a pending plan change require renegotiating minimum commitment tiers?

Each tool invocation in an ungrounded trace may return HTTP 200 responses with valid payloads. The agent can still commit a critical business blunder because tool contracts define interface mechanics rather than organizational semantics.

In our [semantic retrieval experiment](file:///C:/Users/j/afiavana/research/experiments/enterprise_agent_lab/experiments/semantic_retrieval), we set out to solve this disconnect by establishing a machine-readable layer between model reasoning and tool execution:
> An enterprise agent requires an explicit, machine-readable map that binds business terminology to underlying database records, operational policies, temporal validity rules, and evidentiary provenance.

---

## When Valid Tool Calls Produce Broken Decisions

When software teams expose tools to large language models, they typically rely on standard OpenAPI descriptions and JSON schema schemas. These schemas tell the agent which fields to populate, but they cannot tell the agent when the enterprise permits those fields to be acted upon.

Consider what happens in a naive tool execution trace:

```text
Request
  ├── get_usage_record(account_id="acme", month="2026-05")
  ├── get_invoice(account_id="acme", month="2026-05")
  └── draft_service_credit(account_id="acme", amount=1250.00)
```

The model populated every required parameter, passed input validation, and created a draft credit. Yet the decision failed on four operational counts:
1. It equated an unbilled telemetry peak with contractual billable usage, ignoring the clause in Acme's contract that caps billable overages.
2. It evaluated May usage against an April contract amendment that had already been superseded.
3. It computed a credit value based on gross list pricing rather than Acme's negotiated net rate.
4. It issued an automated credit draft for $1,250 when corporate governance requires VP approval for any adjustment exceeding $500.

To avoid these traps, an agent must construct a semantic plan before touching transactional tools:

```text
Request
  ├── 1. Resolve business concepts against corporate catalog
  ├── 2. Retrieve authoritative definitions, validity windows, and constraints
  ├── 3. Bind concepts to concrete system-of-record schemas
  ├── 4. Select typed tools within explicit policy boundaries
  ├── 5. Collect evidentiary proof across systems
  ├── 6. Verify temporal currency and sign-off thresholds
  └── 7. Generate structured action or request human review
```

---

## Modeling Business Meaning as an Explicit Data Dependency

Raw database fields rarely convey business intent. A database column named `status` might describe an account standing, a payment state, or a server deployment stage. 

In Aster Cloud's semantic catalog, we treat operational concepts as structured, queryable entities:

```python
from datetime import datetime
from pydantic import BaseModel, Field

class BusinessConcept(BaseModel):
    catalog_id: str
    concept: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    entity: str
    grain: str
    maps_to: list[str]
    time_basis: str
    authority_source: str
    allowed_joins: list[str]
    forbidden_sources: list[str] = Field(default_factory=list)
    effective_from: datetime
    effective_to: datetime | None = None
```

For the concept `billable_seats`, the catalog entry specifies operational boundaries:

```json
{
  "catalog_id": "concept-billable-seats",
  "concept": "billable_seats",
  "definition": "The highest seat count recorded during the invoice period when the active contract permits usage billing.",
  "synonyms": ["used seats", "licensed seats", "seat usage"],
  "entity": "subscription",
  "grain": "account_month",
  "maps_to": [
    "usage.monthly_seat_peak",
    "contracts.seat_billing_rule"
  ],
  "time_basis": "invoice_period",
  "authority_source": "contracts_v2",
  "allowed_joins": [
    "subscription.account_id = usage.account_id"
  ],
  "forbidden_sources": ["support.ticket_count"],
  "effective_from": "2026-01-01T00:00:00"
}
```

This catalog provides an agent with critical constraints that vector similarity search over raw documentation cannot infer. It dictates the authoritative source system, specifies allowable table joins, and explicitly warns the model against using informal support ticket tallies to determine customer seat volume.

Our semantic architecture catalogs eight distinct record types:

| Record Type | Operational Role |
| :--- | :--- |
| `concept` | Canonical business terms, colloquial synonyms, and scope boundaries. |
| `entity` | Core business nouns (e.g. Account, Subscription, Invoice) anchoring data records. |
| `relationship` | Validated joins and associations connecting disparate business entities. |
| `metric` | Mathematical formulas, aggregation grains, and temporal dimensions. |
| `policy` | Decision thresholds, approval requirements, and operational invariants. |
| `tool_contract` | Purpose definitions, side-effect ratings, and execution prerequisites. |
| `source` | Ownership records, update cadences, and authoritative provenance ratings. |
| `negative_rule` | Explicit constraints detailing forbidden data paths and unsupported inferences. |

---

## Controlled Action Boundaries and Typed Agent Decisions

To prevent rogue execution, tools must be categorized by their potential side effects. In our lab environment, read tools retrieve context while mutation tools produce non-destructive drafts that enforce policy preconditions:

```python
from typing import Literal
from pydantic import BaseModel

class DraftAction(BaseModel):
    action_type: str
    target_account: str
    payload: dict
    required_signoff_role: str | None = None
    policy_reference: str

class AgentDecision(BaseModel):
    status: Literal[
        "answer",
        "needs_clarification",
        "needs_human_review",
        "insufficient_evidence"
    ]
    resolved_concepts: list[str]
    selected_tools: list[str]
    evidence_ids: list[str]
    policy_checks: list[str]
    claims: list[str]
    draft_action: DraftAction | None = None
```

By enforcing a closed set of decision statuses, the framework prevents an ungrounded model from masking missing data with fluent prose. If the agent cannot substantiate a claim with authoritative evidence IDs, or if a proposed service credit exceeds pre-programmed limits, the runtime rejects the action and forces a status of `needs_human_review` or `insufficient_evidence`.

---

## Experimental Setup: Testing Three Context Retrieval Architectures

To evaluate whether a semantic catalog materially alters agent reliability, we built the `enterprise_agent_lab`. We modeled Aster Cloud with synthetic CRM, contract, billing, usage, and support databases stored locally in SQLite, using SQLite FTS5 for keyword retrieval and local MiniLM vectors (`sentence-transformers/all-MiniLM-L6-v2`) for conceptual similarity.

We tested three distinct context architectures while keeping agent instructions, model interfaces, and database records identical:

1. `raw_schema`: The model receives standard database table structures, column definitions, and basic tool documentation.
2. `prose_rag`: The model receives retrieved passages from unstructured enterprise documentation and internal wiki pages.
3. `semantic_catalog`: The model queries our typed semantic layer, retrieving structured definitions, join constraints, temporal validity rules, and policy bounds.

We evaluated all three conditions across 24 deterministic enterprise cases (72 total evaluation traces) categorized into concept resolution, cross-system joins, policy-governed actions, and temporal/missing authority challenges. The deterministic evaluation suite ran in replay mode (Run ID: `replay-142b33589d57073d`) targeting the `google:gemini-3.5-flash-lite` contract inside our [semantic retrieval experiment](file:///C:/Users/j/afiavana/research/experiments/enterprise_agent_lab/experiments/semantic_retrieval).

---

## Benchmark Results: The Semantic Catalog Divide

Evaluating agent performance requires measuring intermediate decision quality alongside final outcomes:

| Condition | Concept Acc | Tool Acc | Argument Acc | Evidence Recall | Policy Acc | Temporal Acc | Final Decision | Correct Abstention | Unsafe Action Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.4306 | 0.2500 | 1.0000 | 0.0417 | 0.0000 | 0.0000 | 0.0000 | 0.5417 | 0.2917 |
| `prose_rag` | 0.4514 | 0.2500 | 1.0000 | 0.0417 | 0.0000 | 0.0000 | 0.0000 | 0.5417 | 0.2917 |
| `semantic_catalog` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

Both `raw_schema` and `prose_rag` achieved perfect argument syntax (1.0000) whenever they called tools. However, their grasp of business context was severely deficient: both scored under 0.46 on concept accuracy, achieved a dismal 0.0417 on evidence recall, and failed completely on policy and temporal verification (0.0000). Crucially, nearly 30% of their traces resulted in unsafe actions that bypassed mandatory financial approval controls.

The `semantic_catalog` achieved 1.0000 across all evaluation axes, reducing the unsafe action rate to 0.0000.

Breaking down the results by case classification illustrates where traditional retrieval fails:

| Case Category | Retrieval Condition | Concept Acc | Tool Acc | Evidence Recall | Policy Acc | Final Decision | Correct Abstention |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `concept_resolution` | `raw_schema` | 0.8333 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `concept_resolution` | `prose_rag` | 0.8333 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `concept_resolution` | `semantic_catalog` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `cross_system` | `raw_schema` | 0.2222 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| `cross_system` | `prose_rag` | 0.2222 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| `cross_system` | `semantic_catalog` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `policy_and_action` | `raw_schema` | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 |
| `policy_and_action` | `prose_rag` | 0.5833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1667 |
| `policy_and_action` | `semantic_catalog` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `temporal_authority` | `raw_schema` | 0.1667 | 0.1667 | 0.1667 | 0.0000 | 0.0000 | 0.3333 |
| `temporal_authority` | `prose_rag` | 0.1667 | 0.1667 | 0.1667 | 0.0000 | 0.0000 | 0.3333 |
| `temporal_authority` | `semantic_catalog` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

In the featured Acme Robotics inquiry (`case-01`), the expected operational outcome is `needs_human_review`:

| Architecture | Emitted Status | Concept Acc | Tool Acc | Evidence Recall | Policy Acc | Final Score |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | `answer` | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `prose_rag` | `answer` | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `semantic_catalog` | `needs_human_review` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Both the raw schema and prose RAG conditions attempted to directly answer the customer, miscalculating seat definitions and triggering policy gate violations. The semantic catalog resolved the contractual seat definition, identified the governing May addendum, flagged that the requested credit crossed the autonomous signing limit, and correctly routed the request to human operations.

---

## Engineering Checklists for Grounded Enterprise Agents

Before giving an agent write permissions or connecting it to customer workflows, evaluate its reasoning chain against this verification sequence:

1. **Concept Disambiguation**: Does the agent resolve user terminology to canonical business concepts before planning tool calls?
2. **Authority Hierarchy**: Does the architecture define which system of record takes legal precedence when CRM records conflict with ERP invoices?
3. **Temporal Binding**: Are all retrieved records validated against the timestamp of the event rather than current runtime clock?
4. **Negative Retrival Rules**: Does the catalog explicitly bar tools from joining incompatible tables or reading deprecated fields?
5. **Deterministic Policy Gates**: Is policy enforcement isolated in deterministic code outside the probabilistic language model?
6. **Explicit Abstention Pathways**: Can the agent cleanly return `insufficient_evidence` or `needs_human_review` without defaulting to speculative filler?

Prompts can be made longer and models can be made larger, but neither will bridge the gap between interface schemas and business truth. Grounding enterprise agents requires modeling business semantics as first-class, machine-readable data.

All code, schemas, and benchmark logs can be explored directly in the [semantic retrieval experiment](file:///C:/Users/j/afiavana/research/experiments/enterprise_agent_lab/experiments/semantic_retrieval).
