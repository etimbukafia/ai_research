# When Customer Assistants Need Graph Memory

Customer support tickets arrive as deceptively simple sentences. A customer writes in with a straightforward inquiry, yet fulfilling that request requires an assistant to traverse half a dozen disconnected backend systems.

Consider a familiar support message:

> The battery for the camera I bought last year has failed. Is it covered by my warranty? Please use my current delivery address.

To formulate an answer that an operations team can safely trust, an automated assistant must perform five distinct tasks in sequence. It must identify the customer, locate the exact camera order from twelve months ago, inspect the warranty schedule tied to that specific SKU, confirm which replacement pack fits that model, and verify the customer's current shipping address. 

The required answer is an explicit path through enterprise records. When you store those facts in graph memory, that traversal becomes a first-class operation. When you attach validity timestamps and source IDs to every edge, the path becomes safe for production use.

The question we set out to answer in our [customer assistant memory experiment](file:///C:/Users/j/afiavana/research/experiments/customer_assistant_memory) is straightforward: under what conditions does graph memory justify its operational overhead?

---

## The Answer Path Crosses Multiple Records

Every production support case has an underlying topology. In our benchmark, the camera support ticket maps to a directed graph:

```text
Customer
  ├── PLACED ──> Order ── CONTAINS ──> Camera
  │                                      ├── COVERED_BY ──> Warranty
  │                                      └── COMPATIBLE_WITH ──> Battery
  ├── OPENED ──> Support ticket ── REPORTS ──> Battery issue
  └── HAS_ADDRESS ──> Current delivery address
```

Each edge resolves a necessary condition of the request:
- The purchase order identifies the specific hardware variant the customer actually owns.
- The hardware node points directly to the active warranty contract and compatible part catalog.
- The support ticket links the reported symptom to the hardware fault.
- The address record designates where replacement hardware may lawfully be shipped.

Critically, the assistant must constrain this entire traversal to a single customer scope while checking temporal validity at every step. A relocated customer frequently has an expired address on file alongside a current one. A warranty expires on a specific calendar date. A manufacturer issues updated replacement guidelines that override older support articles.

Our guiding architectural rule reflects this reality:
> Graph memory proves its worth when an answer requires connected facts across distinct records, time horizons, and source systems.

---

## A Flat List of Notes Discards Structural Joins

When teams implement memory using standard flat document stores, they typically convert customer interactions into isolated text chunks:

```text
Customer note: Customer owns a Lumix camera.
Order note: Customer purchased camera body on 2025-04-12.
Ticket note: Camera battery fails to hold charge.
Warranty note: Standard battery packs carry 24-month limited coverage.
Address note: Customer updated shipping address in Austin.
```

Each note contains relevant keywords, but the relational glue between them has evaporated.

If you query this store with a recent-window heuristic, the system retrieves the latest support ticket and the updated address, yet it drops the older purchase invoice and warranty documentation. If you rely purely on semantic vector search, the embedding model frequently retrieves a high-scoring battery ticket belonging to an entirely different customer who described identical symptoms. The language model must then guess whether every retrieved snippet belongs to the same hardware asset.

Graph memory internalizes these relational joins into the retrieval pass itself. Execution begins at the authenticated customer identity, traverses typed edges to the target order, and collects only the records bound to that validated lineage:

```python
from datetime import datetime
from pydantic import BaseModel, Field

class MemoryEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    observed_at: datetime
    valid_from: datetime
    valid_to: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = "active"

    def is_currently_valid(self, as_of: datetime) -> bool:
        return self.valid_from <= as_of <= self.valid_to and self.status == "active"
```

A graph containing unverified links or stale timestamps will still mislead an agent. The database structure alone does not guarantee correctness; temporal boundaries and authority scores supply the necessary safeguards.

---

## Grounding Memory in Typed Entities and Valid Time

Our implementation models customer interactions across nine distinct node types:

| Node | Functional Responsibility |
| :--- | :--- |
| `Customer` | The verified account entity that anchors record lineage. |
| `Ticket` | The active support interaction and its state machine. |
| `Order` | The financial purchase record and fulfillment timestamp. |
| `Product` | The hardware SKU, assembly variant, or replacement part. |
| `Warranty` | Legal coverage clauses, duration limits, and exceptions. |
| `Issue` | The categorized mechanical or electrical failure. |
| `Address` | Physical delivery coordinates bound to effective date ranges. |
| `Preference` | Communication channels and opt-in settings. |
| `Resolution` | Authoritative support decisions and approved remediation policies. |

The relationships define the precise operational dependencies between nodes:

```text
Customer -[PLACED]-> Order
Order -[CONTAINS]-> Product
Product -[COVERED_BY]-> Warranty
Ticket -[REPORTS]-> Issue
Product -[COMPATIBLE_WITH]-> Replacement
Customer -[HAS_ADDRESS]-> Address
```

Every edge in our benchmark carries six metadata attributes:
- `observed_at`: The exact timestamp when our ingestion pipeline recorded the fact.
- `valid_from` and `valid_to`: The interval during which the relationship legally governs support decisions.
- `source_id`: The system of record providing evidentiary provenance (e.g. `billing-postgres`, `zendesk-core`).
- `confidence`: Extraction certainty score for inferred relations.
- `status`: Lifecycle marker (`active`, `closed`, `superseded`, or `disputed`).

These attributes protect the agent from common production traps. When a customer moves, the previous address edge is marked `superseded` with a terminated `valid_to` date. When an agent verifies warranty eligibility, it evaluates coverage as of the ticket creation date rather than today's wall-clock time.

---

## Evaluating Four Memory Architectures Under Equal Conditions

To measure the real impact of memory layout, we constructed a controlled test harness. The agent policy, skill toolset, prompts, and output schema remained strictly identical across runs; only the underlying retrieval engine changed:

| Architecture | Retrieval Strategy | Primary Failure Mode |
| :--- | :--- | :--- |
| `flat_recent` | Reads a fixed chronological window of recent interactions. | Older foundational purchase and warranty records fall outside the window. |
| `vector_chroma` | Cosine similarity retrieval over customer-scoped chunks. | Surface semantic matches lack relational joins or pull noisy neighbor data. |
| `graph_ladybug` | Breadth-first graph traversal anchored to customer ID with time filters. | Broken entity links or incorrect timestamps yield empty or diverted paths. |
| `hybrid_graph_vector` | Vector search seeds candidates; graph traversal validates structural path. | Multi-stage pipeline adds latency and operational complexity. |

The evaluation harness ran across 24 enterprise support cases divided into four equal categories: direct lookups, multi-step relational traversals, temporal state transitions, and missing-evidence conflict cases. 

We used embedded LadybugDB for the graph store, Chroma with `sentence-transformers/all-MiniLM-L6-v2` for dense embeddings, and executed the entire suite deterministically on CPU inside the [`customer_assistant_memory` harness](file:///C:/Users/j/afiavana/research/experiments/customer_assistant_memory).

---

## Benchmark Results: The Multi-Step Relational Divide

When looking across all 24 cases, top-line accuracy figures can easily hide where systems fail:

| Method | Evidence Recall | Path Recall | Answer Accuracy | Provenance Precision |
| :--- | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.6201 | 0.0000 | 0.4583 | 0.5000 |
| `vector_chroma` | 0.9625 | 0.0000 | 0.8750 | 0.9708 |
| `graph_ladybug` | 0.9545 | 0.9545 | 0.9583 | 0.9292 |
| `hybrid_graph_vector` | 0.9545 | 0.9545 | 0.9583 | 0.9292 |

The critical divergence emerges when we isolate the multi-step relationship category:

| Method | Evidence Recall | Path Recall | Answer Accuracy |
| :--- | ---: | ---: | ---: |
| `flat_recent` | 0.4958 | 0.0000 | 0.0000 |
| `vector_chroma` | 0.8625 | 0.0000 | 0.5000 |
| `graph_ladybug` | 1.0000 | 1.0000 | 1.0000 |
| `hybrid_graph_vector` | 1.0000 | 1.0000 | 1.0000 |

Dense embeddings retrieved individual documents effectively, achieving 0.8625 evidence recall on multi-step queries, but failed to assemble the relational path between them. Because vector search treats each chunk as an island in embedding space, it scored 0.0000 on path recall, leading to an answer accuracy of only 0.5000. 

Both `graph_ladybug` and `hybrid_graph_vector` achieved 1.0000 across evidence recall, path recall, and final answer accuracy on these multi-step queries. They followed the verified link from order to SKU to warranty contract without relying on language model guesswork.

In our featured camera battery case, this difference determined whether the case required human intervention:

| Architecture | System Decision | Evidence Recall | Path Recall |
| :--- | :--- | ---: | ---: |
| `flat_recent` | `needs_human_review` | 0.6250 | 0.0000 |
| `vector_chroma` | `needs_human_review` | 0.6250 | 0.0000 |
| `graph_ladybug` | `answer` | 1.0000 | 1.0000 |
| `hybrid_graph_vector` | `answer` | 1.0000 | 1.0000 |

Both flat and vector retrievers missed the conjunction of active warranty policy and current shipping coordinates, triggering our safety policy and routing the ticket to an agent queue. The graph engines assembled the entire chain (`order-01` -> `product-01` -> `warranty-01` -> `replacement-01` -> `address-current-01`), validating both currency and authority before generating an automated response.

---

## Inspecting Execution Traces

Rather than reviewing unstructured model dialogue, production monitoring requires inspecting execution traces:

```text
Ticket Input (Customer: cust-01)
  ├── 1. find_customer(cust-01) ──> OK
  ├── 2. traverse_graph(cust-01 -[PLACED]-> Order) ──> order-01
  ├── 3. traverse_graph(order-01 -[CONTAINS]-> Product) ──> product-01 (Camera)
  ├── 4. traverse_graph(product-01 -[COVERED_BY]-> Warranty) ──> warranty-01 (Valid through 2026-12-31)
  ├── 5. traverse_graph(product-01 -[COMPATIBLE_WITH]-> Part) ──> replacement-01 (Battery Pack)
  ├── 6. filter_edges(cust-01 -[HAS_ADDRESS]-> Address, status="active") ──> address-current-01
  └── 7. synthesize_response() ──> Approved for automated resolution
```

When an engine fails, the trace exposes the breakdown immediately:
- The flat window drops `warranty-01` because the original invoice sits outside the rolling 30-day window.
- The vector retriever pulls an address update from `cust-04` due to overlapping street terminology.
- An unindexed graph returns an expired billing address because the query omitted the `valid_to` predicate.

Making the traversal explicit allows engineering teams to debug retrieval failures deterministically, isolating whether an issue stemmed from missing raw data, faulty entity resolution, or improper temporal filtering.

---

## Architectural Trade-offs and Decision Framework

Graph databases introduce operational overhead that pure vector stores avoid. Before committing to a graph architecture, engineering teams should evaluate their retrieval requirements along four dimensions:

1. **Relational Depth**: Does answering a request require joining three or more distinct business entities? If queries consistently touch a single document or simple key-value pairs, a vector index or standard document store is more maintainable.
2. **Temporal Volatility**: Do customer attributes change over time while older interactions remain on record? If shipping addresses, warranty terms, or account tiers evolve, explicit validity intervals on graph edges prevent stale data contamination.
3. **Auditability and Compliance**: Must the system prove exactly why an automated warranty replacement or refund was authorized? Graph memory preserves an immutable chain of custody from original purchase to current ticket.
4. **Data Hygiene Dependency**: A graph is only as reliable as its entity extraction and edge maintenance. If your pipeline creates incorrect edges between similar products, the assistant will traverse that erroneous path with complete confidence.

When your application requires joining disparate business records across shifting temporal boundaries, graph memory provides the structural foundation that vector similarity alone cannot deliver.

All evaluation scripts, datasets, and benchmark traces are available in the [customer assistant memory experiment](file:///C:/Users/j/afiavana/research/experiments/customer_assistant_memory).
