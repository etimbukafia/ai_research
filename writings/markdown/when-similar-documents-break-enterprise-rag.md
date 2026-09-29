# When Similar Documents Break Enterprise RAG

In enterprise search, one question routinely matches dozens of plausible documents. The real engineering problem is rarely finding text that looks relevant; the problem is selecting the single document whose operational scope actually applies.

Consider this user request:

> What pressure limit applies to the Atlas X200 in Europe with firmware 4.2?

When you query an enterprise knowledge base with this prompt, five documents surface near the top of the candidate list:

| Document | Scope | Status | Pressure Limit | Role in the Case |
| :--- | :--- | :--- | ---: | :--- |
| `spec-17` | Atlas X200, Europe, firmware 4.2 | Approved and current | 220 kPa | Correct source |
| `spec-12` | Atlas X200, global, firmware 3.8 | Superseded | 250 kPa | Old-version distractor |
| `spec-19` | Atlas X200, Europe, firmware 4.3 | Draft | 230 kPa | Newer draft distractor |
| `bulletin-08` | Atlas X200, Europe, firmware 4.2 | Informational | 220 kPa | Supporting source without specification authority |
| `spec-27` | Atlas X210, Europe, firmware 4.2 | Approved and current | 240 kPa | Similar-product distractor |

Every single row contains the exact words the embedding model is looking for: Atlas, Europe, firmware, pressure, and limit. Yet only `spec-17` satisfies the joint constraints of product line, firmware revision, geographical region, effective timestamp, approval status, and organizational authority.

When I ran this test case through standard retrieval pipelines in our [dense enterprise RAG experiment](file:///C:/Users/j/afiavana/research/experiments/dense_enterprise_rag), the limitations of pure semantic search became immediate. 

Standard vector search returned `spec-17`, but it also surfaced superseded and draft versions with nearly identical cosine scores. BM25 selected `bulletin-08`, which contained the correct numeric value (220 kPa) but carried zero legal specification authority.

The complete benchmark evaluated 96 synthetic documents across 32 test cases structured into 16 matched pairs. The findings settled an important architectural question: metadata-filtered vector retrieval and graph-guided hybrid retrieval tied at 100.00% strict applicability across all 24 answer cases. 

Where the graph earned its keep was in evidence-path verification: metadata filtering produced 0.00% (0/24) path accuracy because it discards relationship provenance, whereas graph retrieval delivered a fully verified evidence chain in 100.00% (24/24) of cases.

The governing engineering rule is straightforward:
> Use semantic similarity to discover candidates. Apply explicit scope checks before answering. Add a knowledge graph when the answer must carry a verifiable, auditable chain of custody.

---

## Similarity Finds Candidates, Applicability Selects Evidence

In production knowledge bases, teams frequently maintain multiple active revisions of manuals, regional addenda, compliance policies, and service bulletins. Each variant repeats identical terminology, field labels, and schematic structures. 

This creates dense document overlap: a condition where dozens of documents share high semantic similarity while their real-world applicability diverges sharply.

The mathematical distinction between candidate discovery and evidentiary proof is fundamental:

```text
similarity(question, document) -> candidate pool
applicable(document, question) -> verified evidence
```

A candidate can score a 0.92 cosine similarity while failing basic version applicability. Conversely, a lower-ranked candidate can represent the only legally binding document for the user's specific context.

In the Atlas experiment, applicability is defined as a strict conjunction of operational constraints:

```python
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentScope(BaseModel):
    product_id: str
    firmware_version: str
    region: str
    effective_date: datetime
    expiration_date: datetime | None = None
    status: str = Field(description="Must be 'approved_current'")
    authoritative_body: str

    def is_applicable(self, target_product: str, target_version: str, target_region: str, query_time: datetime) -> bool:
        if self.product_id != target_product:
            return False
        if self.firmware_version != target_version:
            return False
        if self.region != target_region:
            return False
        if self.status != "approved_current":
            return False
        if query_time < self.effective_date:
            return False
        if self.expiration_date and query_time > self.expiration_date:
            return False
        return True
```

If any single term evaluates to false, the document is inapplicable, regardless of how high its vector score ranks in the candidate pool.

---

## Making the Evidence Chain Inspectable Through Graphs

Vector databases project documents into continuous geometric coordinates, discarding the relational structure connecting entities, documents, and authorities. 

When you index these relationships inside a property graph, the system returns a verified traversal path rather than an isolated chunk. In `case-01-dense`, graph-guided hybrid retrieval returned this explicit traversal chain:

```text
product-atlas-x200
  -> firmware-4-2
  -> region-europe
  -> document-spec-17
  -> fact-spec-17-pressure-limit
  -> authority-product-safety
```

The traversal verifies each hop: it confirms that `spec-17` applies to firmware 4.2 in Europe, defines the pressure limit as 220 kPa, and originates from Product Safety rather than marketing or general documentation.

```python
import networkx as nx

def verify_document_path(graph: nx.MultiDiGraph, product: str, version: str, region: str, doc_id: str) -> bool:
    # Verify the topological path from product entity to document authority
    try:
        has_firmware = graph.has_edge(product, version)
        valid_region = graph.has_edge(version, region)
        governing_doc = graph.has_edge(region, doc_id)
        is_approved = graph.nodes[doc_id].get("status") == "approved_current"
        
        return has_firmware and valid_region and governing_doc and is_approved
    except KeyError:
        return False
```

---

## How Strong is the Metadata Baseline?

Before reaching for a graph database, engineers should measure the strength of direct metadata filtering. 

If source documents have clean, structured metadata tags (product, version, region, effective date, and status), pre-filtering the vector index can solve applicability without graph traversal overhead.

The experiment compared four retrieval configurations over the exact same 96 documents:

| Condition | Primary Operation | Scope Enforcement |
| :--- | :--- | :--- |
| `bm25` | Weighted lexical term matching over title and body | None (lexical similarity only) |
| `vector` | Dense MiniLM embedding similarity | None (semantic similarity only) |
| `metadata_filtered_vector` | Filter direct scope attributes, then rank candidates by vector distance | Strict pre-filters on product, version, region, time, and status |
| `graph_guided_hybrid` | Retrieve top-20 vector candidates, then traverse and validate graph paths | Multi-hop validation of authority, status, and topological constraints |

Across the 24 answer cases and eight abstention cases, metadata-filtered vector search achieved 100.00% accuracy, matching the graph condition on document selection. 

Where metadata filtering falls short is on complex relational questions: verifying that Document A supersedes Document B, or validating that the department issuing a technical bulletin owns the regulatory authority for the underlying specification.

---

## The Reality of Automated Graph Extraction

A common assumption in GenAI marketing is that LLMs can automatically construct knowledge graphs from unstructured documentation with minimal human intervention.

To test this assumption, I built a schema-constrained extraction runner using `google:gemini-3.5-flash-lite` through PydanticAI. The extraction prompt enforced typed schemas, requiring models to extract entities, relationships, temporal validity bounds, and authority tags from 16 technical documents.

The audit completed 16 requests with zero network or provider errors. However, the extraction quality was remarkably low:

| Extraction Metric | Benchmark Result |
| :--- | ---: |
| **Node Precision** | 5.49% |
| **Node Recall** | 5.85% |
| **Edge Precision** | 2.08% |
| **Edge Recall** | 2.08% |
| **Scope Field Accuracy** | 4.17% |
| **Validity Field Accuracy** | 0.00% |
| **Authority Accuracy** | 0.00% |
| **Invalid Edge Rejection** | 100.00% |
| **Human Review Required** | 100.00% (16/16) |

The validator successfully rejected every hallucinated edge, forcing all 16 outputs into human review queues. 

This result highlights an unvarnished engineering reality: general-purpose language models cannot reliably build enterprise knowledge graphs without human-in-the-loop review. The deterministic retrieval benchmark succeeded because it evaluated a verified gold graph. Relying on uncurated, fully automated extraction in production guarantees corrupted edge topologies.

---

## Live Inference: Full Documents Outperform Evidence Packets

In the live evaluation run, the system executed 64 requests using `google:gemini-3.5-flash-lite` through PydanticAI at a sustained rate of 15 requests per minute, recording zero errors and zero retries.

The run tested two context delivery forms: feeding the model full document text versus feeding it compressed evidence packets stripped down to key graph assertions.

| Context Form | Exact Fact Accuracy | Answer Status Accuracy | Source Recall | Input Tokens | P95 Latency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Full Documents** | 100.00% (32/32) | 100.00% (32/32) | 100.00% (32/32) | 40,025 | 2,463 ms |
| **Evidence Packets** | 75.00% (24/32) | 75.00% (24/32) | 65.62% (21/32) | 31,062 | 2,049 ms |

I expected compressed evidence packets to win on latency while maintaining near-perfect accuracy. Instead, the live model stumbled: dropping from 100% down to 75% on exact facts because aggressive compression stripped subtle contextual qualifiers that the model needed to interpret the rule correctly.

While evidence packets reduced input token consumption by 22.39% and lowered P95 latency by 414 milliseconds, the accuracy cost was unacceptable for enterprise compliance.

---

## Review Enterprise RAG in This Order

When architecting retrieval systems over dense, overlapping document catalogs, execute your engineering evaluation in this sequence:

1. **Identify the Target Entity:** Exactly which product, SKU, or organizational unit does the query concern?
2. **Bind Environmental Dimensions:** Which firmware version, geographical region, and timestamp apply to the user's active session?
3. **Verify Source Authority:** Does the authoring department hold legal or operational jurisdiction over the requested specification?
4. **Evaluate Supersession Rules:** Has a newer bulletin or specification superseded the candidate document?
5. **Test Metadata Baseline First:** Can direct relational filters resolve the constraints before introducing graph dependencies?
6. **Validate the Full Path:** When regulatory audits or high-stakes agents require proof, does the system output an inspectable evidence chain?

Semantic similarity is a powerful tool for discovering candidates in an archive. In high-stakes enterprise systems, similarity is only the opening step. True retrieval accuracy requires enforcing explicit scope boundaries, and using graph structures when you need to prove why an answer is true.
