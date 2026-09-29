# Information Retrieval from the Bottom Up

Most tutorials on modern search begin by embedding text into high-dimensional vector spaces. By jumping immediately into neural dense retrieval, developers frequently bypass the foundational question that governs all search architectures: what specific conditions make a retrieved document relevant to an information need?

Relevance does not originate in an embedding model. It emerges from the relationship between the user's intent, the vocabulary distribution of the corpus, the indexing data structure, and the scoring function that evaluates candidate matches.

To understand where retrieval pipelines break down, we constructed an empirical benchmark comparing three classical and modern approaches: exact boolean term matching, Okapi BM25 weighted lexical scoring, and dense vector similarity search. By evaluating all three strategies against the exact same corpus under controlled conditions, we can isolate how representation choices dictate retrieval failures.

The code, document fixtures, and deterministic runner for this study are published in our [information retrieval experiment](file:///C:/Users/j/afiavana/research/experiments/information_retrieval).

The central lesson from this bottom-up investigation is straightforward:
> A search system can rank only the information that its corpus vocabulary, index structures, query syntax, and relevance rubrics make visible.

---

## Tracing an Information Need

Consider a software engineer querying an internal engineering wiki:

> How can a search system find relevant text when the user's words differ from the words in the documents?

This query articulates the classic vocabulary mismatch problem. The user is seeking conceptual explanations of synonym expansion, semantic embeddings, and paraphrase handling, yet they phrased the inquiry using generic colloquial terms: `search`, `words`, `differ`, and `relevant`.

A robust benchmark must expose how different indexing representations react to this phrasing:
- Does boolean matching find anything when terms overlap only on generic stop-words?
- Does BM25's inverse document frequency discount the common terms while surfacing conceptual discussions?
- Does dense cosine distance capture the underlying semantic need, or does it become distracted by conversational sentence structure?

---

## Constructing the Controlled Corpus

To eliminate noise from massive uncurated datasets, we designed a clean twelve-document technical corpus. Each record represents a discrete architectural concept within information retrieval:

```python
from pydantic import BaseModel, Field

class SearchDocument(BaseModel):
    id: str
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)

    def full_content(self) -> str:
        return f"{self.title}. {self.text}"
```

Our evaluation suite indexes twelve core documents:

| Document ID | Canonical Topic | Experimental Purpose |
| :--- | :--- | :--- |
| `doc-01` | Exact keyword matching | Verifies unweighted lexical overlap. |
| `doc-02` | Okapi BM25 | Evaluates frequency and document-length normalization. |
| `doc-03` | Inverted indexing | Tests postings lists and term dictionary retrieval. |
| `doc-04` | Synonyms and paraphrases | Ground-truth target for vocabulary mismatch handling. |
| `doc-05` | Dense vector embeddings | Ground-truth target for continuous representation. |
| `doc-06` | Cosine vector similarity | Ground-truth target for geometric distance scoring. |
| `doc-07` | Text chunking strategies | Tests document boundary effects. |
| `doc-08` | Metadata filtering | Evaluates deterministic scope constraints. |
| `doc-09` | Cross-encoder reranking | Tests secondary precision scoring stages. |
| `doc-10` | Human relevance judgments | Ground truth definition for evaluation metrics. |
| `doc-11` | Retrieval metrics (Recall@5, MRR) | Defines mathematical evaluation formulas. |
| `doc-12` | Stale or missing records | Negative distractor representing index decay. |

Consider `doc-04` in detail:

```json
{
  "id": "doc-04",
  "title": "Synonyms and paraphrases",
  "text": "Paraphrase handling links automobile with car. It connects alternative expressions that describe one information need. A relevance decision checks whether two passages answer the same question.",
  "tags": ["semantics", "paraphrase", "relevance"]
}
```

This passage directly answers the query intent, yet it intentionally avoids repeating the query's phrasing: it uses `paraphrase handling`, `alternative expressions`, and `passages` rather than `search system`, `words differ`, or `documents`. For a lexical retriever, this creates a severe vocabulary gap.

---

## Comparing Three Indexing Paradigms

We implemented three distinct retrieval backends over the exact same twelve-document dataset:

### 1. Exact Keyword Matching
An inverted index maps normalized vocabulary tokens to document posting lists. The scoring algorithm tallies unique query terms appearing in each document title and body, breaking score ties by ascending document identifier.

While straightforward, this approach is rigid: a query that shares three incidental words with an irrelevant passage will always outrank a highly relevant document that shares only one word.

### 2. Okapi BM25 Scoring
BM25 refines raw lexical counts by introducing non-linear term saturation and length penalization:

```text
Score(D, Q) = Sum[ IDF(q_i) * (TF(q_i, D) * (k1 + 1)) / (TF(q_i, D) + k1 * (1 - b + b * (|D| / avgDL))) ]
```

We parameterized the engine with standard production values: `k1 = 1.5` for term saturation and `b = 0.75` for document length normalization. BM25 appropriately reduces the influence of frequent words like `system` and `find`, rewarding rare terms. However, it remains fundamentally constrained by lexical identity: if a concept is discussed using distinct vocabulary, BM25 assigns it a score of zero.

### 3. Dense Vector Similarity
Using `sentence-transformers/all-MiniLM-L6-v2`, we mapped each document title and body into a normalized 384-dimensional dense vector, storing embeddings in an in-memory Chroma collection. At query time, we compute the cosine similarity between the embedded inquiry and the document representations.

Vectors promise semantic understanding, but they introduce geometric distortions: a document describing general search concepts can sit closer in embedding space to a conversational query than a concise technical passage discussing paraphrases.

---

## Benchmark Results: The Vocabulary Failure

We evaluated all three methods across our primary question using two standard information retrieval metrics:
- **Recall@5**: The fraction of authoritative documents (`doc-04`, `doc-05`, `doc-06`) surfaced within the top five positions.
- **Mean Reciprocal Rank (MRR)**: The multiplicative inverse of the rank of the first relevant result.

| Retrieval Method | Surfaced Top-5 Document IDs | Recall@5 | MRR |
| :--- | :--- | ---: | ---: |
| `exact_keyword` | `doc-03`, `doc-01`, `doc-07`, `doc-09`, `doc-11` | 0.0000 | 0.0000 |
| `bm25` | `doc-03`, `doc-07`, `doc-11`, `doc-01`, `doc-09` | 0.0000 | 0.0000 |
| `vector_chroma` | `doc-01`, `doc-03`, `doc-12`, `doc-02`, `doc-07` | 0.0000 | 0.0000 |

The primary result provides an important engineering reality check: every single method failed completely, yielding 0.0000 Recall@5 and 0.0000 MRR.

The failure mechanics are revealing:
- Exact keyword search and BM25 latched onto the shared words `search`, `documents`, and `system`, heavily boosting `doc-03` (Inverted indexes) and `doc-07` (Chunking), while completely omitting `doc-04`, `doc-05`, and `doc-06`.
- Vector search similarly stumbled. The dense model was overwhelmed by the conversational syntax of the query (`How can a search system find...`), causing it to place `doc-01` (Exact keyword search) and `doc-03` at the top of the rank list, pushing the true explanatory documents outside the top five cutoff.

---

## Validation Queries: Exposing the Mechanism

To confirm whether this failure stemmed from fundamental algorithm mechanics or query phrasing, we ran two validation queries through the identical harness:

### Validation Query 1: Direct Terminology
> Which data structure maps terms to document IDs for fast exact keyword search?
*(Gold Target: `doc-03`)*

### Validation Query 2: Focused Paraphrase
> How can a machine match a question to a passage that uses different language?
*(Gold Targets: `doc-04`, `doc-05`, `doc-06`)*

| Evaluation Metric | Query Type | `exact_keyword` | `bm25` | `vector_chroma` |
| :--- | :--- | ---: | ---: | ---: |
| **Recall@5** | Direct Terminology | 1.0000 | 1.0000 | 1.0000 |
| **MRR** | Direct Terminology | 1.0000 | 1.0000 | 1.0000 |
| **Recall@5** | Focused Paraphrase | 1.0000 | 0.6667 | 0.3333 |
| **MRR** | Focused Paraphrase | 0.5000 | 1.0000 | 1.0000 |

When queried with explicit terminology, all three retrievers achieved perfect 1.0000 scores, placing `doc-03` at rank one. 

When queried with the focused paraphrase, BM25 and vector search both achieved an MRR of 1.0000, placing an authoritative document at rank one. Interestingly, exact keyword matching recovered all three relevant passages (Recall@5 of 1.0000) through diffuse term matches (`question`, `passage`, `match`), but its first relevant result appeared at rank two (MRR of 0.5000).

---

## Diagnosing Retrieval Failures in Production

When a retrieval pipeline yields poor results in production, engineering teams frequently rush to replace their embedding models. In practice, the breakdown typically lies elsewhere in the system stack. 

Follow this diagnostic sequence before changing models:

1. **Verify Corpus Coverage**: Does the corpus actually contain the factual answers required by the user, or is the index suffering from data staleness or omission?
2. **Inspect Lexical Invariants**: Are queries failing because exact product names, error codes, or part identifiers are being diluted by vector approximations? If so, implement hybrid BM25 retrieval with Reciprocal Rank Fusion.
3. **Analyze Query Normalization**: Are conversational filler phrases dominating the embedding representation? Stripping conversational preambles often restores vector precision.
4. **Calibrate Document Chunk Boundaries**: Are document boundaries splitting related concepts across arbitrary token windows, degrading semantic cohesion?
5. **Separate Discovery from Verification**: Use retrieval to assemble candidate pools, but enforce structured metadata and policy filters before passing context to generation models.

Dense vectors provide powerful semantic capabilities, but they cannot compensate for an impoverished corpus or an uncalibrated evaluation suite. Reliable information retrieval begins with understanding the data from the bottom up.

Inspect the complete experimental pipeline, corpus json, and execution logs in our [information retrieval experiment repository](file:///C:/Users/j/afiavana/research/experiments/information_retrieval).
