# Experiment report

Status: **complete**

This report records the local comparison for the article. The labels are manual labels for the 12-document corpus.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Top-k: `5`
- Corpus SHA-256: `ad08a657eadf8bd2327dfc27a80f5540ed52ddb16067e7476c16a903ff788acd`
- Python: `3.14.6`
- Platform: `Windows-11-10.0.26200-SP0`

## Method rules

- `exact_keyword` counts unique query terms that occur in the title or body.
- `bm25` uses the visible tokenizer and parameters in `run_experiment.py`.
- `vector_chroma` uses normalized MiniLM vectors in an in-memory Chroma collection with cosine distance. The reported score is `1 - cosine distance`.
- Ties use ascending document ID after score order.

## Results

### `main`

> How can a search system find relevant text when the user's words differ from the words in the documents?

Manual relevant IDs: `doc-04, doc-05, doc-06`

| Method | Top five (score) | Recall@5 | MRR |
| --- | --- | ---: | ---: |
| `exact_keyword` | `doc-03` 3; `doc-01` 2; `doc-07` 2; `doc-09` 2; `doc-11` 2 | 0.000000 | 0.000000 |
| `bm25` | `doc-03` 3.940272; `doc-07` 3.066683; `doc-11` 2.807139; `doc-01` 2.306270; `doc-09` 2.169365 | 0.000000 | 0.000000 |
| `vector_chroma` | `doc-01` 0.565786 (distance 0.434214); `doc-03` 0.509476 (distance 0.490524); `doc-12` 0.486152 (distance 0.513848); `doc-02` 0.442068 (distance 0.557932); `doc-07` 0.388660 (distance 0.611340) | 0.000000 | 0.000000 |

### Main-query notes

- Exact keyword top-five IDs: `doc-03, doc-01, doc-07, doc-09, doc-11`.
- BM25 top-five IDs: `doc-03, doc-07, doc-11, doc-01, doc-09`.
- Vector top-five IDs: `doc-01, doc-03, doc-12, doc-02, doc-07`.
- Use the scores as observations for this corpus. They do not prove answer quality.

### `validation_direct`

> Which data structure maps terms to document IDs for fast exact keyword search?

Manual relevant IDs: `doc-03`

| Method | Top five (score) | Recall@5 | MRR |
| --- | --- | ---: | ---: |
| `exact_keyword` | `doc-03` 9; `doc-01` 6; `doc-12` 3; `doc-02` 2; `doc-07` 2 | 1.000000 | 1.000000 |
| `bm25` | `doc-03` 14.367039; `doc-01` 8.841265; `doc-12` 3.322997; `doc-02` 2.394935; `doc-05` 1.674318 | 1.000000 | 1.000000 |
| `vector_chroma` | `doc-03` 0.747693 (distance 0.252307); `doc-01` 0.616479 (distance 0.383521); `doc-12` 0.465065 (distance 0.534935); `doc-02` 0.447129 (distance 0.552871); `doc-07` 0.375902 (distance 0.624098) | 1.000000 | 1.000000 |

### `validation_paraphrase`

> How can a machine match a question to a passage that uses different language?

Manual relevant IDs: `doc-04, doc-05, doc-06`

| Method | Top five (score) | Recall@5 | MRR |
| --- | --- | ---: | ---: |
| `exact_keyword` | `doc-01` 2; `doc-06` 2; `doc-02` 1; `doc-04` 1; `doc-05` 1 | 1.000000 | 0.500000 |
| `bm25` | `doc-06` 3.569617; `doc-01` 3.404928; `doc-02` 1.990336; `doc-04` 1.733171; `doc-11` 1.703236 | 0.666667 | 1.000000 |
| `vector_chroma` | `doc-05` 0.297868 (distance 0.702132); `doc-07` 0.282367 (distance 0.717633); `doc-12` 0.266105 (distance 0.733895); `doc-03` 0.262775 (distance 0.737225); `doc-01` 0.259628 (distance 0.740372) | 0.333333 | 1.000000 |

## Package versions

```text
chromadb: 1.0.20
numpy: 2.5.2
sentence-transformers: 5.1.0
torch: 2.13.0
transformers: 4.57.6
```

## Limitations

- The corpus has 12 short documents and three queries.
- The relevance labels are manual and binary. They do not capture partial usefulness.
- Exact keyword search and BM25 use no stemming or synonym expansion.
- Vector similarity measures representation closeness. It does not verify facts or evidence.
- The embedding model is small and general. Results can change with another model.
- Chroma runs in memory. The experiment does not test persistence, updates, latency, or scale.
- Runtime depends on the local CPU, package versions, model cache, and model download state.
