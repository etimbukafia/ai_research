# Customer assistant graph memory experiment

The experiment compares four memory conditions over 24 synthetic support cases.
It uses the same customer scope, fixed skills, deterministic answer rule, and as-of date for every condition.

## Runtime

- As-of date: `2026-08-25`
- Cases: `24`
- Case classes: `conflict_authority_missing_evidence=6, direct=6, multi_step_relationship=6, temporal=6`
- Top-k: `8`
- Graph database: embedded LadybugDB, backend `pybind`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Overall metrics

| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy | Provenance precision | Unsupported claim rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.6201 | 0.0 | 0.4583 | 1.0 | 0.5 | 0.0 |
| `vector_chroma` | 0.9625 | 0.0 | 0.875 | 1.0 | 0.9708 | 0.0 |
| `graph_ladybug` | 0.9545 | 0.9545 | 0.9583 | 1.0 | 0.9292 | 0.0 |
| `hybrid_graph_vector` | 0.9545 | 0.9545 | 0.9583 | 1.0 | 0.9292 | 0.0 |

## Metrics by case class

### `direct`

| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy |
| --- | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.6667 | 0.0 | 0.6667 | 1.0 |
| `vector_chroma` | 1.0 | 0.0 | 1.0 | 1.0 |
| `graph_ladybug` | 1.0 | 1.0 | 1.0 | 1.0 |
| `hybrid_graph_vector` | 1.0 | 1.0 | 1.0 | 1.0 |

### `multi_step_relationship`

| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy |
| --- | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.4958 | 0.0 | 0.0 | None |
| `vector_chroma` | 0.8625 | 0.0 | 0.5 | None |
| `graph_ladybug` | 1.0 | 1.0 | 1.0 | None |
| `hybrid_graph_vector` | 1.0 | 1.0 | 1.0 | None |

### `temporal`

| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy |
| --- | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.6667 | 0.0 | 0.6667 | 1.0 |
| `vector_chroma` | 1.0 | 0.0 | 1.0 | 1.0 |
| `graph_ladybug` | 1.0 | 1.0 | 1.0 | 1.0 |
| `hybrid_graph_vector` | 1.0 | 1.0 | 1.0 | 1.0 |

### `conflict_authority_missing_evidence`

| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy |
| --- | ---: | ---: | ---: | ---: |
| `flat_recent` | 0.6667 | 0.0 | 0.5 | 1.0 |
| `vector_chroma` | 1.0 | 0.0 | 1.0 | 1.0 |
| `graph_ladybug` | 0.75 | 0.75 | 0.8333 | 1.0 |
| `hybrid_graph_vector` | 0.75 | 0.75 | 0.8333 | 1.0 |

## Featured case

Case `case-multi-01` is the running warranty and replacement case.
The trace records the retrieved records, graph path, time decision, cited records, and final decision for each method.

## Limits

The record set is synthetic and small. The results show the behavior of this representation and these cases.
They do not prove that graph memory is best for every customer assistant.

The experiment isolates memory retrieval. It uses fixed skills and templates instead of a language model that can change the answer policy.
