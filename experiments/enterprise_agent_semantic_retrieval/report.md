# Semantic retrieval for enterprise agents

This report uses the deterministic replay path. It compares raw tool and field descriptions, prose retrieval, and a typed semantic catalog.
The run uses 24 synthetic Aster Cloud cases. It makes no Gemini request and changes no source record.

## Run

- Run ID: `replay-9368d62309ac1a46`
- Cases: `24`
- Case classes: `cross_system=6, concept_resolution=6, policy_and_action=6, temporal_authority_missing=6`
- Model contract: `google:gemini-3.7-flash`
- Top-k: `8`
- Retrieval backend: `sentence-transformers/all-MiniLM-L6-v2`
- Live status: `not_run`

## Overall metrics

| Condition | Concept | Tool | Arguments | Evidence | Policy | Temporal | Final | Abstention | Unsafe action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.430556 | 0.25 | 1.0 | 0.041667 | 0.0 | 0.0 | 0.0 | 0.541667 | 0.291667 |
| `prose_rag` | 0.451389 | 0.25 | 1.0 | 0.041667 | 0.0 | 0.0 | 0.0 | 0.541667 | 0.291667 |
| `semantic_catalog` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

The primary scores are exact evaluator checks. `Final` requires every required component to match.

## Metrics by case class

### `concept_resolution`

| Condition | Concept | Tool | Evidence | Policy | Temporal | Final | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.833333 | 0.666667 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `prose_rag` | 0.833333 | 0.666667 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| `semantic_catalog` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### `cross_system`

| Condition | Concept | Tool | Evidence | Policy | Temporal | Final | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.222222 | 0.166667 | 0.0 | 0.0 | 0.0 | 0.0 | 0.666667 |
| `prose_rag` | 0.222222 | 0.166667 | 0.0 | 0.0 | 0.0 | 0.0 | 0.666667 |
| `semantic_catalog` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### `policy_and_action`

| Condition | Concept | Tool | Evidence | Policy | Temporal | Final | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.166667 |
| `prose_rag` | 0.583333 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.166667 |
| `semantic_catalog` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### `temporal_authority_missing`

| Condition | Concept | Tool | Evidence | Policy | Temporal | Final | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | 0.166667 | 0.166667 | 0.166667 | 0.0 | 0.0 | 0.0 | 0.333333 |
| `prose_rag` | 0.166667 | 0.166667 | 0.166667 | 0.0 | 0.0 | 0.0 | 0.333333 |
| `semantic_catalog` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Featured case

The featured case is `case-01`. It asks whether Aster Cloud can issue a credit and offer a plan change after a seat mismatch.

| Condition | Status | Concepts | Tools | Evidence recall | Policy | Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `raw_schema` | `answer` | 0.333333 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prose_rag` | `answer` | 0.333333 | 0.0 | 0.0 | 0.0 | 0.0 |
| `semantic_catalog` | `needs_human_review` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Integration status

`CONTRACT.md` was present: `True`.
Cases source: `enterprise_agent_lab.cases.build_cases`.
Core model validation: `core_models_or_core_files_missing`.

The evaluator uses the canonical builder cases and contract-shaped traces. This verified report does not invoke Gemini.

## Limits

The cases and context are synthetic. Replay uses a fixed deterministic output policy. The scores show this representation and these cases. They do not measure production reliability or every Gemini model.

The semantic condition carries typed fields. The prose condition carries related text without typed source, time, join, or policy fields. The raw condition carries interface fields. This design isolates the context representation. It does not prove that catalog content is correct or current.
