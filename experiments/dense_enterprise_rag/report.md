# Dense Enterprise RAG Experiment Report

## Run contract

This report comes from `run_experiment.py`. The data is synthetic.
The run uses 96 documents and 32 cases in 16 matched pairs.
Eight cases are holdouts. The deterministic answer composer is the
primary answer path.

- Run ID: `dense-enterprise-rag-20260828-2a6222500524d893`
- Seed: `20260828`
- Documents: `96`
- Cases: `32`
- Holdouts: `8`
- Question date: `2026-08-28`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding backend: `sentence-transformers`
- Embedding fallback reason: `none`
- Python: `3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026, 10:26:10) [MSC v.1944 64 bit (AMD64)]`

## Overall retrieval result

Strict applicability uses the exact applicable document for the 24
answer cases. Answer status uses all 32 cases.

| Condition | Strict applicability | Answer status | Unsupported claim | Full / packet tokens | Retrieval P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 50.00% (12/24) | 75.00% (24/32) | 50.00% (12/24) | 368.4 / 30.2 | 4.906 ms |
| vector | 16.67% (4/24) | 75.00% (24/32) | 83.33% (20/24) | 321.3 / 30.6 | 42.561 ms |
| metadata_filtered_vector | 100.00% (24/24) | 100.00% (32/32) | 0.00% (0/24) | 95.1 / 30.5 | 41.558 ms |
| graph_guided_hybrid | 100.00% (24/24) | 100.00% (32/32) | 0.00% (0/24) | 45.4 / 34.5 | 43.604 ms |

## Clean, dense, and holdout cases

The clean and dense rows are the two members of each matched pair.
Holdout rows remain separate from threshold selection. This run uses
fixed rules and does not tune on the holdouts.

| Condition | Slice | Strict applicability | Exact fact | Correct abstention |
| --- | --- | ---: | ---: | ---: |
| bm25 | clean | 100.00% (12/12) | 100.00% (12/12) | 0.00% (0/4) |
| bm25 | dense_overlap | 0.00% (0/12) | 100.00% (12/12) | 0.00% (0/4) |
| bm25 | holdout | n/a | n/a | 0.00% (0/8) |
| vector | clean | 0.00% (0/12) | 0.00% (0/12) | 0.00% (0/4) |
| vector | dense_overlap | 33.33% (4/12) | 33.33% (4/12) | 0.00% (0/4) |
| vector | holdout | n/a | n/a | 0.00% (0/8) |
| metadata_filtered_vector | clean | 100.00% (12/12) | 100.00% (12/12) | 100.00% (4/4) |
| metadata_filtered_vector | dense_overlap | 100.00% (12/12) | 100.00% (12/12) | 100.00% (4/4) |
| metadata_filtered_vector | holdout | n/a | n/a | 100.00% (8/8) |
| graph_guided_hybrid | clean | 100.00% (12/12) | 100.00% (12/12) | 100.00% (4/4) |
| graph_guided_hybrid | dense_overlap | 100.00% (12/12) | 100.00% (12/12) | 100.00% (4/4) |
| graph_guided_hybrid | holdout | n/a | n/a | 100.00% (8/8) |

## Results by case class

| Condition | Case class | Strict applicability | Answer status |
| --- | --- | ---: | ---: |
| bm25 | authority_and_status | 50.00% (4/8) | 100.00% (8/8) |
| bm25 | missing_or_ambiguous_evidence | n/a | 0.00% (0/8) |
| bm25 | region_and_entity_scope | 50.00% (4/8) | 100.00% (8/8) |
| bm25 | version_and_supersession | 50.00% (4/8) | 100.00% (8/8) |
| vector | authority_and_status | 12.50% (1/8) | 100.00% (8/8) |
| vector | missing_or_ambiguous_evidence | n/a | 0.00% (0/8) |
| vector | region_and_entity_scope | 25.00% (2/8) | 100.00% (8/8) |
| vector | version_and_supersession | 12.50% (1/8) | 100.00% (8/8) |
| metadata_filtered_vector | authority_and_status | 100.00% (8/8) | 100.00% (8/8) |
| metadata_filtered_vector | missing_or_ambiguous_evidence | n/a | 100.00% (8/8) |
| metadata_filtered_vector | region_and_entity_scope | 100.00% (8/8) | 100.00% (8/8) |
| metadata_filtered_vector | version_and_supersession | 100.00% (8/8) | 100.00% (8/8) |
| graph_guided_hybrid | authority_and_status | 100.00% (8/8) | 100.00% (8/8) |
| graph_guided_hybrid | missing_or_ambiguous_evidence | n/a | 100.00% (8/8) |
| graph_guided_hybrid | region_and_entity_scope | 100.00% (8/8) | 100.00% (8/8) |
| graph_guided_hybrid | version_and_supersession | 100.00% (8/8) | 100.00% (8/8) |

## Compression

The packet keeps the fact, unit, scope, validity, authority, source,
and evidence path. Token counts use the stable word-token proxy in
`compression.py`.

| Condition | Mean full tokens | Mean packet tokens | Mean reduction | Required evidence retained |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 368.38 | 30.19 | 91.73% | 0.00% (0/24) |
| vector | 321.31 | 30.59 | 90.44% | 0.00% (0/24) |
| metadata_filtered_vector | 95.12 | 30.50 | 72.75% | 0.00% (0/24) |
| graph_guided_hybrid | 45.38 | 34.50 | 42.96% | 100.00% (24/24) |

## Matched pairs

The full pair record is in `results.json` under
`matched_pairs.<condition>`.

- `bm25`: `12` dense cases lost strict applicability relative to their clean pair.
- `vector`: `0` dense cases lost strict applicability relative to their clean pair.
- `metadata_filtered_vector`: `0` dense cases lost strict applicability relative to their clean pair.
- `graph_guided_hybrid`: `0` dense cases lost strict applicability relative to their clean pair.

## Failure traces

Each class has one dense trace below. The complete trace set is in
`traces.jsonl`.

### `authority_and_status`

- Case: `case-09-dense`
- Vector selected: `spec-17`
- Graph selected: `spec-17`
- Vector candidates: `spec-17, spec-19, spec-18, spec-14, spec-24`
- Graph path: `product-atlas-x200 -> firmware-4-2 -> region-europe -> document-spec-17 -> fact-spec-17-pressure-limit -> authority-product-safety`

### `missing_or_ambiguous_evidence`

- Case: `case-13-dense`
- Vector selected: `spec-11`
- Graph selected: `None`
- Vector candidates: `spec-11, spec-08, spec-05, spec-02, spec-17`
- Graph path: `none`

### `region_and_entity_scope`

- Case: `case-05-dense`
- Vector selected: `spec-17`
- Graph selected: `spec-17`
- Vector candidates: `spec-17, spec-19, spec-18, spec-14, spec-24`
- Graph path: `product-atlas-x200 -> firmware-4-2 -> region-europe -> document-spec-17 -> fact-spec-17-pressure-limit -> authority-product-safety`

### `version_and_supersession`

- Case: `case-01-dense`
- Vector selected: `spec-17`
- Graph selected: `spec-17`
- Vector candidates: `spec-17, spec-19, spec-18, spec-14, spec-24`
- Graph path: `product-atlas-x200 -> firmware-4-2 -> region-europe -> document-spec-17 -> fact-spec-17-pressure-limit -> authority-product-safety`

## Interpretation

The graph condition is useful only when its source edges are correct
and the question contains enough scope to form a valid path.
Metadata filters can solve direct field checks. The graph condition
adds relation checks for authority and supersession.

This synthetic set does not estimate enterprise failure rates. Local
latency is hardware-dependent. It is not comparable to a hosted
vendor latency claim.

## Result field paths for the article

- Overall strict applicability: `results.json` -> `conditions.<condition>.metrics.overall.strict_applicability_accuracy`
- Dense strict applicability: `results.json` -> `conditions.<condition>.metrics.dense_overlap.strict_applicability_accuracy`
- Case-class result: `results.json` -> `conditions.<condition>.metrics.by_case_class.<case_class>.strict_applicability_accuracy`
- Compression: `results.json` -> `conditions.<condition>.metrics.overall.compression`
- Retrieval latency: `results.json` -> `conditions.<condition>.metrics.overall.retrieval_latency_ms`
- Extraction audit: `extraction_results.json`
- Live generation: `live_results.json`

## Model check status

- Relation extraction: `completed`; see `extraction_results.json`.
- Live Gemini slice: `completed`; see `live_results.json`.

### Relation-extraction audit

| Measure | Value |
| --- | ---: |
| Node precision | 5.49% |
| Node recall | 5.85% |
| Edge precision | 2.08% |
| Edge recall | 2.08% |
| Scope-field accuracy | 4.17% |
| Validity-field accuracy | 0.00% |
| Authority accuracy | 0.00% |
| Invalid-edge rejection | 100.00% |
| Review rate | 100.00% |

### Live answer check

| Slice | Exact fact | Answer status | Source recall | Requests |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 15/16 (93.75%) | 15/16 (93.75%) | 12/16 (75.00%) | 16 |
| vector | 9/16 (56.25%) | 9/16 (56.25%) | 9/16 (56.25%) | 16 |
| metadata_filtered_vector | 16/16 (100.00%) | 16/16 (100.00%) | 16/16 (100.00%) | 16 |
| graph_guided_hybrid | 16/16 (100.00%) | 16/16 (100.00%) | 16/16 (100.00%) | 16 |
| full_documents | 32/32 (100.00%) | 32/32 (100.00%) | 32/32 (100.00%) | 32 |
| evidence_packet | 24/32 (75.00%) | 24/32 (75.00%) | 21/32 (65.62%) | 32 |
