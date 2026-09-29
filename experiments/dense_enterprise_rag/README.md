# When Similar Documents Break Enterprise RAG

This directory contains the reproducible experiment for the field note
`When Similar Documents Break Enterprise RAG`.

The experiment tests one rule:

> Use similarity to find possible evidence. Use explicit scope relations to
> decide which evidence applies.

The data is synthetic. It contains 96 documents, 32 cases, 16 matched pairs,
and 8 holdout cases. The documents use the same product, firmware, region,
pressure, and limit terms. Their scope, time, status, and authority differ.

## Run the deterministic experiment

From this directory, run:

```text
python -B build_graph.py
python -B run_experiment.py
python -B -m unittest -v test_dense_enterprise_rag.py
```

`run_experiment.py` runs these conditions over the same cases:

1. BM25 lexical retrieval.
2. Vector retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
3. Metadata-filtered vector retrieval.
4. Graph-guided hybrid retrieval.

The preferred embedding backend is the pinned MiniLM model. A clean local
checkout can run without a model cache. In that case the run uses a stable
hashed-token fallback and records the backend and reason in `results.json` and
`run_manifest.json`. This fallback is a local execution path. It is not a
claim about MiniLM quality.

The graph is a local NetworkX directed multigraph. The JSON export contains
stable node and edge IDs. No Docker service, hosted vector database, or
external graph database is required.

## Run the model checks

The relation audit uses PydanticAI and
`google:gemini-3.5-flash-lite` on 16 documents. The live answer slice uses
8 cases, 4 conditions, 2 trials, and 64 requests. Trial 1 sends full document
context. Trial 2 sends an evidence packet.

Both commands read `GOOGLE_API_KEY` from the process environment. They never
write or print the key. The client waits five seconds between requests. This
keeps the run below the project limit of 15 requests per minute.

```text
python -B extract_relations.py
python -B run_live.py
```

If the environment has no key, both artifacts record `not_run` with a clear
reason. The live runner writes `.live_checkpoint.json` after every completed
request. Re-run `python -B run_live.py` to resume a partial run.

## Result files

- `documents.json`: the 96 controlled source documents.
- `cases.json`: the 32 cases and their expected evidence.
- `graph_schema.json`: node, edge, and validation rules.
- `graph.json`: the gold graph export.
- `results.json`: deterministic metrics and result-field paths.
- `traces.jsonl`: one trace for every case and retrieval condition.
- `extraction_results.json`: the separate relation-extraction audit.
- `live_results.json`: the separate Gemini generation slice.
- `report.md`: a compact report with counts, tables, and failure traces.
- `run_manifest.json`: source hashes, package versions, model, and run rules.

The article must cite numeric values only from checked fields in `results.json`,
`extraction_results.json`, and `live_results.json`. A missing live run has no
numeric live result.
