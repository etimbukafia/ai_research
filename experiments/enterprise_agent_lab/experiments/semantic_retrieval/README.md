# Semantic retrieval for enterprise agents

This folder contains the evaluator for the article experiment.

The experiment tests one question:

> Does a typed semantic catalog help an enterprise agent resolve business concepts, select tools, use correct arguments, cite evidence, apply policy, respect time, and stop when evidence is missing?

It uses 24 deterministic synthetic Aster Cloud cases. There are six cases in
each class:

- `concept_resolution`
- `cross_system`
- `policy_and_action`
- `temporal_authority_missing`

Each case runs under the same interface and the same top-eight limit. Only the
context representation changes:

1. `raw_schema`: tool names and field names without business definitions.
2. `prose_rag`: related prose without typed source, time, join, or policy fields.
3. `semantic_catalog`: typed catalog entries with definitions, mappings, time,
   authority, source IDs, preconditions, and negative rules.

## Run the replay evaluation

Replay needs no API key. The command below uses the requested local virtual
environment:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' `
  'enterprise_agent_lab\experiments\semantic_retrieval\run_experiment.py'
```

The default run tries the cached
`sentence-transformers/all-MiniLM-L6-v2` model in offline mode. Use
`--no-vectors` when the model is not cached:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' `
  'enterprise_agent_lab\experiments\semantic_retrieval\run_experiment.py' `
  --no-vectors
```

The evaluator uses SQLite FTS5 through the common
`search_business_context(query, account_scope, concept_type, top_k)` shape.
When the local MiniLM model is available, it merges lexical and vector scores
with a fixed tie order. Replay uses a deterministic output fixture. It makes
no Gemini request.

## Output files

- `results.json`: exact case scores, aggregate scores, class scores, and live status.
- `traces.jsonl`: one contract-shaped trace for every case and condition.
- `featured_trace.json`: the full trace for `case-01` under all conditions.
- `run_manifest.json`: model, prompt, source hashes, package versions, retrieval
  settings, and the builder contract status.
- `report.md`: a short handoff report with measured replay values.

The primary evaluator checks are exact. They cover concept resolution, tool
selection, typed arguments, evidence, policy, temporal validity, final status,
claims, draft action, abstention, safety, and operation counts.

## Live mode

Live mode requires `GOOGLE_API_KEY` and uses the pinned model name
`google:gemini-3.5-flash-lite` unless `GEMINI_MODEL` supplies another
non-`latest` model suffix. The core runner spaces requests at five seconds by
default. Set `GEMINI_RPM` and `GEMINI_RATE_SAFETY` for another project limit.

```powershell
$env:GOOGLE_API_KEY = '...'
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' `
  'enterprise_agent_lab\experiments\semantic_retrieval\run_experiment.py' `
  --mode live --case case-01 --condition semantic_catalog
```

The evaluator first looks for an importable live runner in the lab core. The
public `CONTRACT.md` currently defines the CLI and model surfaces but does not
define a Python live-run function. If the core runner is absent, the narrow
adapter uses a local PydanticAI agent with the same `AgentDecision` fields and
the retrieved context tool. Replay does not use this adapter.

Do not put API keys, model caches, SQLite files, or live traces in the
repository.

## Contract note

The evaluator follows the public names in `enterprise_agent_lab/CONTRACT.md`:
`AgentDecision`, `RunTrace`, `ToolCallRecord`, `PolicyCheck`,
`SemanticRetriever.search_business_context`, and the stable tool names.

`cases.json` is the exact JSON serialization of
`enterprise_agent_lab.cases.build_cases()` with `featured_case_id` set to
`case-01`. The runner loads `build_cases()` and verifies the JSON copy before
the replay starts. It uses the canonical `case-01` through `case-24` IDs,
`account-01` IDs, and the core argument names such as `at`, `period_start`,
`period_end`, `target_plan`, `target_seats`, and `action`.
