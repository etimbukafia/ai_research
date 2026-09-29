# Miranda Distortion experiment

This experiment tests access decisions for a synthetic company called Northstar Systems.

It asks whether a machine-readable semantic contract reduces a plausible but wrong business interpretation.

The case set has 24 cases in 12 matched pairs. Each pair has one clean case and one case with a controlled trap. The traps cover role names, scope, permissions, approval type, source authority, stale approval, current state, policy exceptions, status, missing evidence, ambiguous requests, and prompt injection.

The three replay conditions use the same records and cases:

- `raw-record`: a compact raw-record index.
- `retrieval`: the raw index plus prose policy text.
- `semantic`: the raw index plus the versioned semantic contract.

The prompt-injection cases are a separate security control. They do not affect the primary Miranda Distortion score.

## Run the replay

Run this command from the repository root:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' enterprise_agent_lab\experiments\miranda_distortion\run_experiment.py
```

The replay makes no provider request. It needs no API key and no Docker service.

The runner uses `agent-improvement-lab` trace contracts and its Pydantic Evals runner. The mock tools read JSON records and keep mutation state in memory. A rejected `grant_access` call remains in the trace and does not change the final state.

## Outputs

- `cases.json`: the versioned case source.
- `semantic_contract.json`: Northstar concepts, authority rules, and access rules.
- `records/`: synthetic HR, IAM, approval, policy, and note records.
- `traces.jsonl`: 72 traces: 24 cases across three conditions.
- `results.json`: case scores and condition summaries.
- `comparison.json`: raw-record comparisons with retrieval and semantic.
- `report.md`: generated replay report.
- `run_manifest.json`: model label, package versions, run controls, and SHA-256 hashes.

Live Gemini validation is not run by this deterministic command. The saved metadata keeps the planned `gemini-3.5-flash-lite` model, one worker, five-second interval, and 15 RPM limit. A future live run must read `GOOGLE_API_KEY` from the environment and save its own traces before any model-dependent result is reported.
