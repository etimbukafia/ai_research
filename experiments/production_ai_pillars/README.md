# Production AI pillars

This experiment supports the field note `Pillars of Enterprise Production AI`.

It runs one small software-access request service through the public
`enterprise-agent-harness` runtime. The workflow has three bounded child
agents:

1. `identity-specialist` resolves the requester.
2. `data-policy-specialist` resolves the product, current policy, and budget.
3. `action-specialist` creates a pending request or proposes a grant.

The application owns the JSON business records and mock handlers. The harness
owns typed tool contracts, delegation ceilings, permission checks, approval
gates, idempotency, state, trace records, and outcome verification.

## Live run

Use Python 3.11 or newer. Install the pinned dependencies:

```powershell
python -m pip install -e .
```

Set `GOOGLE_API_KEY` in the process environment. The key is never read from a
repository file and is never written to a report.

Run the full 24-case live run:

```powershell
$env:GOOGLE_API_KEY = "..."
python run_experiment.py
```

The provider is Gemini `gemini-3.5-flash-lite` through PydanticAI. The run has
one worker and waits five seconds between provider requests. This keeps the
process below 15 requests per minute. The experiment has no fixed provider
response and no second provider path.

You can run a selected live case while checking the setup:

```powershell
python run_experiment.py --case-id business-current-analytics
```

The selected run still calls the live API. It does not use a stored response.

## Reports

- `reports/results.json` contains the run manifest, raw case results, exported
  harness traces, lab evaluation scores, metrics, and threshold results.
- `reports/summary.md` contains the short result handoff for the article.
- `reports/metrics.csv` contains the measures in a tabular form.
- `traces/live_traces.jsonl` contains one sanitized trace bundle per case.

The result files are created after a live run. They must be committed with the
article when the values are reported.

## Evaluation boundary

`agent-improvement-lab` reads the exported traces through its evaluator
contracts. It does not invoke the provider again. It does not change tool
authority, policy, approval state, or workflow limits.

The case set tests business decisions, data distractors, tenant boundaries,
approval gates, bounded child authority, evidence support, idempotency, and
trace fields. The records and tools are synthetic and local. No tool calls a
real enterprise system.

The run does not measure production throughput. It does not prove general
model quality. It does not replace a security, privacy, legal, or operational
review.

