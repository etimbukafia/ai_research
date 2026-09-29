# Evals 101 for enterprise agents

This experiment supports the article [Evals 101 for Enterprise Agents](../../../frontend/src/content/posts/evals-101-for-enterprise-agents.md).

It tests one question:

> Does a final-answer check overstate enterprise-agent reliability?

The replay uses 24 synthetic Aster Cloud cases. Each of the 12 business
requests has one clean case and one controlled hard-negative case. The run
scores the same traces with an answer-only suite and an enterprise suite.

The enterprise suite checks evidence, source authority, time validity, tool
arguments, policy, approval, action state, and operational limits.

Run the experiment from the repository root:

```powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' enterprise_agent_lab/experiments/evals_101/run_experiment.py
```

The command uses deterministic replay. It makes no provider request and does
not need an API key. It writes the case set, traces, evaluator results,
comparison, promotion records, manifest, and report in this directory.

Live Gemini validation is a separate plan. It uses one worker, a five-second
interval, and the 15 RPM project limit. It is marked `not_run` in the replay
artifacts.

