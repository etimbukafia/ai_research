# Enterprise Agent Lab

This is a small local test bench for enterprise-agent articles.

The lab uses synthetic Aster Cloud records. It has named tools, a typed
semantic catalog, policy checks, trace output, and a Gemini adapter. It does
not connect to an enterprise system. Its action tools create drafts only.

Read CONTRACT.md before changing a public model, tool, or retrieval method.

## Run it

From the repository root:

~~~powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli demo
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli run --case case-01 --mode replay
~~~

Replay mode is local and needs no API key. Live mode needs GOOGLE_API_KEY:

~~~powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli run --case case-01 --mode live
~~~

The default model is google:gemini-3.5-flash-lite. Set GEMINI_MODEL to change
the model name. Do not use a moving latest alias. Live requests use a
five-second spacing target, which stays below a 15 RPM project limit. Set
GEMINI_RPM and GEMINI_RATE_SAFETY to change the limit and safety margin.

## Safety boundary

The source store is SQLite. The action tools create DraftAction records.
They do not change account, contract, billing, usage, or ticket records.
Approval requests are local files under the configured run directory.

## Reuse the agent packages

The lab keeps two integration boundaries:

- `enterprise_agent_lab.integrations.assistant_harness` puts the semantic
  retriever behind `ai-assistant-harness`. The harness owns the principal,
  read-tool permission, session state, citation checks, and response safety.
- `enterprise_agent_lab.integrations.agent_improvement` converts the lab's
  cases and traces to `agent-improvement-lab` contracts. The Lab owns generic
  evaluation, scoring, and experiment reports.

The enterprise lab remains the owner of Aster Cloud records, the semantic
catalog, named tools, policies, draft actions, and Gemini's PydanticAI agent.
The shared harness is read-only. Draft and approval tools stay behind the
enterprise policy layer.

The local development environment can install both repositories with:

~~~powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m pip install -e 'C:\Users\Etimbuk\projects\evals\agent-improvement-lab[evals]'
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m pip install -e 'C:\Users\Etimbuk\projects\ai-assistant-harness'
~~~

Run one replay case through Agent Improvement Lab:

~~~powershell
& 'C:\Users\Etimbuk\projects\.projects\Scripts\python.exe' -m enterprise_agent_lab.cli evaluate --case case-02
~~~
