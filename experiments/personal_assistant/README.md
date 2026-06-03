# Personal Assistant Source

This package contains the runtime for the experimental personal assistant in `experiments/personal_assistant`.

The assistant is a multi-agent, review-gated context system. It is best described as a self-updating context layer, not a self-updating agent: the assistant does not rewrite its own code, tools, or policies. It updates memory, review items, approved context, and knowledge-base entities through explicit user review.

## Runtime Shape

The assistant has these internal agents:

- `intent classifier`: routes the turn.
- `planner`: creates and resolves a context checklist before execution.
- `main assistant`: performs the user-facing task.
- `verifier`: checks task responses when the route requires verification.
- `context synthesizer`: observes completed turns and proposes memory updates, DDC review items, and entity review items.

Supporting services:

- `AssistantFSM`: lightweight routing and state decisions.
- `PlannerRuntimeService`: enforces red/green checklist blocking and continuations.
- `DDCReviewService`: manages review-gated durable memory/context.
- `ContextEntityService`: manages review-gated knowledge-base entities.
- `PersonalContextRenderer`: renders approved SQLite-backed context into the runtime prompt.
- `PersonalAssistantDatabase`: SQLite persistence for memory, DDC reviews, entities, continuations, and context revision state.

## Turn Flow

```text
user message
-> FSM/classifier routes the turn
-> planner runs for task-like turns
-> planner builds a checklist and resolves what it can from approved context, memory, KB, or the current turn
-> red checklist items pause execution and are answered inline
-> main assistant executes only after required context is resolved
-> verifier checks the result when needed
-> context synthesizer proposes memory/DDC/entity updates
-> /review approves, rejects, or revises durable context
```

The main invariant is:

```text
The main assistant should not execute a task while required checklist items are red.
```

## Red/Green Checklist

Planner checklist items are stored in `TaskPlan.gaps` as `MissingInfoItem`.

Each item can include:

- `answer`
- `answer_source`
- `confidence`
- `required_confidence`
- `status`: `green` or `red`
- `risk_level`: `low`, `medium`, or `high`

Risk thresholds:

```text
low    -> 0.45
medium -> 0.75
high   -> 0.90
```

These confidence details are internal. The CLI only shows the human-facing question and why it matters.

## Continuations

When required context is missing, the planner or main assistant saves a `PlannerContinuation`.

The interactive CLI then asks for all red checklist items inline:

```text
Enter      saves and moves forward
Tab        skips forward
Shift+Tab  goes back
```

Answers are used immediately as task-scoped context. They only become durable context after review.

## Context Synthesis

Post-turn synthesis is unified in `ContextSynthesizerAgent`.

It returns `ContextSynthesisResult`:

```text
memory_update
ddc_review_items[]
entity_review_items[]
```

Persistence is deterministic:

- memory updates are applied through `PersonalAssistantMemory.apply_update`
- DDC proposals are persisted through `DDCReviewService`
- entity proposals are persisted through `ContextEntityService`

The synthesizer should arbitrate between DDC and entities:

- Preferences, rules, workflows, schedule defaults, and communication defaults belong in DDC.
- Named concepts, people, tools, systems, documents, projects, and capabilities belong in entity review.

## DDC Review

DDC items are proposed but not durable until approved.

Use `/review` in interactive chat:

```text
/review
```

For each item:

- `approve`: promote it into reusable memory/context
- `reject`: discard it
- `revise`: edit before approving

Tab cycles review actions.

Approved DDC items are rendered into future runtime context by `PersonalContextRenderer`.

## Knowledge Base Entities

Entity proposals are also review-gated.

Approved entities are stored in SQLite and exported to markdown files under:

```text
experiments/personal_assistant/knowledge_base/
```

Each entity type gets its own markdown file, for example:

```text
knowledge_base/concept.md
knowledge_base/person.md
knowledge_base/tool.md
```

Entity types are free-form. The model can propose types like `concept`, `person`, `tool`, `system`, `component`, `document`, `project`, or another task-specific label.

## Configuration

Config is loaded from `experiments/.env` by default.

Required:

```text
GEMINI_API_KEY
```

Model settings:

```text
DEFAULT_AGENT_MODEL
PERSONAL_ASSISTANT_AGENT_MODEL
CONTEXT_SYNTHESIZER_AGENT_MODEL
PLANNER_AGENT_MODEL
VERIFIER_AGENT_MODEL
INTENT_CLASSIFIER_AGENT_MODEL
```

If a per-agent model is not set, it falls back to `DEFAULT_AGENT_MODEL`.

`CONTEXT_SYNTHESIZER_AGENT_MODEL` also falls back to the legacy `MEMORY_SYNTHESIZER_AGENT_MODEL` env var for compatibility.

## Running

From the repo root:

```powershell
python -m experiments.personal_assistant.app interactive
```

Seed or reset the default profile:

```powershell
python -m experiments.personal_assistant.app seed --name Mike
```

Run one query:

```powershell
python -m experiments.personal_assistant.app run "Make me a weekly research plan"
```

Use a specific SQLite DB:

```powershell
python -m experiments.personal_assistant.app interactive --db-path C:\path\to\personal_assistant.sqlite3
```

From another directory in `cmd.exe`:

```bat
set PYTHONPATH=C:\Users\j\afiavana\research
python -m experiments.personal_assistant.app interactive --db-path C:\Users\j\afiavana\research\experiments\personal_assistant\personal_assistant.sqlite3
```

From another directory in PowerShell:

```powershell
$env:PYTHONPATH = "C:\Users\j\afiavana\research"
python -m experiments.personal_assistant.app interactive --db-path C:\Users\j\afiavana\research\experiments\personal_assistant\personal_assistant.sqlite3
```

## Important Files

- `agent.py`: main orchestration and agent wiring.
- `orchestration.py`: FSM and route models.
- `planning.py`: task plans, checklist items, red/green gate, continuations.
- `execution.py`: main-assistant execution result and execution-time context gaps.
- `context_synthesis.py`: unified context synthesis output schema.
- `ddc.py`: DDC review models and promotion service.
- `entities.py`: entity review, approval, and markdown KB export.
- `context_renderer.py`: approved runtime context rendering.
- `db.py`: SQLite schema and persistence.
- `prompt_builder.py`: prompt loading/rendering helpers.
- `prompts/`: agent prompts.
- `tests/`: focused regression tests.

## Tests

Focused checks used during development:

```powershell
python -m pytest experiments\personal_assistant\src\tests\test_agent_prompts.py experiments\personal_assistant\src\tests\test_config.py experiments\personal_assistant\src\tests\test_ddc.py experiments\personal_assistant\src\tests\test_entities.py experiments\personal_assistant\src\tests\test_planning.py experiments\personal_assistant\src\tests\test_orchestration.py
```

Compile key files:

```powershell
python -m compileall experiments\personal_assistant\src experiments\personal_assistant\config.py
```
