# CLI Journal

A local-first command line journal for capturing raw thoughts, organizing them
with an LLM, and recalling them through layered memory using Mem0.

CLI Journal saves every entry immediately to SQLite, then turns raw notes into
structured episodes, entities, tags, and semantic memory over time. It uses
ChromaDB for local familiarity search and mem0 for durable episodic and semantic
recall.

## Install

```powershell
python -m pip install -e .
```

Create a `.env` file or export the relevant keys:

```text
GEMINI_API_KEY=...
JOURNAL_MEM0_AGENT_ID=cli-journal
```

See [.env.example](.env.example) for the full configuration surface.

## Quick Start

```powershell
python -m cli_journal --db .\journal.sqlite3 init --name "You"
python -m cli_journal --db .\journal.sqlite3 add "Need to check checkout logs before filing the issue"
python -m cli_journal --db .\journal.sqlite3 organize
python -m cli_journal --db .\journal.sqlite3 chat
```

If installed as a package, use `journal`:

```powershell
journal add "Book a room at Dumas Hotel"
journal chat
```

## What It Does

- Captures thoughts quickly without waiting on an LLM.
- Extracts and links entities, including `@person` mentions.
- Organizes raw entries into types, tags, significance, and salience.
- Synthesizes repeated episodes into semantic fact hints and promoted facts.
- Provides conversational recall through `journal chat`.

## Core Commands

```text
add                  Capture a raw thought.
chat                 Start the interactive journal shell.
organize             Classify and enrich queued thoughts.
consolidate          Turn repeated episodes into semantic memory.
thoughts list        Show recent captured thoughts.
entities list        Show known entities.
priming search       Search the local ChromaDB familiarity index.
logs list            Inspect background memory and synthesis logs.
```

Inside chat:

```text
/add Need to check the checkout bug logs
/entity Janet person
/recent
/organize
/consolidate
/logs
/quit
```

## Memory Model

```text
SQLite   exact local records: thoughts, episodes, entities, sessions, jobs
ChromaDB local priming: fast familiarity search over rendered records
mem0     durable recall: episodic memories and promoted semantic facts
Gemini   structured functions: organization, grouping, abstraction, merge
```

Capture is intentionally raw. Background memory synthesis handles organization
and consolidation after capture, including on chat idle and chat exit.
