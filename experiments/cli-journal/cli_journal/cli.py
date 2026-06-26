from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .async_utils import run_async
from .agent import JournalAgent
from .db import DEFAULT_DB_PATH, DEFAULT_PROFILE_ID, JournalDatabase
from .models import Entity, Thought
from .models import JournalSession
from .organizer import JournalOrganizer
from .priming import PrimingStore
from .runtime import captured_output, configure_quiet_runtime, quiet_third_party_output


configure_quiet_runtime()
console = Console()
prompt_style = Style.from_dict({"prompt": "ansicyan bold"})
CHAT_SYNTHESIS_IDLE_SECONDS = 180.0
synthesis_lock = threading.Lock()
SLASH_COMMANDS = {
    "/add": "capture a thought",
    "/entity": "add an entity: /entity NAME TYPE [DESCRIPTION]",
    "/recent": "show recent thoughts",
    "/session": "show current chat session state",
    "/organize": "run pending thought organization",
    "/consolidate": "promote repeated episodes into semantic facts",
    "/logs": "show recent app logs",
    "/quit": "exit chat",
    "/exit": "exit chat",
}


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command, description in SLASH_COMMANDS.items():
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )


def main(argv: list[str] | None = None) -> None:
    configure_quiet_runtime()
    parser = _build_parser()
    args = parser.parse_args(argv)
    db = JournalDatabase(args.db)
    organizer = JournalOrganizer(db, profile_id=args.profile_id)

    try:
        args.handler(args, db, organizer)
    finally:
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="journal", description="Capture and query a local CLI journal.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID, help="Profile id to use.")
    sub = parser.add_subparsers(required=True)

    init = sub.add_parser("init", help="Create the database and profile.")
    init.add_argument("--name", default="User")
    init.set_defaults(handler=_cmd_init)

    add = sub.add_parser("add", help="Capture a thought.")
    add.add_argument("text")
    add.add_argument("--entity", action="append", default=[], help="Entity as Name:type. Can be repeated.")
    add.set_defaults(handler=_cmd_add)

    chat = sub.add_parser("chat", help="Start an interactive journal agent.")
    chat.add_argument("--session-id", help="Resume a specific session id.")
    chat.add_argument("--new", action="store_true", help="Start a new session.")
    chat.set_defaults(handler=_cmd_chat)

    thoughts = sub.add_parser("thoughts", help="Work with thoughts.")
    thoughts_sub = thoughts.add_subparsers(required=True)
    thoughts_list = thoughts_sub.add_parser("list")
    thoughts_list.add_argument("--limit", type=int, default=20)
    thoughts_list.set_defaults(handler=_cmd_thoughts_list)
    thoughts_search = thoughts_sub.add_parser("search")
    thoughts_search.add_argument("query")
    thoughts_search.add_argument("--limit", type=int, default=10)
    thoughts_search.set_defaults(handler=_cmd_thoughts_search)

    entities = sub.add_parser("entities", help="Work with entities.")
    entities_sub = entities.add_subparsers(required=True)
    entities_add = entities_sub.add_parser("add")
    entities_add.add_argument("name")
    entities_add.add_argument("type", nargs="?", default="entity")
    entities_add.add_argument("description", nargs="?", default="")
    entities_add.add_argument("--alias", action="append", default=[])
    entities_add.set_defaults(handler=_cmd_entities_add)
    entities_list = entities_sub.add_parser("list")
    entities_list.add_argument("--limit", type=int, default=50)
    entities_list.set_defaults(handler=_cmd_entities_list)

    episodes = sub.add_parser("episodes", help="Work with episodic memory.")
    episodes_sub = episodes.add_subparsers(required=True)
    episodes_list = episodes_sub.add_parser("list")
    episodes_list.add_argument("--limit", type=int, default=20)
    episodes_list.set_defaults(handler=_cmd_episodes_list)

    facts = sub.add_parser("facts", help="Work with semantic memory.")
    facts_sub = facts.add_subparsers(required=True)
    facts_list = facts_sub.add_parser("list")
    facts_list.add_argument("--limit", type=int, default=50)
    facts_list.set_defaults(handler=_cmd_facts_list)

    priming = sub.add_parser("priming", help="Work with the local ChromaDB priming index.")
    priming_sub = priming.add_subparsers(required=True)
    priming_rebuild = priming_sub.add_parser("rebuild", help="Rebuild ChromaDB from local journal records.")
    priming_rebuild.set_defaults(handler=_cmd_priming_rebuild)
    priming_search = priming_sub.add_parser("search", help="Search the local priming index.")
    priming_search.add_argument("query")
    priming_search.add_argument("--limit", type=int, default=8)
    priming_search.set_defaults(handler=_cmd_priming_search)

    consolidate = sub.add_parser("consolidate", help="Promote repeated episodes into semantic facts.")
    consolidate.add_argument("--min-support", type=int, default=3)
    consolidate.set_defaults(handler=_cmd_consolidate)

    organize = sub.add_parser("organize", help="Run the LLM organization worker.")
    organize.add_argument("--limit", type=int, default=10)
    organize.add_argument("--watch", action="store_true", help="Keep processing pending thoughts.")
    organize.add_argument("--sleep", type=float, default=5.0, help="Seconds to wait between watch cycles.")
    organize.set_defaults(handler=_cmd_organize)

    sessions = sub.add_parser("sessions", help="Work with chat sessions.")
    sessions_sub = sessions.add_subparsers(required=True)
    sessions_list = sessions_sub.add_parser("list")
    sessions_list.add_argument("--limit", type=int, default=20)
    sessions_list.set_defaults(handler=_cmd_sessions_list)

    logs = sub.add_parser("logs", help="Inspect app logs.")
    logs_sub = logs.add_subparsers(required=True)
    logs_list = logs_sub.add_parser("list")
    logs_list.add_argument("--limit", type=int, default=20)
    logs_list.set_defaults(handler=_cmd_logs_list)

    return parser


def _cmd_init(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    db.ensure_profile(args.profile_id, args.name)
    console.print(
        Panel.fit(
            f"[bold green]Initialized journal[/bold green]\n[dim]{Path(args.db).expanduser()}[/dim]",
            title="CLI Journal",
            border_style="green",
        )
    )


def _cmd_add(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    db.ensure_profile(args.profile_id)
    log_count_before = db.count_logs(args.profile_id)
    with console.status("[bold green]Saving thought...[/bold green]"):
        thought, episode = organizer.capture(
            args.text,
            explicit_entities=args.entity,
        )
    _print_capture_result(thought, episode, logs_written=db.count_logs(args.profile_id) - log_count_before)


def _cmd_chat(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    db.ensure_profile(args.profile_id)
    session = _load_chat_session(db, args)
    agent = JournalAgent(db, profile_id=args.profile_id)
    console.print(
        Panel(
            "[bold]Commands[/bold]\n"
            "/add, /entity, /recent, /session, /organize, /consolidate, /logs, /quit",
            title=f"Journal Chat [{session.session_id}]",
            border_style="cyan",
        )
    )
    prompt = PromptSession(
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
    )
    while True:
        idle_timer = threading.Timer(
            CHAT_SYNTHESIS_IDLE_SECONDS,
            lambda: _run_memory_synthesis(organizer, reason="idle synthesis"),
        )
        idle_timer.daemon = True
        idle_timer.start()
        try:
            line = prompt.prompt(HTML("<prompt>\njournal > </prompt>"), style=prompt_style).strip()
        except (EOFError, KeyboardInterrupt):
            idle_timer.cancel()
            console.print()
            _run_memory_synthesis(organizer, reason="exit synthesis")
            return
        finally:
            idle_timer.cancel()
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            _run_memory_synthesis(organizer, reason="exit synthesis")
            return
        if line.startswith("/add "):
            log_count_before = db.count_logs(args.profile_id)
            thought, _episode = organizer.capture(line[5:])
            reply = f"saved {thought.thought_id}"
            session.active_thought_ids = _tail_unique([*session.active_thought_ids, thought.thought_id], 20)
            session.last_exchange = {"user": line[5:], "assistant": reply}
            session.rolling_summary = _update_session_summary(session, f"Captured thought: {line[5:]}")
            db.save_session(session)
            _print_capture_result(
                thought,
                _episode,
                logs_written=db.count_logs(args.profile_id) - log_count_before,
            )
            continue
        if line.startswith("/entity "):
            parts = line.split(maxsplit=3)
            if len(parts) < 3:
                console.print("[dim]usage: /entity NAME TYPE [DESCRIPTION][/dim]")
                continue
            description = parts[3] if len(parts) > 3 else ""
            entity = organizer.add_entity(parts[1], parts[2], description)
            reply = f"saved {entity.entity_id}"
            session.active_entity_ids = _tail_unique([*session.active_entity_ids, entity.entity_id], 20)
            session.last_exchange = {"user": line, "assistant": reply}
            session.rolling_summary = _update_session_summary(session, f"Added entity: {entity.canonical_name}")
            db.save_session(session)
            console.print(f"[bold green]{reply}[/bold green]")
            continue
        if line == "/session":
            console.print(_session_panel(session))
            continue
        if line == "/recent":
            _print_thoughts(db.list_thoughts(args.profile_id, limit=10), title="Recent Thoughts")
            continue
        if line == "/logs":
            _print_logs(db.list_logs(args.profile_id, limit=10))
            continue
        if line == "/consolidate":
            try:
                facts = organizer.consolidate()
                reply = f"created {len(facts)} semantic fact(s)"
            except RuntimeError as exc:
                reply = str(exc)
            session.last_exchange = {"user": line, "assistant": reply}
            session.rolling_summary = _update_session_summary(session, "Ran consolidation")
            db.save_session(session)
            console.print(f"[bold green]{reply}[/bold green]" if reply.startswith("created") else f"[bold red]{reply}[/bold red]")
            continue
        if line == "/organize":
            try:
                count = organizer.organize_pending(limit=10)
                reply = f"organized {count} thought(s)"
            except RuntimeError as exc:
                reply = str(exc)
            session.last_exchange = {"user": line, "assistant": reply}
            session.rolling_summary = _update_session_summary(session, "Ran organization worker")
            db.save_session(session)
            console.print(f"[bold green]{reply}[/bold green]" if reply.startswith("organized") else f"[bold red]{reply}[/bold red]")
            continue

        with console.status("[bold green]Journal is thinking...[/bold green]"):
            try:
                result = agent.run(line, session=session)
            except RuntimeError as exc:
                console.print(Panel(str(exc), title="Journal Chat Error", border_style="red"))
                continue
        _save_agent_turn(db, session, line, result.answer)
        if result.response and result.response.should_save_thought:
            thought, _episode = organizer.capture(
                line,
            )
            session.active_thought_ids = _tail_unique([*session.active_thought_ids, thought.thought_id], 20)
            db.save_session(session)
        console.print(Panel(Markdown(result.answer), title="Journal", border_style="green"))
        _print_agent_warnings(result.memory_error, result.priming_error)
        _print_memory_warning(organizer)


def _cmd_thoughts_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    _print_thoughts(db.list_thoughts(args.profile_id, limit=args.limit), title="Thoughts")


def _cmd_thoughts_search(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    table = Table(title="Thought Search")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Date", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Thought")
    for thought, score in db.search_thoughts(args.profile_id, args.query, limit=args.limit):
        table.add_row(f"{score:.1f}", thought.created_at[:10], thought.thought_type, thought.body)
    console.print(table)


def _cmd_entities_add(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    entity = organizer.add_entity(args.name, args.type, args.description, aliases=args.alias)
    console.print(
        Panel.fit(
            f"[bold green]{entity.canonical_name}[/bold green]\n"
            f"[dim]{entity.entity_id} | {entity.type}[/dim]",
            title="Entity Saved",
            border_style="green",
        )
    )


def _cmd_entities_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    _print_entities(db.list_entities(args.profile_id, limit=args.limit))


def _cmd_episodes_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    with quiet_third_party_output() as output_streams:
        results = run_async(
            organizer.memory.recall(
                "journal episodes",
                user_id=args.profile_id,
                memory_type="episode",
                limit=args.limit,
            )
        )
    _log_cli_output(db, args.profile_id, "mem0.episodes_list.output", captured_output(output_streams))
    console.print(Panel.fit(f"{len(results)} result(s)", title="mem0 Episodes", border_style="blue"))
    for item in results:
        text = item.get("memory") or item.get("text") or item.get("content") or ""
        console.print(Panel(str(text).strip(), border_style="blue"))


def _cmd_facts_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    with quiet_third_party_output() as output_streams:
        results = run_async(
            organizer.memory.recall(
                "semantic facts",
                user_id=args.profile_id,
                memory_type="semantic_fact",
                limit=args.limit,
            )
        )
    _log_cli_output(db, args.profile_id, "mem0.facts_list.output", captured_output(output_streams))
    console.print(Panel.fit(f"{len(results)} result(s)", title="mem0 Semantic Facts", border_style="magenta"))
    for item in results:
        text = item.get("memory") or item.get("text") or item.get("content") or ""
        console.print(Panel(str(text).strip(), border_style="magenta"))


def _cmd_priming_rebuild(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    with console.status("[bold green]Rebuilding ChromaDB priming index...[/bold green]"):
        with quiet_third_party_output() as output_streams:
            count = PrimingStore.from_env().rebuild(db, profile_id=args.profile_id)
    _log_cli_output(db, args.profile_id, "priming.rebuild.output", captured_output(output_streams))
    console.print(
        Panel.fit(
            f"[bold green]indexed={count}[/bold green]",
            title="Priming Rebuild",
            border_style="green",
        )
    )


def _cmd_priming_search(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    with quiet_third_party_output() as output_streams:
        hits = PrimingStore.from_env().search(args.query, profile_id=args.profile_id, limit=args.limit)
    _log_cli_output(db, args.profile_id, "priming.search.output", captured_output(output_streams))
    table = Table(title="Priming Search")
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Distance", justify="right", style="cyan")
    table.add_column("Source", style="dim", no_wrap=True)
    table.add_column("Document")
    for hit in hits:
        distance = f"{hit.distance:.3f}" if hit.distance is not None else ""
        table.add_row(hit.memory_type, distance, hit.source_id, hit.document)
    console.print(table)


def _cmd_consolidate(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    with console.status("[bold green]Consolidating semantic memory...[/bold green]"):
        try:
            facts = organizer.consolidate(min_support=args.min_support)
        except RuntimeError as exc:
            console.print(Panel(str(exc), title="Consolidation Error", border_style="red"))
            return
    table = Table(title=f"Created {len(facts)} Semantic Fact(s)")
    table.add_column("Subject", style="cyan")
    table.add_column("Predicate", style="magenta")
    table.add_column("Value")
    for fact in facts:
        table.add_row(fact.subject_entity_id, fact.predicate, fact.value)
    console.print(table)
    _print_memory_warning(organizer)


def _cmd_organize(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    db.ensure_profile(args.profile_id)
    while True:
        try:
            count = organizer.organize_pending(limit=args.limit)
            console.print(f"[bold green]organized={count}[/bold green]")
        except RuntimeError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            return
        if not args.watch:
            return
        time.sleep(args.sleep)



def _cmd_sessions_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    table = Table(title="Chat Sessions")
    table.add_column("Session")
    table.add_column("Last Active", no_wrap=True)
    table.add_column("Status", style="cyan")
    table.add_column("Name")
    for session in db.list_sessions(args.profile_id, limit=args.limit):
        table.add_row(session.session_id, session.last_active_at[:19], session.status, session.name)
    console.print(table)


def _cmd_logs_list(args, db: JournalDatabase, organizer: JournalOrganizer) -> None:
    _print_logs(db.list_logs(args.profile_id, limit=args.limit))


def _log_cli_output(db: JournalDatabase, profile_id: str, source: str, output: str) -> None:
    if not output:
        return
    db.add_log(
        profile_id=profile_id,
        level="debug",
        source=source,
        message=output[:2000],
        context={},
    )


def _load_chat_session(db: JournalDatabase, args) -> JournalSession:
    if args.session_id:
        session = db.get_session(args.profile_id, args.session_id)
        if session is None:
            raise ValueError(f"Unknown session: {args.session_id}")
        return session
    if not args.new:
        session = db.get_latest_active_session(args.profile_id)
        if session is not None:
            return session
    return db.create_session(JournalSession(profile_id=args.profile_id))


def _print_thoughts(thoughts: list[Thought], *, title: str) -> None:
    table = Table(title=title)
    table.add_column("Date", no_wrap=True)
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Label", style="cyan")
    table.add_column("Thought")
    table.add_column("ID", style="dim", no_wrap=True)
    for thought in thoughts:
        table.add_row(
            thought.created_at[:10],
            thought.thought_type,
            thought.thought or "",
            thought.body,
            thought.thought_id,
        )
    console.print(table)


def _print_entities(entities: list[Entity]) -> None:
    table = Table(title="Entities")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Description")
    table.add_column("ID", style="dim", no_wrap=True)
    for entity in entities:
        table.add_row(entity.canonical_name, entity.type, entity.description, entity.entity_id)
    console.print(table)


def _print_capture_result(thought: Thought, episode, *, logs_written: int = 0) -> None:
    lines = [
        f"thought={thought.thought_id}",
        f"episode={episode.episode_id}",
        f"type={thought.thought_type}",
    ]
    lines.append("organization=queued")
    if logs_written:
        lines.append(f"logs={logs_written}")
    console.print(Panel("\n".join(lines), border_style="green"))


def _print_logs(rows) -> None:
    table = Table(title="App Logs")
    table.add_column("Time", no_wrap=True)
    table.add_column("Level", style="yellow", no_wrap=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Message")
    table.add_column("Context", style="dim")
    for row in rows:
        context = row["context_json"]
        try:
            context = json.dumps(json.loads(context), ensure_ascii=False)
        except Exception:
            pass
        table.add_row(row["created_at"][:19], row["level"], row["source"], row["message"], context)
    console.print(table)


def _print_memory_warning(organizer: JournalOrganizer) -> None:
    if organizer.last_memory_error or organizer.last_priming_error:
        console.print("[dim]background memory warning logged; run `journal logs list` or `/logs`[/dim]")


def _print_agent_warnings(memory_error: str | None, priming_error: str | None) -> None:
    if priming_error:
        console.print(Panel(priming_error, title="ChromaDB priming skipped", border_style="yellow"))
    if memory_error:
        console.print(Panel(memory_error, title="mem0 recall skipped", border_style="yellow"))


def _run_memory_synthesis(organizer: JournalOrganizer, *, reason: str) -> None:
    if not synthesis_lock.acquire(blocking=False):
        return
    try:
        _run_memory_synthesis_locked(organizer, reason=reason)
    finally:
        synthesis_lock.release()


def _run_memory_synthesis_locked(organizer: JournalOrganizer, *, reason: str) -> None:
    pending = organizer.db.list_organization_jobs(organizer.profile_id, status="pending", limit=1)
    unconsolidated = organizer.db.list_episodes(organizer.profile_id, unconsolidated=True, limit=3)
    if not pending and len(unconsolidated) < 3:
        return
    organized = 0
    facts = []
    errors: list[str] = []
    with console.status(f"[bold green]Synthesizing journal memory ({reason})...[/bold green]"):
        if pending:
            try:
                organized = organizer.organize_pending(limit=10)
            except Exception as exc:
                _log_synthesis_error(organizer, reason, "organize", exc)
                errors.append(str(exc))
        unconsolidated = organizer.db.list_episodes(organizer.profile_id, unconsolidated=True, limit=3)
        if len(unconsolidated) >= 3:
            try:
                facts = organizer.consolidate()
            except Exception as exc:
                _log_synthesis_error(organizer, reason, "consolidate", exc)
                errors.append(str(exc))
    if organized or facts:
        console.print(
            Panel.fit(
                f"organized={organized}\nsemantic_facts={len(facts)}",
                title=f"Memory Synthesis: {reason}",
                border_style="green",
            )
        )
    if errors:
        console.print(f"[dim]memory synthesis warning logged ({reason}); run `/logs`[/dim]")
    _print_memory_warning(organizer)


def _log_synthesis_error(organizer: JournalOrganizer, reason: str, operation: str, exc: Exception) -> None:
    organizer.db.add_log(
        profile_id=organizer.profile_id,
        level="warning",
        source=f"synthesis.{operation}",
        message=str(exc),
        context={"reason": reason},
    )


def _save_agent_turn(db: JournalDatabase, session: JournalSession, user_message: str, assistant_reply: str) -> None:
    session.recent_queries = _tail_unique([*session.recent_queries, user_message], 20)
    session.last_exchange = {"user": user_message, "assistant": assistant_reply}
    session.rolling_summary = _update_session_summary(session, f"Chatted: {user_message}")
    db.save_session(session)


def _session_panel(session: JournalSession) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("session_id", session.session_id)
    table.add_row("started_at", session.started_at)
    table.add_row("last_active_at", session.last_active_at)
    table.add_row("active_thoughts", ", ".join(session.active_thought_ids) or "none")
    table.add_row("active_entities", ", ".join(session.active_entity_ids) or "none")
    table.add_row("recent_queries", ", ".join(session.recent_queries[-5:]) or "none")
    table.add_row("last_user", session.last_exchange.get("user") if session.last_exchange else "none")
    table.add_row("last_assistant", session.last_exchange.get("assistant") if session.last_exchange else "none")
    table.add_row("summary", session.rolling_summary or "none")
    return Panel(table, title="Session Memory", border_style="cyan")


def _update_session_summary(session: JournalSession, event: str) -> str:
    lines = [line for line in session.rolling_summary.splitlines() if line.strip()]
    lines.append(event.strip())
    return "\n".join(lines[-12:])


def _tail_unique(values: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if value in result:
            result.remove(value)
        result.append(value)
    return result[-limit:]
