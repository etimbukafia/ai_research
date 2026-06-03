import asyncio
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from experiments.personal_assistant.src.agent import PersonalAssistant
from experiments.personal_assistant.src.db import DEFAULT_DB_PATH, DEFAULT_PROFILE_ID, PersonalAssistantDatabase, seed_default_profile
from experiments.personal_assistant.src.ddc import DDCReviewService
from experiments.personal_assistant.src.entities import ContextEntityService, EntityReviewItem
from experiments.personal_assistant.src.planning import PlannerContinuation
from experiments.personal_assistant.src.review_text import review_text

app = typer.Typer(
    help="Personal Assistant: a supportive personal companion.",
    no_args_is_help=True,
)
ddc_app = typer.Typer(help="Review demand-driven context items.")
app.add_typer(ddc_app, name="ddc")
console = Console()

style = Style.from_dict({"prompt": "ansimagenta bold"})


@app.command()
def run(query: str, db_path: str = str(DEFAULT_DB_PATH)):
    """Run a single query against Personal Assistant."""

    async def _run():
        agent = PersonalAssistant(db_path=db_path)
        try:
            with console.status("[bold green]Personal Assistant is thinking...[/bold green]"):
                result = await agent.run(query)
            console.print("\n[bold green]Personal Assistant >[/bold green]")
            console.print(Markdown(result))
            console.print()
        finally:
            await agent.close()

    asyncio.run(_run())


@app.command()
def interactive(db_path: str = str(DEFAULT_DB_PATH)):
    """Start an interactive chat session with Personal Assistant."""

    async def _chat():
        agent = PersonalAssistant(db_path=db_path)

        console.print(
            "[bold blue]Personal Assistant is ready. "
            "Type 'exit' or 'quit' to end the session. Type '/review' for pending context reviews.[/bold blue]"
        )
        session = PromptSession()

        try:
            while True:
                try:
                    user_input = await session.prompt_async(HTML("<prompt>\nYou > </prompt>"), style=style)
                    user_input = user_input.strip()

                    if user_input.lower() in ("exit", "quit"):
                        break
                    if not user_input:
                        continue
                    if user_input.startswith("/review"):
                        if user_input == "/review":
                            await _handle_review_command(db_path, agent)
                        else:
                            console.print("[dim]Use /review to show pending context reviews.[/dim]")
                        continue

                    with console.status("[bold green]Personal Assistant is thinking...[/bold green]"):
                        result = await agent.run(user_input)

                    suppress_result = False
                    while _has_pending_continuation(db_path):
                        resumed = await _handle_pending_continuation_questions(db_path, agent)
                        if not resumed:
                            suppress_result = True
                            break
                        result = resumed

                    if not suppress_result:
                        console.print("\n[bold green]Personal Assistant >[/bold green]")
                        console.print(Markdown(result))
                    with console.status("[dim]Checking context review queue...[/dim]"):
                        await agent.wait_for_context_review_updates()
                    _print_review_notice(db_path)

                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            console.print("[bold blue]Saving state...[/bold blue]")
            try:
                await agent.close()
            except Exception as e:
                console.print(f"[bold red]Failed to save state:[/bold red] {e}")

        console.print("\n[bold blue]Goodbye![/bold blue]")

    asyncio.run(_chat())


@app.command()
def seed(name: str = "User", db_path: str = str(DEFAULT_DB_PATH)):
    """Create the SQLite DB and initialize the single assistant profile."""
    db = seed_default_profile(db_path, name=name)
    db.close()
    console.print(f"[bold green]Seeded Personal Assistant DB:[/bold green] {db_path}")


@ddc_app.command("pending")
def ddc_pending(db_path: str = str(DEFAULT_DB_PATH)):
    """List pending DDC review items."""
    db, service = _ddc_service(db_path)
    try:
        _print_pending_review_items(service)
    finally:
        db.close()


@ddc_app.command("show")
def ddc_show(review_id: str, db_path: str = str(DEFAULT_DB_PATH)):
    """Show one DDC review item in detail."""
    db, service = _ddc_service(db_path)
    try:
        item = service.show(DEFAULT_PROFILE_ID, review_id)
        if item is None:
            raise typer.BadParameter(f"Unknown DDC review item: {review_id}")
        console.print(f"[bold]DDC Review Item:[/bold] {item.review_id}\n")
        console.print(f"[bold]Category:[/bold] {item.category}")
        console.print(f"[bold]Risk:[/bold] {item.risk}")
        console.print(f"[bold]Status:[/bold] {item.status}")
        console.print(f"[bold]Source Task:[/bold] {item.source_task}\n")
        console.print("[bold]Missing Context:[/bold]")
        console.print(_review_text(item.missing_context))
        console.print("\n[bold]Proposed Memory:[/bold]")
        console.print(_review_text(item.proposed_memory))
        console.print("\n[bold]Reason:[/bold]")
        console.print(_review_text(item.reason))
        console.print("\n[bold]Actions:[/bold] approve | reject | revise")
    finally:
        db.close()


@ddc_app.command("approve")
def ddc_approve(review_id: str, db_path: str = str(DEFAULT_DB_PATH)):
    """Approve and promote one DDC review item."""
    db, service = _ddc_service(db_path)
    try:
        item = service.approve(DEFAULT_PROFILE_ID, review_id)
        console.print(f"[bold green]Approved and promoted:[/bold green] {item.review_id}")
    finally:
        db.close()


@ddc_app.command("reject")
def ddc_reject(review_id: str, db_path: str = str(DEFAULT_DB_PATH)):
    """Reject one DDC review item without mutating memory."""
    db, service = _ddc_service(db_path)
    try:
        item = service.reject(DEFAULT_PROFILE_ID, review_id)
        console.print(f"[bold yellow]Rejected:[/bold yellow] {item.review_id}")
    finally:
        db.close()


@ddc_app.command("revise")
def ddc_revise(review_id: str, proposed_memory: str, db_path: str = str(DEFAULT_DB_PATH)):
    """Revise proposed memory text, then approve and promote it."""
    db, service = _ddc_service(db_path)
    try:
        item = service.revise(DEFAULT_PROFILE_ID, review_id, proposed_memory)
        console.print(f"[bold green]Revised, approved, and promoted:[/bold green] {item.review_id}")
    finally:
        db.close()


@ddc_app.command("log")
def ddc_log(limit: int = 20, db_path: str = str(DEFAULT_DB_PATH)):
    """Show recent DDC cycle logs."""
    db, service = _ddc_service(db_path)
    try:
        logs = service.logs(DEFAULT_PROFILE_ID, limit=limit)
        table = Table(title="Recent DDC Cycles")
        table.add_column("ID", no_wrap=True)
        table.add_column("Review", no_wrap=True)
        table.add_column("Action")
        table.add_column("Category")
        table.add_column("Promoted Memory")
        for log in logs:
            table.add_row(
                log.cycle_id,
                log.review_id,
                log.action,
                log.category,
                _truncate(log.promoted_memory or "", 72),
            )
        console.print(table)
        if not logs:
            console.print("[dim]No DDC cycle logs yet.[/dim]")
    finally:
        db.close()


def _ddc_service(db_path: str) -> tuple[PersonalAssistantDatabase, DDCReviewService]:
    db = PersonalAssistantDatabase(db_path)
    return db, DDCReviewService(db)


def _review_services(db_path: str) -> tuple[PersonalAssistantDatabase, DDCReviewService, ContextEntityService]:
    db = PersonalAssistantDatabase(db_path)
    return db, DDCReviewService(db), ContextEntityService(db)


async def _handle_review_command(db_path: str, agent: PersonalAssistant) -> None:
    await agent.invalidate_memory_cache(DEFAULT_PROFILE_ID)
    db, service, entity_service = _review_services(db_path)
    try:
        items = service.pending(DEFAULT_PROFILE_ID)
        entity_items = entity_service.pending(DEFAULT_PROFILE_ID)
        if not items and not entity_items:
            console.print("[dim]No pending context reviews.[/dim]")
            return

        console.print(f"[bold]Pending Context Reviews:[/bold] {len(items) + len(entity_items)}")
        console.print("[dim]For each item, use Tab to choose approve/reject/revise, then Enter.[/dim]")
        for item in items:
            current = service.show(DEFAULT_PROFILE_ID, item.review_id)
            if current is None or current.status != "pending":
                continue
            _print_inline_review_item(current)
            action, inline_revision = await _prompt_review_action()
            if action == "approve":
                approved = service.approve(DEFAULT_PROFILE_ID, current.review_id)
                await agent.invalidate_memory_cache(DEFAULT_PROFILE_ID)
                console.print(f"[bold green]Approved:[/bold green] {approved.review_id}")
            elif action == "reject":
                rejected = service.reject(DEFAULT_PROFILE_ID, current.review_id)
                console.print(f"[bold yellow]Rejected:[/bold yellow] {rejected.review_id}")
            elif action == "revise":
                if inline_revision:
                    proposed_memory = inline_revision
                else:
                    proposed_memory = await PromptSession().prompt_async(
                        HTML("<prompt>Revised memory > </prompt>"),
                        default=current.proposed_memory,
                        style=style,
                    )
                if proposed_memory.strip():
                    revised = service.revise(DEFAULT_PROFILE_ID, current.review_id, proposed_memory.strip())
                    await agent.invalidate_memory_cache(DEFAULT_PROFILE_ID)
                    console.print(f"[bold green]Revised and approved:[/bold green] {revised.review_id}")
                else:
                    console.print("[dim]Skipped empty revision.[/dim]")
        for item in entity_items:
            current = entity_service.show(DEFAULT_PROFILE_ID, item.review_id)
            if current is None or current.status != "pending":
                continue
            _print_inline_entity_review_item(current)
            action, _inline_revision = await _prompt_review_action()
            if action == "approve":
                approved = entity_service.approve(DEFAULT_PROFILE_ID, current.review_id)
                await agent.invalidate_memory_cache(DEFAULT_PROFILE_ID)
                console.print(f"[bold green]Approved entity:[/bold green] {approved.name}")
            elif action == "reject":
                rejected = entity_service.reject(DEFAULT_PROFILE_ID, current.review_id)
                console.print(f"[bold yellow]Rejected entity:[/bold yellow] {rejected.review_id}")
            elif action == "revise":
                revised_fields = await _prompt_entity_revision(current)
                if revised_fields is None:
                    console.print("[dim]Skipped empty entity revision.[/dim]")
                    continue
                revised = entity_service.revise(DEFAULT_PROFILE_ID, current.review_id, **revised_fields)
                await agent.invalidate_memory_cache(DEFAULT_PROFILE_ID)
                console.print(f"[bold green]Revised and approved entity:[/bold green] {revised.name}")
    finally:
        db.close()


async def _handle_pending_continuation_questions(db_path: str, agent: PersonalAssistant) -> str | None:
    db = PersonalAssistantDatabase(db_path)
    try:
        continuation = db.get_pending_planner_continuation(DEFAULT_PROFILE_ID)
    finally:
        db.close()
    if continuation is None:
        return None

    answers = await _prompt_continuation_answers(continuation)
    if answers is None:
        return None

    answer_text = _format_continuation_answers(continuation, answers)
    with console.status("[bold green]Personal Assistant is continuing...[/bold green]"):
        return await agent.run(answer_text)


def _has_pending_continuation(db_path: str) -> bool:
    db = PersonalAssistantDatabase(db_path)
    try:
        return db.get_pending_planner_continuation(DEFAULT_PROFILE_ID) is not None
    finally:
        db.close()


async def _prompt_continuation_answers(continuation: PlannerContinuation) -> dict[int, str] | None:
    questions = continuation.blocking_questions
    if not questions:
        return None

    console.print()
    console.print(f"[bold yellow]I need {len(questions)} red checklist item(s) resolved before I can continue.[/bold yellow]")
    console.print("[dim]Enter saves and moves forward. Tab skips forward. Shift+Tab goes back.[/dim]")

    answers = [""] * len(questions)
    idx = 0
    while 0 <= idx < len(questions):
        item = questions[idx]
        console.print()
        console.print(f"[bold cyan][{idx + 1}/{len(questions)}] {item.label or item.category}[/bold cyan]")
        console.print(f"[white]{item.question}[/white]")
        if item.why_needed:
            console.print(f"[dim]Why: {item.why_needed}[/dim]")

        action, value = await _prompt_context_answer(default=answers[idx])
        answers[idx] = value.strip()
        if action == "previous":
            idx = max(0, idx - 1)
        else:
            idx += 1

    missing = [pos + 1 for pos, answer in enumerate(answers) if not answer.strip()]
    if missing:
        console.print(
            "[dim]Still waiting on answer(s) for question "
            f"{', '.join(str(pos) for pos in missing)}. The task will stay paused.[/dim]"
        )
        return None

    console.print()
    console.print("[bold green]Continuing with:[/bold green]")
    for pos, (item, answer) in enumerate(zip(questions, answers), start=1):
        console.print(f"[cyan]{pos}. {item.label or item.category}:[/cyan] [white]{answer}[/white]")
    return {pos: answer for pos, answer in enumerate(answers, start=1)}


async def _prompt_context_answer(default: str = "") -> tuple[str, str]:
    state = {"text": default}
    bindings = KeyBindings()

    @bindings.add("tab")
    def _(event) -> None:
        state["text"] = event.app.current_buffer.text
        event.app.exit(result="__next__")

    @bindings.add("s-tab")
    def _(event) -> None:
        state["text"] = event.app.current_buffer.text
        event.app.exit(result="__previous__")

    session = PromptSession(key_bindings=bindings)
    result = await session.prompt_async(
        HTML("<prompt>Answer > </prompt>"),
        default=default,
        style=style,
    )
    if result == "__previous__":
        return "previous", state["text"]
    if result == "__next__":
        return "next", state["text"]
    return "next", result


def _format_continuation_answers(continuation: PlannerContinuation, answers: dict[int, str]) -> str:
    lines = [
        "Answers to the paused task's blocking context checklist:",
    ]
    for pos, gap in enumerate(continuation.blocking_questions, start=1):
        lines.append(f"{pos}. Question: {gap.question}")
        lines.append(f"   Answer: {answers.get(pos, '').strip()}")
    lines.append("")
    lines.append("Use these answers for the paused task and continue.")
    return "\n".join(lines)


def _print_review_notice(db_path: str) -> None:
    db, service, entity_service = _review_services(db_path)
    try:
        pending_count = len(service.pending(DEFAULT_PROFILE_ID)) + len(entity_service.pending(DEFAULT_PROFILE_ID))
    finally:
        db.close()
    if pending_count:
        console.print(f"\n[dim][Review] {pending_count} context item(s) pending. Type /review.[/dim]")


def _print_pending_review_items(service: DDCReviewService) -> None:
    items = service.pending(DEFAULT_PROFILE_ID)
    table = Table(title="Pending Context Reviews")
    table.add_column("ID", no_wrap=True)
    table.add_column("Category")
    table.add_column("Risk")
    table.add_column("Source")
    table.add_column("Proposed Memory")
    for item in items:
        table.add_row(
            item.review_id,
            item.category,
            item.risk,
            _review_text(item.source_task),
            _review_text(item.proposed_memory),
        )
    console.print(table)
    if not items:
        console.print("[dim]No pending context reviews.[/dim]")
    else:
        console.print(
            "[dim]Use the ddc CLI commands to approve, reject, or revise review items.[/dim]"
        )


def _print_inline_review_item(item) -> None:
    console.print()
    console.print(f"[bold cyan]Review:[/bold cyan] [white]{item.review_id}[/white]")
    console.print(
        f"[bold magenta]Category:[/bold magenta] [white]{item.category}[/white]  "
        f"[bold yellow]Risk:[/bold yellow] [white]{item.risk}[/white]"
    )
    _print_review_block("Source", _review_text(item.source_task), "blue")
    _print_review_block("Missing Context", _review_text(item.missing_context), "yellow")
    _print_review_block("Proposed Memory", _review_text(item.proposed_memory), "green")
    _print_review_block("Reason", _review_text(item.reason), "cyan")


def _print_inline_entity_review_item(item: EntityReviewItem) -> None:
    console.print()
    console.print(f"[bold cyan]Entity Review:[/bold cyan] [white]{item.review_id}[/white]")
    console.print(
        f"[bold magenta]Type:[/bold magenta] [white]{item.entity_type}[/white]  "
        f"[bold yellow]Risk:[/bold yellow] [white]{item.risk}[/white]"
    )
    _print_review_block("Entity", item.name, "green")
    _print_review_block("Aliases", ", ".join(item.aliases) or "none", "blue")
    _print_review_block("Context", item.description, "white")
    _print_review_block("Source", item.source_task, "blue")
    _print_review_block("Reason", item.reason, "cyan")


def _print_review_block(label: str, value: str, color: str) -> None:
    console.print(f"[bold {color}]{label}:[/bold {color}]")
    console.print(f"[white]{value}[/white]")


async def _prompt_review_action() -> tuple[str, str | None]:
    actions = ["approve", "reject", "revise"]
    state = {"idx": 0}
    bindings = KeyBindings()

    def _cycle(event) -> None:
        state["idx"] = (state["idx"] + 1) % len(actions)
        action = actions[state["idx"]]
        event.app.current_buffer.document = Document(action, cursor_position=len(action))

    @bindings.add("tab")
    def _(event) -> None:
        _cycle(event)

    session = PromptSession(key_bindings=bindings)
    while True:
        action = await session.prompt_async(
            HTML("<prompt>Action > </prompt>"),
            default=actions[state["idx"]],
            style=style,
        )
        action = action.strip().lower()
        if action.startswith("revise "):
            return "revise", action.removeprefix("revise ").strip()
        if action in actions:
            return action, None
        console.print("[dim]Choose approve, reject, or revise. You can also type: revise <new memory>.[/dim]")


async def _prompt_entity_revision(item: EntityReviewItem) -> dict | None:
    session = PromptSession()
    name = await session.prompt_async(HTML("<prompt>Entity name > </prompt>"), default=item.name, style=style)
    entity_type = await session.prompt_async(HTML("<prompt>Entity type > </prompt>"), default=item.entity_type, style=style)
    aliases = await session.prompt_async(
        HTML("<prompt>Aliases comma-separated > </prompt>"),
        default=", ".join(item.aliases),
        style=style,
    )
    description = await session.prompt_async(
        HTML("<prompt>Context > </prompt>"),
        default=item.description,
        style=style,
    )
    if not name.strip() or not entity_type.strip() or not description.strip():
        return None
    return {
        "name": name.strip(),
        "entity_type": entity_type.strip(),
        "aliases": [alias.strip() for alias in aliases.split(",") if alias.strip()],
        "description": description.strip(),
    }


def _truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _review_text(value: str) -> str:
    return review_text(value)


if __name__ == "__main__":
    app()
