import asyncio
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

from experiments.personal_assistant.src.agent import PersonalAssistant
from experiments.personal_assistant.src.db import DEFAULT_DB_PATH, seed_default_profile

app = typer.Typer(
    help="Personal Assistant: a supportive personal companion.",
    no_args_is_help=True,
)
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

        console.print("[bold blue]Personal Assistant is ready. Type 'exit' or 'quit' to end the session.[/bold blue]")
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

                    with console.status("[bold green]Personal Assistant is thinking...[/bold green]"):
                        result = await agent.run(user_input)

                    console.print("\n[bold green]Personal Assistant >[/bold green]")
                    console.print(Markdown(result))

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


if __name__ == "__main__":
    app()
