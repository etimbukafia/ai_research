import asyncio
import typer
from uuid import UUID
from rich.console import Console

from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from experiments.ader.src.agent import Ader
from experiments.ader.src.db import create_default_db

# Default test user (Alex — Regulated Profile)
DEFAULT_USER_ID = UUID("d3b07384-d113-4956-a5e2-4c5b3648a301")

app = typer.Typer(
    help="Ader: A supportive cognitive companion for neurodivergents.",
    no_args_is_help=True
)
console = Console()

style = Style.from_dict({
    'prompt': 'ansimagenta bold',
})


@app.command()
def run(query: str):
    """Run a single query against Ader."""
    async def _run():
        db = create_default_db()
        agent = Ader(db=db, user_id=DEFAULT_USER_ID)
        try:
            with console.status("[bold green]Ader is thinking...[/bold green]"):
                result = await agent.run(query)
            console.print(f"\n[bold green]Ader ❯[/bold green]")
            console.print(Markdown(result))
            console.print()
        finally:
            await agent.close()
        
    asyncio.run(_run())


@app.command()
def interactive():
    """Start an interactive chat session with Ader."""
    async def _chat():
        db = create_default_db()
        agent = Ader(db=db, user_id=DEFAULT_USER_ID)
        
        console.print("[bold blue]Ader is ready. Type 'exit' or 'quit' to end the session.[/bold blue]")
        
        session = PromptSession()
        
        try:
            while True:
                try:
                    # Use prompt_toolkit for a conversational interface (history, arrows, etc.)
                    user_input = await session.prompt_async(HTML('<prompt>\nYou ❯ </prompt>'), style=style)
                    user_input = user_input.strip()
                    
                    if user_input.lower() in ("exit", "quit"):
                        break
                    if not user_input:
                        continue
                    
                    with console.status("[bold green]Ader is thinking...[/bold green]"):
                        result = await agent.run(user_input)
                    
                    console.print(f"\n[bold green]Ader ❯[/bold green]")
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


if __name__ == "__main__":
    app()
