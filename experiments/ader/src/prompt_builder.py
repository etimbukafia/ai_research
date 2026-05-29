"""
Ader prompt assembly — combines user context and agent directive.

Leverages PromptAssembler from the harness to render Jinja2 templates
with dynamic memory state, then assembles the final system prompt.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from experiments.harness.prompt import PromptAssembler
from experiments.ader.src.memory import AderMemory


def build_ader_prompt(
    memory: AderMemory,
    user_name: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    """
    Build the final system prompt for the Ader agent by combining user context
    (dynamically rendered from memory state) and the static agent directive.

    Args:
        memory: The AderMemory object containing all state.
        user_name: The user's name for personalization.
        prompts_dir: Override path to prompts directory. Defaults to ./prompts/.

    Returns:
        The fully rendered system prompt string ready for the LLM.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    user_context_path = prompts_dir / "user_context.md"
    ader_agent_path = prompts_dir / "ader_agent_prompt.md"

    # Prepare the memory context dictionary
    memory_ctx = {
        "user_name": user_name,
        "working_memory": memory.working,
        "affective_state": memory.affective,
        "semantic_memory": memory.semantic,
        "procedural_memory": memory.procedural,
        "recent_episodes": memory.recent_episodes(n=3),  # Last 3 episodes
    }

    # Render user context
    user_context_str = user_context_path.read_text(encoding="utf-8")
    assembler = PromptAssembler(user_context_str)
    rendered_user_context = assembler.build(**memory_ctx)

    # Render agent directive (static, no dynamic vars needed)
    ader_agent_str = ader_agent_path.read_text(encoding="utf-8")
    assembler_agent = PromptAssembler(ader_agent_str)
    rendered_agent_directive = assembler_agent.build()

    # Combine: user context first (so the agent knows who they're with and why),
    # then agent directive (the system instructions).
    final_prompt = f"{rendered_user_context}\n\n---\n\n{rendered_agent_directive}"

    return final_prompt
