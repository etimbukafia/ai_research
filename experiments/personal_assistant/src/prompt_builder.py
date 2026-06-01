"""
Personal Assistant prompt assembly - combines user context and agent directive.

Leverages PromptAssembler from the harness to render Jinja2 templates
with dynamic memory state, then assembles the final system prompt.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from experiments.harness.prompt import PromptAssembler
from experiments.personal_assistant.src.memory import PersonalAssistantMemory


def build_personal_assistant_prompt(
    memory: PersonalAssistantMemory,
    user_name: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    """
    Build the final system prompt for the Personal Assistant agent by combining user context
    (dynamically rendered from memory state) and the static agent directive.

    Args:
        memory: The PersonalAssistantMemory object containing all state.
        user_name: The user's name for personalization.
        prompts_dir: Override path to prompts directory. Defaults to ./prompts/.

    Returns:
        The fully rendered system prompt string ready for the LLM.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    user_context_path = prompts_dir / "user_context.md"
    personal_assistant_agent_path = prompts_dir / "personal_assistant_agent_prompt.md"

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
    personal_assistant_agent_str = personal_assistant_agent_path.read_text(encoding="utf-8")
    assembler_agent = PromptAssembler(personal_assistant_agent_str)
    rendered_agent_directive = assembler_agent.build()

    # Combine: user context first (so the agent knows who they're with and why),
    # then agent directive (the system instructions).
    final_prompt = f"{rendered_user_context}\n\n---\n\n{rendered_agent_directive}"

    return final_prompt


def load_prompts(prompt_folder: str = "prompts") -> tuple[str, str]:
    base_path = Path(__file__).parent / prompt_folder
    memory_path = base_path / "memory_synthesizer_agent_prompt.md"
    personal_assistant_path = base_path / "personal_assistant_agent_prompt.md"

    if not memory_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {memory_path}")
    if not personal_assistant_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {personal_assistant_path}")

    memory_template = memory_path.read_text(encoding="utf-8")
    memory_prompt = PromptAssembler(memory_template).build(**memory_synthesizer_prompt_context())

    return memory_prompt, personal_assistant_path.read_text(encoding="utf-8")


def memory_synthesizer_prompt_context() -> dict[str, Any]:
    return {
        "confidence_threshold": 0.7,
        "working_memory_fields": [
            {"name": "current_focus", "description": "What the user is focused on in the current session."},
            {"name": "active_goals", "description": "Outcomes the user is trying to move forward."},
            {"name": "active_tasks", "description": "Concrete next tasks the user is working on."},
            {"name": "open_loops", "description": "Unresolved obligations, worries, or loose ends that may need resurfacing."},
            {"name": "pending_decisions", "description": "Decisions the user has not made yet."},
            {"name": "waiting_on", "description": "JSON object where each key is the thing needed and each value is the person, system, event, or condition blocking it."},
            {"name": "waiting_on_resolved", "description": "List of waiting-on item keys that are now resolved and should be removed."},
        ],
        "capacity_fields": [
            {"name": "stress_level", "description": "Higher means more stress or anxiety."},
            {"name": "energy_level", "description": "Higher means more available energy."},
            {"name": "cognitive_load", "description": "Higher means the user is holding more complexity."},
            {"name": "social_energy", "description": "Higher means more capacity for interaction."},
            {"name": "emotional_regulation", "description": "Higher means steadier emotional state."},
            {"name": "executive_function", "description": "Higher means easier task initiation and switching."},
        ],
        "semantic_memory_fields": [
            {"name": "preferences", "description": "Stable likes, defaults, or preferred ways of working."},
            {"name": "triggers", "description": "Situations or patterns that reliably create friction or distress."},
            {"name": "best_focus_time", "description": "The user's clearest focus window: `morning`, `afternoon`, or `night`."},
            {"name": "sensitive_to_noise", "description": "Whether the user explicitly mentions noise sensitivity."},
            {"name": "prefers_direct_language", "description": "Whether the user prefers direct, explicit language."},
            {"name": "dislikes_open_ended_questions", "description": "Whether the user prefers fewer open-ended questions."},
        ],
        "procedural_memory_fields": [
            {"name": "successful_interventions", "description": "Assistant tactics that helped."},
            {"name": "routines_that_worked", "description": "Repeatable routines that helped."},
            {"name": "effective_grouping_strategies", "description": "Ways of grouping tasks that reduced friction."},
            {"name": "preferred_planning_structures", "description": "Formats the user responds well to."},
        ],
        "episodic_memory_fields": [
            {"name": "episode_title", "description": "Short label for the remembered moment."},
            {"name": "episode_summary", "description": "What happened and why it matters for future continuity."},
            {"name": "episode_category", "description": "Type of episode to save."},
            {"name": "episode_people", "description": "People involved or mentioned."},
            {"name": "episode_related_goals", "description": "Goals this episode connects to."},
            {"name": "episode_commitments", "description": "Things the user or assistant committed to."},
            {"name": "episode_follow_ups", "description": "Future check-ins or reminders the assistant should bring back."},
            {"name": "episode_risks", "description": "Risks, blockers, or failure modes connected to this episode."},
            {"name": "episode_salience", "description": "Importance for future recall from 0.0 to 1.0."},
        ],
        "episode_categories": [
            "conversation",
            "commitment",
            "decision",
            "preference_signal",
            "life_event",
            "task_progress",
            "follow_up",
            "reflection",
            "goal",
        ],
        "confidence_key_examples": [
            "current_focus",
            "active_goals",
            "waiting_on",
            "episode_title",
            "episode_risks",
        ],
    }
