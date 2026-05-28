import os
from pathlib import Path

def load_prompts(prompt_folder: str = "prompts") -> tuple[str, str]:
    """Load memory synthesizer and Ader agent prompts.

    Returns a tuple of the contents of ``memory_synthesizer_agent_prompt.md`` and
    ``ader_agent_prompt.md`` located in the given ``prompt_folder`` relative to
    this module's parent directory.
    """
    base_path = os.path.join(os.path.dirname(__file__).parent, prompt_folder)
    memory_path = Path(base_path) / "memory_synthesizer_agent_prompt.md"
    ader_path = Path(base_path) / "ader_agent_prompt.md"

    if not memory_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {memory_path}")
    if not ader_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {ader_path}")

    memory_prompt = memory_path.read_text()
    ader_prompt = ader_path.read_text()
    return (memory_prompt, ader_prompt)

