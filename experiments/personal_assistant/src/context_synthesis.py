from __future__ import annotations

from pydantic import BaseModel, Field

from experiments.personal_assistant.src.ddc import DDCGapProposal
from experiments.personal_assistant.src.entities import EntityProposal
from experiments.personal_assistant.src.memory import PersonalAssistantMemoryStateUpdate


class ContextSynthesisResult(BaseModel):
    """Unified post-turn synthesis output.

    The LLM decides what context changed; deterministic services decide how to persist it.
    """

    memory_update: PersonalAssistantMemoryStateUpdate | None = None
    ddc_review_items: list[DDCGapProposal] = Field(default_factory=list)
    entity_review_items: list[EntityProposal] = Field(default_factory=list)
