from __future__ import annotations

from typing import Literal, Optional

from typing import Any

from pydantic import BaseModel, Field, model_validator

from experiments.personal_assistant.src.planning import MissingInfoItem


class ContextGap(BaseModel):
    summary: str = "I need a few details before I can continue."
    items: list[MissingInfoItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "ContextGap":
        if not self.items:
            raise ValueError("context gaps require at least one missing information item")
        return self

    @property
    def blocking_items(self) -> list[MissingInfoItem]:
        return [item for item in self.items if item.blocks_execution] or self.items

    def user_message(self) -> str:
        items = self.blocking_items
        if len(items) == 1:
            return items[0].question
        lines = [self.summary.strip() or "I need a few details before I can continue."]
        for idx, item in enumerate(items, start=1):
            label = item.label.strip() or item.category.strip() or f"Item {idx}"
            lines.append(f"\n{idx}. {label}\nQuestion: {item.question.strip()}")
            if item.why_needed.strip():
                lines.append(f"Why: {item.why_needed.strip()}")
        return "\n".join(lines)


class AssistantExecutionResult(BaseModel):
    kind: Literal["final", "context_gap"] = "final"
    final_response: Optional[str] = None
    context_gap: Optional[ContextGap] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_context_gap(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            gap = data.get("context_gap")
            if isinstance(gap, MissingInfoItem):
                data["context_gap"] = ContextGap(items=[gap])
            elif isinstance(gap, dict) and "items" not in gap and "question" in gap:
                data["context_gap"] = {"items": [gap]}
        return data

    @model_validator(mode="after")
    def validate_result_shape(self) -> "AssistantExecutionResult":
        if self.kind == "final" and not self.final_response:
            raise ValueError("final execution results require final_response")
        if self.kind == "context_gap" and self.context_gap is None:
            raise ValueError("context_gap execution results require context_gap")
        return self

    def user_message(self) -> str:
        if self.kind == "context_gap" and self.context_gap is not None:
            return self.context_gap.user_message()
        return self.final_response or ""
