from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


PlannerContinuationStatus = Literal["pending", "completed", "cancelled"]
PlannerContextCategory = Literal[
    "person",
    "preference",
    "schedule_rule",
    "communication_rule",
    "decision_rule",
    "recurring_workflow",
    "commitment",
    "boundary",
    "project",
    "tool_procedure",
    "topic_concept",
]
PlannerRiskLevel = Literal["low", "medium", "high"]
PlannerAnswerSource = Literal[
    "approved_context",
    "memory",
    "knowledge_base",
    "current_turn",
    "assumption",
    "unknown",
]
PlannerChecklistStatus = Literal["green", "red"]


class MissingInfoItem(BaseModel):
    label: str = ""
    category: str
    question: str
    why_needed: str = ""
    blocks_execution: bool = False
    risk_level: PlannerRiskLevel
    answer: Optional[str] = None
    answer_source: PlannerAnswerSource = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    required_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: Optional[PlannerChecklistStatus] = None
    suggested_ddc_category: Optional[PlannerContextCategory] = None
    suggested_review_text: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_gap(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "why_needed" not in data and "reason" in data:
                data["why_needed"] = data["reason"]
            if "label" not in data:
                data["label"] = _label_from_question(data.get("question", ""))
            if "suggested_ddc_category" not in data and data.get("category") in PlannerContextCategory.__args__:
                data["suggested_ddc_category"] = data["category"]
        return data

    @model_validator(mode="after")
    def normalize_resolution_gate(self) -> "MissingInfoItem":
        required = self.required_confidence
        if required is None:
            required = _required_confidence(self.risk_level)
            self.required_confidence = required

        has_answer = bool((self.answer or "").strip())
        has_usable_source = self.answer_source != "unknown"
        enough_confidence = self.confidence >= required

        if has_answer and has_usable_source and enough_confidence:
            self.status = "green"
        elif self.status is None:
            self.status = "red" if self.blocks_execution or self.risk_level == "high" else "green"

        if self.status == "green":
            if has_answer and has_usable_source and not enough_confidence:
                self.status = "red"
            elif not (has_answer and has_usable_source) and self.risk_level in {"medium", "high"}:
                self.status = "red"

        self.blocks_execution = self.status == "red"
        return self

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any):
        return super().model_validate(obj, *args, **kwargs)

    @property
    def reason(self) -> str:
        return self.why_needed


PlannerGap = MissingInfoItem


class PlannerAssumption(BaseModel):
    text: str
    category: PlannerContextCategory
    risk_level: PlannerRiskLevel = "low"
    needs_review: bool = False
    reason: str


class TaskPlan(BaseModel):
    objective: str
    steps: list[str] = Field(default_factory=list)
    gaps: list[MissingInfoItem] = Field(default_factory=list)
    assumptions: list[PlannerAssumption] = Field(default_factory=list)
    blocked: bool = False
    user_message: Optional[str] = None


class PlannerContinuation(BaseModel):
    continuation_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    profile_id: UUID
    original_user_task: str
    planner_output_json: str
    blocking_questions_json: str
    status: PlannerContinuationStatus = "pending"
    session_id: Optional[UUID] = None
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    @property
    def blocking_questions(self) -> list[MissingInfoItem]:
        raw = json.loads(self.blocking_questions_json or "[]")
        return [MissingInfoItem.model_validate(item) for item in raw]

    @property
    def task_plan(self) -> TaskPlan:
        return TaskPlan(**json.loads(self.planner_output_json))


class PlannerRuntimeService:
    """Deterministic policy and persistence glue around model-generated plans."""

    def __init__(self, db) -> None:
        self.db = db

    def blocking_gaps(self, plan: TaskPlan) -> list[MissingInfoItem]:
        return self.red_checklist_items(plan)

    def red_checklist_items(self, plan: TaskPlan) -> list[MissingInfoItem]:
        return [gap for gap in plan.gaps if self._gap_is_red(gap)]

    def green_checklist_items(self, plan: TaskPlan) -> list[MissingInfoItem]:
        return [gap for gap in plan.gaps if not self._gap_is_red(gap)]

    def blocking_assumption_gaps(self, plan: TaskPlan) -> list[MissingInfoItem]:
        return [
            _gap_from_planner_assumption(assumption)
            for assumption in plan.assumptions
            if self._assumption_blocks(assumption)
        ]

    def is_blocked(self, plan: TaskPlan) -> bool:
        return bool(plan.blocked or self.blocking_gaps(plan) or self.blocking_assumption_gaps(plan))

    def create_continuation(
        self,
        *,
        profile_id: UUID,
        original_user_task: str,
        plan: TaskPlan,
        session_id: Optional[UUID] = None,
    ) -> PlannerContinuation:
        gaps = self.checklist_gaps(plan)
        continuation = PlannerContinuation(
            profile_id=profile_id,
            original_user_task=original_user_task,
            planner_output_json=_model_to_json(plan),
            blocking_questions_json=json.dumps([_model_to_dict(gap) for gap in gaps]),
            session_id=session_id,
        )
        self.db.insert_planner_continuation(continuation)
        return continuation

    def create_continuation_for_gap(
        self,
        *,
        profile_id: UUID,
        original_user_task: str,
        gap: Any,
        session_id: Optional[UUID] = None,
    ) -> PlannerContinuation:
        if hasattr(gap, "blocking_items"):
            gaps = list(gap.blocking_items)
            summary = getattr(gap, "summary", "")
            return self.create_continuation_for_context_gap(
                profile_id=profile_id,
                original_user_task=original_user_task,
                summary=summary,
                gaps=gaps,
                session_id=session_id,
            )
        gap = MissingInfoItem.model_validate(gap)
        plan = TaskPlan(
            objective=original_user_task,
            gaps=[gap],
            blocked=True,
            user_message=gap.question,
        )
        return self.create_continuation(
            profile_id=profile_id,
            original_user_task=original_user_task,
            plan=plan,
            session_id=session_id,
        )

    def create_continuation_for_context_gap(
        self,
        *,
        profile_id: UUID,
        original_user_task: str,
        summary: str,
        gaps: list[MissingInfoItem],
        session_id: Optional[UUID] = None,
    ) -> PlannerContinuation:
        blocking_gaps = [gap for gap in gaps if gap.blocks_execution] or gaps
        plan = TaskPlan(
            objective=original_user_task,
            gaps=blocking_gaps,
            blocked=True,
            user_message=self.blocking_message_for_gaps(summary, blocking_gaps),
        )
        return self.create_continuation(
            profile_id=profile_id,
            original_user_task=original_user_task,
            plan=plan,
            session_id=session_id,
        )

    def blocking_message(self, plan: TaskPlan) -> str:
        gaps = self.checklist_gaps(plan)
        if not gaps:
            if plan.user_message and plan.user_message.strip():
                return plan.user_message.strip()
            return "I need one detail before I can do this well."
        return self.blocking_message_for_gaps("I need a few details before I can do this well:", gaps)

    def checklist_gaps(self, plan: TaskPlan) -> list[MissingInfoItem]:
        gaps = self.blocking_gaps(plan) or [gap for gap in plan.gaps if gap.blocks_execution]
        gaps = gaps + [
            gap for gap in self.blocking_assumption_gaps(plan) if not _question_already_present(gap.question, gaps)
        ]
        if not plan.blocked or not plan.user_message:
            return gaps

        repaired = list(gaps)
        for question in _numbered_questions_from_text(plan.user_message):
            if _question_already_present(question, repaired):
                continue
            repaired.append(_gap_from_user_message_question(question))
        if not repaired and plan.user_message.strip():
            repaired.append(_gap_from_user_message_question(plan.user_message.strip()))
        return repaired

    def blocking_message_for_gaps(self, summary: str, gaps: list[MissingInfoItem]) -> str:
        if not gaps:
            return summary.strip() or "I need one detail before I can do this well."
        if len(gaps) == 1:
            return gaps[0].question.strip()
        heading = summary.strip() or "I need a few details before I can do this well:"
        lines = [heading]
        for idx, gap in enumerate(gaps, start=1):
            label = gap.label.strip() or _label_from_question(gap.question)
            lines.append(f"\n{idx}. {label}\nQuestion: {gap.question.strip()}")
            if gap.why_needed.strip():
                lines.append(f"Why: {gap.why_needed.strip()}")
        return "\n".join(lines)

    def _gap_blocks(self, gap: MissingInfoItem) -> bool:
        return self._gap_is_red(gap)

    def _gap_is_red(self, gap: MissingInfoItem) -> bool:
        required = gap.required_confidence if gap.required_confidence is not None else _required_confidence(gap.risk_level)
        if gap.status == "red":
            return True
        if gap.status == "green":
            return False
        if gap.answer and gap.answer_source != "unknown" and gap.confidence >= required:
            return False
        if gap.blocks_execution or gap.risk_level == "high":
            return True
        return False

    def _assumption_blocks(self, assumption: PlannerAssumption) -> bool:
        return assumption.risk_level in {"medium", "high"}


def _label_from_question(question: str) -> str:
    text = question.strip().rstrip("?")
    if not text:
        return "Missing information"
    words = text.split()
    return " ".join(words[:4]).capitalize()


def _required_confidence(risk_level: PlannerRiskLevel) -> float:
    if risk_level == "high":
        return 0.9
    if risk_level == "medium":
        return 0.75
    return 0.45


def _numbered_questions_from_text(text: str) -> list[str]:
    matches = re.findall(r"(?:^|\s)\d+[\).]\s*(.*?)(?=(?:\s+\d+[\).]\s*)|$)", text, flags=re.S)
    questions: list[str] = []
    for match in matches:
        question = re.sub(r"\s+", " ", match).strip(" :-")
        if not question:
            continue
        if "?" in question:
            question = question[: question.rfind("?") + 1]
        if not question.endswith("?"):
            question = f"{question}?"
        questions.append(question)
    return questions


def _question_already_present(question: str, gaps: list[MissingInfoItem]) -> bool:
    question_tokens = _meaningful_tokens(question)
    for gap in gaps:
        existing_tokens = _meaningful_tokens(gap.question)
        if not question_tokens or not existing_tokens:
            continue
        overlap = question_tokens & existing_tokens
        if len(overlap) >= min(2, len(question_tokens), len(existing_tokens)):
            return True
    return False


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "be",
        "can",
        "could",
        "do",
        "does",
        "for",
        "is",
        "it",
        "like",
        "me",
        "of",
        "or",
        "the",
        "this",
        "to",
        "what",
        "with",
        "you",
        "your",
    }
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if token not in stopwords}


def _gap_from_user_message_question(question: str) -> MissingInfoItem:
    return MissingInfoItem(
        label=_label_from_question(question),
        category="context_detail",
        question=question,
        why_needed="This detail changes how the task should be completed.",
        blocks_execution=True,
        risk_level="medium",
        answer_source="unknown",
        confidence=0.0,
        status="red",
    )


def _gap_from_planner_assumption(assumption: PlannerAssumption) -> MissingInfoItem:
    return MissingInfoItem(
        label=_label_from_question(assumption.text),
        category="assumption_confirmation",
        question=f"Please confirm or correct this before I continue: {assumption.text}",
        why_needed=assumption.reason or "This assumption materially changes how the task should be completed.",
        blocks_execution=True,
        risk_level=assumption.risk_level,
        answer=assumption.text,
        answer_source="assumption",
        confidence=0.0,
        status="red",
        suggested_ddc_category=assumption.category,
        suggested_review_text=assumption.text,
    )


def _model_to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def _model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
