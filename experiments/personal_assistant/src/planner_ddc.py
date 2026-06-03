from __future__ import annotations

import re
from typing import Optional
from uuid import UUID

from experiments.personal_assistant.src.ddc import DDCGapProposal, DDCReviewService
from experiments.personal_assistant.src.planning import PlannerAssumption, PlannerContinuation


class PlannerDDCBridge:
    """Convert planner runtime context into DDC review items."""

    def __init__(self, ddc_review_service: DDCReviewService) -> None:
        self.ddc_review_service = ddc_review_service

    def create_review_items_for_assumptions(
        self,
        *,
        profile_id: UUID,
        source_task: str,
        assumptions: list[PlannerAssumption],
        session_id: Optional[UUID] = None,
    ):
        proposals = [
            DDCGapProposal(
                category=assumption.category,
                risk=assumption.risk_level,
                missing_context=assumption.reason,
                proposed_memory=assumption.text,
                reason="Planner used this runtime assumption and marked it durable enough for review.",
            )
            for assumption in assumptions
            if assumption.needs_review and assumption.text.strip()
        ]
        if not proposals:
            return []
        return self.ddc_review_service.create_review_items(
            profile_id=profile_id,
            source_task=source_task,
            proposals=proposals,
            session_id=session_id,
        )

    def create_review_items_for_continuation_answer(
        self,
        *,
        profile_id: UUID,
        continuation: PlannerContinuation,
        user_answer: str,
        session_id: Optional[UUID] = None,
    ):
        proposals = [
            DDCGapProposal(
                category=gap.suggested_ddc_category or "project",
                risk=gap.risk_level,
                missing_context=gap.question,
                proposed_memory=(
                    "Adding reviewed context from a paused task answer: "
                    f"{_label(gap.label, gap.category)} = "
                    f"{_answer_for_gap(gap.question, user_answer)}"
                ),
                reason="You answered a blocking planner gap; review before making it durable context.",
            )
            for gap in continuation.blocking_questions
            if _answer_for_gap(gap.question, user_answer)
        ]
        return self.ddc_review_service.create_review_items(
            profile_id=profile_id,
            source_task=continuation_source_summary(continuation),
            proposals=proposals,
            session_id=session_id,
        )


def _second_person(text: str) -> str:
    replacements = {
        "The user's": "Your",
        "the user's": "your",
        "The user": "You",
        "the user": "you",
        "User's": "Your",
        "user's": "your",
        "User": "You",
        "user": "you",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def continuation_answer_context(continuation: PlannerContinuation, user_answer: str) -> str:
    lines = [
        f"Task summary: {continuation_source_summary(continuation)}",
        f"Planner objective: {continuation.task_plan.objective}",
        "",
        "Blocking checklist answers:",
    ]
    for gap in continuation.blocking_questions:
        answer = _answer_for_gap(gap.question, user_answer)
        if not answer:
            continue
        lines.extend(
            [
                f"- Label: {gap.label or gap.category}",
                f"  Question: {gap.question}",
                f"  Answer: {answer}",
                f"  Why needed: {gap.why_needed}",
                f"  Suggested category: {gap.suggested_ddc_category or 'none'}",
                f"  Risk: {gap.risk_level}",
            ]
        )
    return "\n".join(lines)


def continuation_source_summary(continuation: PlannerContinuation) -> str:
    objective = re.sub(r"\s+", " ", continuation.task_plan.objective).strip()
    if not objective:
        objective = "Paused personal assistant task"
    return f"Task: {objective}; Domain: General"


def _answers_by_question(text: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    pattern = re.compile(
        r"(?:^|\n)\s*\d+\.\s*Question:\s*(?P<question>.*?)\s*\n\s*Answer:\s*(?P<answer>.*?)(?=\n\s*\d+\.\s*Question:|\n\s*Use these answers|\Z)",
        flags=re.S,
    )
    for match in pattern.finditer(text):
        question = _normalize_question(match.group("question"))
        answer = re.sub(r"\s+", " ", match.group("answer")).strip()
        if question and answer:
            answers[question] = answer
    return answers


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().rstrip("?").lower()


def _label(label: str, category: str) -> str:
    text = label.strip() or category.replace("_", " ").strip()
    return text[:1].lower() + text[1:] if text else "the missing detail"


def _answer_for_gap(question: str, user_answer: str) -> str:
    answers_by_question = _answers_by_question(user_answer)
    return answers_by_question.get(_normalize_question(question), user_answer.strip()).strip()
