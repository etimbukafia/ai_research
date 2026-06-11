"""Procedural lesson generation, validation, regression, and approval."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pydantic import ValidationError

from experiments.drug_repurposing_agent.src.memory import ExperimentMemory
from experiments.drug_repurposing_agent.src.models import (
    EvaluatorFeedback,
    Lesson,
    LessonApprovalRecord,
    LessonCandidate,
    LessonProvenance,
    LessonRegressionCase,
    LessonRegressionResult,
    LessonRejectionRecord,
)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "lesson_generator.md"
_BIOMEDICAL_OR_TREATMENT = (
    r"\b(?:effective|efficacious|beneficial|harmful|safe|unsafe)\b",
    r"\b(?:recommend|prescribe|administer|dose|dosing|cures?|prevents?)\b",
    r"\b(?:use|used|using)\s+.+\s+(?:to treat|for treating)\b",
)


class LessonGenerationError(ValueError):
    """Raised when lesson generation or approval violates the contract."""


class LessonModel(Protocol):
    def complete(self, prompt: str) -> str | dict[str, Any] | LessonCandidate:
        """Return one structured lesson candidate."""


class LessonWorkflow:
    """Generate candidates, log rejections, run regressions, and approve lessons."""

    def __init__(
        self,
        model: LessonModel,
        *,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.model = model
        self.instructions = prompt_path.read_text(encoding="utf-8").strip()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.rejections: list[LessonRejectionRecord] = []

    def generate_candidate(
        self,
        *,
        feedback: EvaluatorFeedback,
        evidence_types: Sequence[str],
        pair_terms: Sequence[str],
        existing_lessons: Sequence[str] = (),
    ) -> LessonCandidate | None:
        """Generate and deterministically validate one learning-stream candidate."""

        if feedback.phase != "learning_stream" or not feedback.learning_feedback_allowed:
            raise LessonGenerationError(
                "lesson candidates may be generated only from learning-stream feedback"
            )
        payload = {
            "lesson_candidate_json_schema": LessonCandidate.model_json_schema(),
            "feedback": feedback.model_dump(mode="json"),
            "applicable_evidence_types": list(evidence_types),
            "forbidden_pair_terms": list(pair_terms),
            "existing_validated_lessons": list(existing_lessons),
        }
        raw = self.model.complete(
            f"{self.instructions}\n\n## Runtime Input\n```json\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}\n```\n"
        )
        candidate_payload = _payload(raw)
        candidate_id = str(candidate_payload.get("candidate_id") or "invalid-candidate")
        reasons: list[str] = []
        try:
            candidate = LessonCandidate.model_validate(candidate_payload)
        except ValidationError as exc:
            reasons.append(f"malformed candidate: {exc}")
            candidate = None
        if candidate is not None:
            reasons.extend(
                validate_candidate(
                    candidate,
                    feedback=feedback,
                    pair_terms=pair_terms,
                    existing_lessons=existing_lessons,
                )
            )
        if reasons:
            self.rejections.append(
                LessonRejectionRecord(
                    candidate_id=candidate_id,
                    source_task=feedback.pair_id,
                    rejected_at=self.now(),
                    reasons=list(dict.fromkeys(reasons)),
                    candidate_payload=candidate_payload,
                )
            )
            return None
        return candidate

    def create_regression_cases(
        self,
        candidate: LessonCandidate,
    ) -> list[LessonRegressionCase]:
        return [
            LessonRegressionCase(
                regression_id=f"{candidate.candidate_id}:{check}",
                candidate_id=candidate.candidate_id,
                description=description,
                check=check,
            )
            for check, description in (
                ("procedural_scope", "Candidate remains a general procedure."),
                ("feedback_supported", "Candidate is supported by evaluator feedback."),
                ("not_pair_specific", "Candidate excludes source-pair terms."),
                ("not_treatment_recommendation", "Candidate excludes biomedical and treatment claims."),
                ("not_duplicate", "Candidate is not a duplicate validated lesson."),
            )
        ]

    def run_regressions(
        self,
        candidate: LessonCandidate,
        cases: Sequence[LessonRegressionCase],
        *,
        feedback: EvaluatorFeedback,
        pair_terms: Sequence[str],
        existing_lessons: Sequence[str] = (),
    ) -> list[LessonRegressionResult]:
        reasons = set(
            validate_candidate(
                candidate,
                feedback=feedback,
                pair_terms=pair_terms,
                existing_lessons=existing_lessons,
            )
        )
        mapping = {
            "procedural_scope": "candidate is not procedural",
            "feedback_supported": "candidate is not supported by evaluator feedback",
            "not_pair_specific": "candidate contains a source-pair term",
            "not_treatment_recommendation": "candidate contains a biomedical claim or treatment recommendation",
            "not_duplicate": "candidate duplicates an existing validated lesson",
        }
        return [
            LessonRegressionResult(
                regression_id=case.regression_id,
                candidate_id=candidate.candidate_id,
                passed=mapping[case.check] not in reasons,
                details=(
                    "Passed deterministic regression."
                    if mapping[case.check] not in reasons
                    else mapping[case.check]
                ),
            )
            for case in cases
        ]

    def approve(
        self,
        candidate: LessonCandidate,
        *,
        feedback: EvaluatorFeedback,
        regression_results: Sequence[LessonRegressionResult],
        approval_record_id: str,
        approved_by: str,
    ) -> tuple[Lesson, LessonApprovalRecord]:
        """Create an explicit approval record and its validated Lesson."""

        if feedback.pair_id != candidate.source_task:
            raise LessonGenerationError("candidate and evaluator feedback must match")
        expected_regression_ids = {
            case.regression_id for case in self.create_regression_cases(candidate)
        }
        observed_regression_ids = {
            result.regression_id for result in regression_results
        }
        if observed_regression_ids != expected_regression_ids:
            raise LessonGenerationError(
                "all required lesson regression cases must run before approval"
            )
        approval = LessonApprovalRecord(
            approval_record_id=approval_record_id,
            candidate_id=candidate.candidate_id,
            evaluator_feedback_id=f"feedback:{feedback.pair_id}",
            approved=True,
            approved_by=approved_by,
            approved_at=self.now(),
            regression_results=list(regression_results),
        )
        lesson = Lesson(
            lesson_id=candidate.candidate_id,
            lesson=candidate.lesson,
            failure_type=candidate.failure_type,
            validated=True,
            applicable_evidence_types=candidate.applicable_evidence_types,
            source_task=candidate.source_task,
            confidence=candidate.confidence,
            provenance=LessonProvenance(
                evaluator_feedback_id=approval.evaluator_feedback_id,
                approval_record_id=approval.approval_record_id,
                approved_by=approval.approved_by,
                approved_at=approval.approved_at,
            ),
            supersession=candidate.supersession,
        )
        return lesson, approval

    def approve_and_store(
        self,
        candidate: LessonCandidate,
        *,
        feedback: EvaluatorFeedback,
        regression_results: Sequence[LessonRegressionResult],
        approval_record_id: str,
        approved_by: str,
        memory: ExperimentMemory,
    ) -> str:
        lesson, approval = self.approve(
            candidate,
            feedback=feedback,
            regression_results=regression_results,
            approval_record_id=approval_record_id,
            approved_by=approved_by,
        )
        return memory.store_validated_lesson(
            lesson,
            evaluator_feedback=feedback,
            approval=approval,
        )


def validate_candidate(
    candidate: LessonCandidate,
    *,
    feedback: EvaluatorFeedback,
    pair_terms: Sequence[str],
    existing_lessons: Sequence[str],
) -> list[str]:
    """Return deterministic rejection reasons for an unapproved candidate."""

    reasons: list[str] = []
    normalized = _normalize(candidate.lesson)
    if candidate.scope != "procedural":
        reasons.append("candidate is not procedural")
    if candidate.source_task != feedback.pair_id or candidate.failure_type not in feedback.failure_types:
        reasons.append("candidate is not supported by evaluator feedback")
    if not set(_tokens(candidate.lesson)) & set(_tokens(feedback.feedback)):
        reasons.append("candidate is not supported by evaluator feedback")
    if any(_normalize(term) in normalized for term in pair_terms if term.strip()):
        reasons.append("candidate contains a source-pair term")
    if any(re.search(pattern, candidate.lesson, re.IGNORECASE) for pattern in _BIOMEDICAL_OR_TREATMENT):
        reasons.append("candidate contains a biomedical claim or treatment recommendation")
    if any(_similar(normalized, _normalize(existing)) >= 0.8 for existing in existing_lessons):
        reasons.append("candidate duplicates an existing validated lesson")
    return list(dict.fromkeys(reasons))


def _payload(raw: str | dict[str, Any] | LessonCandidate) -> dict[str, Any]:
    if isinstance(raw, LessonCandidate):
        return raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LessonGenerationError(f"model returned invalid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise LessonGenerationError("model must return one LessonCandidate JSON object")


def _similar(left: str, right: str) -> float:
    left_tokens, right_tokens = set(_tokens(left)), set(_tokens(right))
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 1.0


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))
