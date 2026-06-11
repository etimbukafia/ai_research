from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from experiments.drug_repurposing_agent.src.lessons import LessonGenerationError, LessonWorkflow
from experiments.drug_repurposing_agent.src.memory import ExperimentMemory, InMemoryBackend
from experiments.drug_repurposing_agent.src.models import EvaluatorFeedback, FailureType

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


class MockModel:
    def __init__(self, response):
        self.response = response
        self.prompt = ""

    def complete(self, prompt: str):
        self.prompt = prompt
        return self.response


def feedback(*, phase: str = "learning_stream") -> EvaluatorFeedback:
    return EvaluatorFeedback(
        pair_id="disease-drug-pair-021",
        expected_label="weakly_supported",
        observed_label="supported",
        classification_correct=False,
        relationship_extraction_score=1.0,
        citation_correctness_score=1.0,
        failure_types=[FailureType.EVIDENCE_STRENGTH_OVERESTIMATION],
        feedback="Treat target association as indirect evidence, not direct efficacy evidence.",
        phase=phase,
        learning_feedback_allowed=phase == "learning_stream",
    )


def candidate_payload(**overrides):
    payload = {
        "candidate_id": "lesson-021",
        "lesson": "Treat target association as indirect evidence, not direct efficacy evidence.",
        "failure_type": "evidence_strength_overestimation",
        "applicable_evidence_types": ["target_association"],
        "source_task": "disease-drug-pair-021",
        "confidence": 0.9,
        "feedback_basis": "Evaluator identified evidence-strength overestimation.",
        "supersession": {
            "supersedes_lesson_ids": [],
            "superseded_by_lesson_ids": [],
        },
        "scope": "procedural",
    }
    payload.update(overrides)
    return payload


def workflow(response=None) -> LessonWorkflow:
    return LessonWorkflow(MockModel(response or candidate_payload()), now=lambda: NOW)


def generate(service: LessonWorkflow, **kwargs):
    return service.generate_candidate(
        feedback=kwargs.pop("feedback_value", feedback()),
        evidence_types=["target_association"],
        pair_terms=kwargs.pop("pair_terms", ["Disease A", "Drug A", "MONDO_1", "CHEMBL1"]),
        existing_lessons=kwargs.pop("existing_lessons", []),
    )


def test_generates_procedural_candidate_only_from_learning_feedback() -> None:
    service = workflow()
    candidate = generate(service)

    assert candidate is not None
    assert candidate.scope == "procedural"
    assert "Generated output is an unapproved candidate" in service.model.prompt

    with pytest.raises(LessonGenerationError, match="learning-stream"):
        generate(service, feedback_value=feedback(phase="held_out_evaluation"))


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (
            candidate_payload(lesson="Drug A is effective for Disease A."),
            "candidate contains a source-pair term",
        ),
        (
            candidate_payload(lesson="Recommend dosing this treatment."),
            "candidate contains a biomedical claim or treatment recommendation",
        ),
        (
            candidate_payload(failure_type="citation_mismatch"),
            "candidate is not supported by evaluator feedback",
        ),
    ],
)
def test_rejects_and_logs_invalid_candidates(payload, reason) -> None:
    service = workflow(payload)

    assert generate(service) is None
    assert reason in service.rejections[0].reasons
    assert service.rejections[0].candidate_payload == payload


def test_rejects_duplicate_candidate() -> None:
    service = workflow()

    assert generate(
        service,
        existing_lessons=[
            "Treat target association as indirect evidence, not direct efficacy evidence."
        ],
    ) is None
    assert "candidate duplicates an existing validated lesson" in service.rejections[0].reasons


def test_regressions_must_pass_before_approval() -> None:
    service = workflow()
    candidate = generate(service)
    assert candidate is not None
    cases = service.create_regression_cases(candidate)
    results = service.run_regressions(
        candidate,
        cases,
        feedback=feedback(),
        pair_terms=["Disease A", "Drug A"],
    )
    assert all(result.passed for result in results)

    with pytest.raises(LessonGenerationError, match="all required"):
        service.approve(
            candidate,
            feedback=feedback(),
            regression_results=results[:-1],
            approval_record_id="approval-021",
            approved_by="reviewer",
        )

    failed = results[0].model_copy(update={"passed": False, "details": "Failed."})
    with pytest.raises(ValidationError, match="all lesson regression cases must pass"):
        service.approve(
            candidate,
            feedback=feedback(),
            regression_results=[failed, *results[1:]],
            approval_record_id="approval-021",
            approved_by="reviewer",
        )


def test_explicit_approval_is_required_before_validated_memory_write() -> None:
    service = workflow()
    candidate = generate(service)
    assert candidate is not None
    cases = service.create_regression_cases(candidate)
    results = service.run_regressions(
        candidate,
        cases,
        feedback=feedback(),
        pair_terms=["Disease A", "Drug A"],
    )
    backend = InMemoryBackend()
    memory = ExperimentMemory(
        condition="validated_lessons", run_id="run-a", backend=backend
    )

    memory_id = service.approve_and_store(
        candidate,
        feedback=feedback(),
        regression_results=results,
        approval_record_id="approval-021",
        approved_by="reviewer",
        memory=memory,
    )

    assert memory_id == candidate.candidate_id
    record = backend.records[0]
    assert record["metadata"]["provenance"]["approval_record_id"] == "approval-021"
    assert record["metadata"]["validated"] is True
