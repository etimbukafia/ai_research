from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from experiments.drug_repurposing_agent.src.memory import (
    ExperimentMemory,
    InMemoryBackend,
    Mem0Backend,
    MemoryPolicyError,
)
from experiments.drug_repurposing_agent.src.models import (
    Assessment,
    DiseaseDrugPair,
    EvaluatorFeedback,
    FailureType,
    Lesson,
    LessonApprovalRecord,
    LessonProvenance,
    LessonRegressionResult,
    RunRecord,
    SupersessionMetadata,
)

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def feedback(*, phase: str = "learning_stream") -> EvaluatorFeedback:
    return EvaluatorFeedback(
        pair_id="disease-drug-pair-021",
        expected_label="supported",
        observed_label="weakly_supported",
        classification_correct=False,
        relationship_extraction_score=1.0,
        citation_correctness_score=1.0,
        failure_types=[FailureType.EVIDENCE_STRENGTH_OVERESTIMATION],
        feedback="Distinguish direct from indirect evidence.",
        phase=phase,
        learning_feedback_allowed=phase == "learning_stream",
    )


def empty_assessment() -> Assessment:
    return Assessment(
        pair_id="disease-drug-pair-021",
        label="insufficient_evidence",
        confidence=0.8,
        evidence_items=[],
        relationships=[],
        explanation="No evidence was available.",
        limitations=["Retrieval failed."],
        citations=[],
    )


def raw_run(*, phase: str = "learning_stream", run_id: str = "run-a") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        contract_version="1.0.0",
        pair=DiseaseDrugPair(
            pair_id="disease-drug-pair-021",
            disease_name="Disease A",
            disease_id="MONDO_1",
            drug_name="Drug A",
            drug_id="CHEMBL1",
        ),
        condition="raw_memory",
        phase=phase,
        seed=42,
        started_at=NOW,
        completed_at=NOW,
        assessment=empty_assessment(),
        evaluator_feedback=feedback(phase=phase),
        prompt_versions={"assessor": "assessor-v1.0.0"},
        input_tokens=10,
        output_tokens=5,
        latency_seconds=0.1,
    )


def lesson() -> Lesson:
    return Lesson(
        lesson_id="lesson-001",
        lesson="Treat target association as indirect evidence.",
        failure_type=FailureType.EVIDENCE_STRENGTH_OVERESTIMATION,
        validated=True,
        applicable_evidence_types=["target_association"],
        source_task="disease-drug-pair-021",
        confidence=0.9,
        provenance=LessonProvenance(
            evaluator_feedback_id="feedback-021",
            approval_record_id="approval-001",
            approved_by="reviewer",
            approved_at=NOW,
        ),
        supersession=SupersessionMetadata(
            supersedes_lesson_ids=["lesson-old"],
        ),
    )


def approval() -> LessonApprovalRecord:
    return LessonApprovalRecord(
        approval_record_id="approval-001",
        candidate_id="lesson-001",
        evaluator_feedback_id="feedback-021",
        approved=True,
        approved_by="reviewer",
        approved_at=NOW,
        regression_results=[
            LessonRegressionResult(
                regression_id="regression-001",
                candidate_id="lesson-001",
                passed=True,
                details="Passed.",
            )
        ],
    )


def test_no_memory_never_reads_or_writes() -> None:
    backend = InMemoryBackend()
    memory = ExperimentMemory(condition="no_memory", run_id="run-a", backend=backend)

    assert memory.retrieve(query="anything", evidence_types=["literature"]) == []
    with pytest.raises(MemoryPolicyError):
        memory.store_raw_run(raw_run())
    with pytest.raises(MemoryPolicyError):
        memory.store_validated_lesson(
            lesson(), evaluator_feedback=feedback(), approval=approval()
        )
    assert backend.calls == []


def test_raw_memory_stores_and_retrieves_completed_trajectory() -> None:
    backend = InMemoryBackend()
    memory = ExperimentMemory(condition="raw_memory", run_id="run-a", backend=backend)

    memory_id = memory.store_raw_run(raw_run())
    entries = memory.retrieve(query="indirect evidence", evidence_types=[])

    assert entries[0].memory_id == memory_id
    assert entries[0].kind == "raw_trajectory"
    assert "evaluator_feedback" in entries[0].content


def test_memory_is_isolated_by_condition_and_run() -> None:
    backend = InMemoryBackend()
    writer = ExperimentMemory(condition="raw_memory", run_id="run-a", backend=backend)
    writer.store_raw_run(raw_run())

    assert ExperimentMemory(
        condition="raw_memory", run_id="run-b", backend=backend
    ).retrieve(query="evidence", evidence_types=[]) == []
    assert ExperimentMemory(
        condition="validated_lessons", run_id="run-a", backend=backend
    ).retrieve(query="evidence", evidence_types=["target_association"]) == []


def test_validated_lessons_preserve_metadata_and_filter_evidence_type() -> None:
    backend = InMemoryBackend()
    memory = ExperimentMemory(
        condition="validated_lessons", run_id="run-a", backend=backend
    )
    memory.store_validated_lesson(
        lesson(), evaluator_feedback=feedback(), approval=approval()
    )

    assert memory.retrieve(query="indirect", evidence_types=["literature"]) == []
    entries = memory.retrieve(query="indirect", evidence_types=["target_association"])

    assert entries[0].metadata["validated"] is True
    assert entries[0].metadata["confidence"] == 0.9
    assert entries[0].metadata["provenance"]["approval_record_id"] == "approval-001"
    assert entries[0].metadata["supersession"]["supersedes_lesson_ids"] == ["lesson-old"]


def test_rejected_lesson_cannot_be_stored_or_retrieved() -> None:
    payload = lesson().model_dump(mode="json")
    payload["validated"] = False

    with pytest.raises(ValidationError):
        Lesson.model_validate(payload)

    backend = InMemoryBackend()
    namespace = "drug-repurposing-agent:validated_lessons:run-a"
    backend.add(
        content="Unvalidated candidate.",
        namespace=namespace,
        metadata={
            "memory_id": "rejected-001",
            "namespace": namespace,
            "condition": "validated_lessons",
            "run_id": "run-a",
            "kind": "validated_lesson",
            "source_task": "disease-drug-pair-021",
            "applicable_evidence_types": ["target_association"],
            "validated": False,
        },
    )
    memory = ExperimentMemory(
        condition="validated_lessons", run_id="run-a", backend=backend
    )
    assert memory.retrieve(
        query="candidate", evidence_types=["target_association"]
    ) == []


@pytest.mark.parametrize("phase", ["initial_evaluation", "held_out_evaluation", "distribution_shift"])
def test_raw_memory_rejects_non_learning_feedback(phase: str) -> None:
    memory = ExperimentMemory(
        condition="raw_memory", run_id="run-a", backend=InMemoryBackend()
    )

    with pytest.raises(MemoryPolicyError, match="learning-stream"):
        memory.store_raw_run(raw_run(phase=phase))


def test_validated_memory_rejects_held_out_feedback() -> None:
    memory = ExperimentMemory(
        condition="validated_lessons", run_id="run-a", backend=InMemoryBackend()
    )

    with pytest.raises(MemoryPolicyError, match="learning-stream"):
        memory.store_validated_lesson(
            lesson(),
            evaluator_feedback=feedback(phase="held_out_evaluation"),
            approval=approval(),
        )


def test_mem0_adapter_uses_condition_run_namespace() -> None:
    class FakeMem0:
        def __init__(self):
            self.add_user_id = None
            self.search_user_id = None

        def add(self, content, *, user_id, metadata):
            self.add_user_id = user_id
            return {"results": [{"id": metadata["memory_id"]}]}

        def search(self, *, query, user_id, filters, limit):
            self.search_user_id = user_id
            return {"results": []}

    client = FakeMem0()
    backend = Mem0Backend(client)
    memory = ExperimentMemory(condition="raw_memory", run_id="run-a", backend=backend)

    memory.store_raw_run(raw_run())
    memory.retrieve(query="evidence", evidence_types=[])

    assert client.add_user_id == memory.namespace
    assert client.search_user_id == memory.namespace
