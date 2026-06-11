from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from experiments.drug_repurposing_agent.src.models import (
    Assessment,
    ClaimCitation,
    DiseaseDrugPair,
    EvidenceItem,
    EvaluatorFeedback,
    ExperimentSession,
    FailureType,
    Lesson,
    LessonProvenance,
    Relationship,
    RunRecord,
    SessionEvent,
    SessionEventType,
    SessionStatus,
    SupersessionMetadata,
)

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)
PAIR_ID = "disease-drug-pair-001"


def evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="pubmed:123",
        source="PubMed",
        source_identifier="123",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
        title="Example study",
        evidence_type="controlled_human_study",
        extracted_claim="The exact pair was evaluated.",
        retrieval_timestamp=NOW,
    )


def assessment() -> Assessment:
    relationship_claim = "Drug A was evaluated in Disease A."
    return Assessment(
        pair_id=PAIR_ID,
        label="supported",
        confidence=0.9,
        evidence_items=[evidence()],
        relationships=[
            Relationship(
                subject="Drug A",
                predicate="evaluated_in",
                object="Disease A",
                claim=relationship_claim,
                evidence_ids=["pubmed:123"],
            )
        ],
        explanation="Direct controlled human evidence supports investigation.",
        limitations=["This label does not establish safety or treatment suitability."],
        citations=[
            ClaimCitation(
                claim=relationship_claim,
                evidence_ids=["pubmed:123"],
            ),
            ClaimCitation(
                claim="Direct controlled human evidence supports investigation.",
                evidence_ids=["pubmed:123"],
            ),
        ],
    )


def provenance() -> LessonProvenance:
    return LessonProvenance(
        evaluator_feedback_id="feedback-001",
        approval_record_id="approval-001",
        approved_by="evaluator-1",
        approved_at=NOW,
    )


def session_event(
    sequence: int,
    event_type: SessionEventType,
    *,
    previous_event_id: str | None = None,
    pair_id: str | None = None,
    run_id: str | None = None,
) -> SessionEvent:
    return SessionEvent(
        event_id=f"event-{sequence:03d}",
        session_id="session-001",
        sequence=sequence,
        previous_event_id=previous_event_id,
        event_type=event_type,
        timestamp=NOW + timedelta(seconds=sequence - 1),
        pair_id=pair_id,
        run_id=run_id,
    )


def test_assessment_rejects_invalid_label() -> None:
    payload = assessment().model_dump(mode="json")
    payload["label"] = "promising"

    with pytest.raises(ValidationError, match="label"):
        Assessment.model_validate(payload)


def test_assessment_rejects_uncited_relationship_claim() -> None:
    payload = assessment().model_dump(mode="json")
    payload["citations"] = [payload["citations"][1]]

    with pytest.raises(ValidationError, match="material assessment claims require citations"):
        Assessment.model_validate(payload)


def test_assessment_rejects_uncited_explanation() -> None:
    payload = assessment().model_dump(mode="json")
    payload["citations"] = [payload["citations"][0]]

    with pytest.raises(ValidationError, match="material assessment claims require citations"):
        Assessment.model_validate(payload)


def test_assessment_rejects_unknown_evidence_reference() -> None:
    payload = assessment().model_dump(mode="json")
    payload["citations"][0]["evidence_ids"] = ["pubmed:missing"]

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        Assessment.model_validate(payload)


def test_evidence_requires_timezone_aware_retrieval_timestamp() -> None:
    payload = evidence().model_dump()
    payload["retrieval_timestamp"] = datetime(2026, 6, 11)

    with pytest.raises(ValidationError, match="timezone"):
        EvidenceItem.model_validate(payload)


def test_lesson_rejects_unvalidated_memory() -> None:
    with pytest.raises(ValidationError, match="validated"):
        Lesson(
            lesson_id="lesson-001",
            lesson="Do not classify target association as direct efficacy evidence.",
            failure_type=FailureType.EVIDENCE_STRENGTH_OVERESTIMATION,
            validated=False,
            applicable_evidence_types=["target_association"],
            source_task=PAIR_ID,
            confidence=0.9,
            provenance=provenance(),
            supersession=SupersessionMetadata(),
        )


def test_lesson_rejects_disease_specific_efficacy_claim() -> None:
    with pytest.raises(ValidationError, match="disease-specific efficacy"):
        Lesson(
            lesson_id="lesson-001",
            lesson="Drug A is effective for Disease A.",
            failure_type=FailureType.OTHER,
            validated=True,
            applicable_evidence_types=["controlled_human_study"],
            source_task=PAIR_ID,
            confidence=0.9,
            provenance=provenance(),
            supersession=SupersessionMetadata(),
        )


def test_lesson_accepts_procedural_rule() -> None:
    lesson = Lesson(
        lesson_id="lesson-001",
        lesson="Do not classify target association as direct efficacy evidence.",
        failure_type=FailureType.EVIDENCE_STRENGTH_OVERESTIMATION,
        validated=True,
        applicable_evidence_types=["target_association"],
        source_task=PAIR_ID,
        confidence=0.9,
        provenance=provenance(),
        supersession=SupersessionMetadata(),
    )

    assert lesson.scope == "procedural"


def test_evaluator_feedback_rejects_inconsistent_correctness() -> None:
    with pytest.raises(ValidationError, match="classification_correct"):
        EvaluatorFeedback(
            pair_id=PAIR_ID,
            expected_label="supported",
            observed_label="unsupported",
            classification_correct=True,
            relationship_extraction_score=0.5,
            citation_correctness_score=1.0,
            unsupported_claims=[],
            failure_types=[FailureType.OTHER],
            feedback="The observed label does not match the gold label.",
        )


def test_public_schemas_generate_json_schema() -> None:
    schemas = (
        DiseaseDrugPair,
        EvidenceItem,
        Relationship,
        Assessment,
        EvaluatorFeedback,
        Lesson,
        RunRecord,
        SessionEvent,
        ExperimentSession,
    )

    assert all(schema.model_json_schema()["type"] == "object" for schema in schemas)


def test_run_record_rejects_memory_activity_for_no_memory_condition() -> None:
    with pytest.raises(ValidationError, match="no_memory"):
        RunRecord(
            run_id="run-001",
            contract_version="1.0.0",
            pair=DiseaseDrugPair(
                pair_id=PAIR_ID,
                disease_name="Disease A",
                disease_id="EFO_001",
                drug_name="Drug A",
                drug_id="CHEMBL001",
            ),
            condition="no_memory",
            phase="initial_evaluation",
            seed=42,
            started_at=NOW,
            completed_at=NOW,
            assessment=assessment(),
            retrieved_memory_ids=["memory-001"],
            prompt_versions={"assessor": "assessor-v1.0.0"},
            input_tokens=100,
            output_tokens=50,
            latency_seconds=1.0,
        )


def test_session_event_is_immutable() -> None:
    event = session_event(1, SessionEventType.SESSION_STARTED)

    with pytest.raises(ValidationError, match="frozen"):
        event.sequence = 2


def test_session_event_requires_previous_event_after_first() -> None:
    with pytest.raises(ValidationError, match="previous event"):
        session_event(2, SessionEventType.CHECKPOINT_SAVED)


def test_experiment_session_accepts_chained_atomic_run_references() -> None:
    events = [
        session_event(1, SessionEventType.SESSION_STARTED),
        session_event(
            2,
            SessionEventType.PAIR_STARTED,
            previous_event_id="event-001",
            pair_id=PAIR_ID,
            run_id="run-001",
        ),
        session_event(
            3,
            SessionEventType.PAIR_COMPLETED,
            previous_event_id="event-002",
            pair_id=PAIR_ID,
            run_id="run-001",
        ),
        session_event(
            4,
            SessionEventType.CHECKPOINT_SAVED,
            previous_event_id="event-003",
        ),
        session_event(
            5,
            SessionEventType.SESSION_PAUSED,
            previous_event_id="event-004",
        ),
    ]

    session = ExperimentSession(
        session_id="session-001",
        contract_version="1.0.0",
        condition="validated_lessons",
        seed=42,
        status=SessionStatus.PAUSED,
        started_at=NOW,
        updated_at=events[-1].timestamp,
        next_pair_ordinal=2,
        completed_run_ids=["run-001"],
        last_checkpoint_sequence=4,
        events=events,
    )

    assert session.completed_run_ids == ["run-001"]


def test_experiment_session_rejects_broken_event_chain() -> None:
    events = [
        session_event(1, SessionEventType.SESSION_STARTED),
        session_event(
            2,
            SessionEventType.PAIR_STARTED,
            previous_event_id="wrong-event",
            pair_id=PAIR_ID,
            run_id="run-001",
        ),
    ]

    with pytest.raises(ValidationError, match="unbroken previous-event chain"):
        ExperimentSession(
            session_id="session-001",
            contract_version="1.0.0",
            condition="no_memory",
            seed=42,
            status=SessionStatus.ACTIVE,
            started_at=NOW,
            updated_at=events[-1].timestamp,
            next_pair_ordinal=1,
            events=events,
        )


def test_experiment_session_rejects_completed_run_without_event() -> None:
    events = [session_event(1, SessionEventType.SESSION_STARTED)]

    with pytest.raises(ValidationError, match="completed_run_ids"):
        ExperimentSession(
            session_id="session-001",
            contract_version="1.0.0",
            condition="no_memory",
            seed=42,
            status=SessionStatus.ACTIVE,
            started_at=NOW,
            updated_at=NOW,
            next_pair_ordinal=1,
            completed_run_ids=["run-001"],
            events=events,
        )


def test_experiment_session_rejects_skipped_next_pair_ordinal() -> None:
    events = [session_event(1, SessionEventType.SESSION_STARTED)]

    with pytest.raises(ValidationError, match="next_pair_ordinal"):
        ExperimentSession(
            session_id="session-001",
            contract_version="1.0.0",
            condition="no_memory",
            seed=42,
            status=SessionStatus.ACTIVE,
            started_at=NOW,
            updated_at=NOW,
            next_pair_ordinal=2,
            events=events,
        )
