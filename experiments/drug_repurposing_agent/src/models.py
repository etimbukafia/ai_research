"""Validated schemas for the biomedical evidence-triage experiment."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from experiments.drug_repurposing_agent.src.constants import CLASSIFICATION_LABELS, MEMORY_CONDITIONS

ClassificationLabel = Literal[
    "supported",
    "weakly_supported",
    "unsupported",
    "insufficient_evidence",
]

MemoryCondition = Literal["no_memory", "raw_memory", "validated_lessons"]

EvidenceSource = Literal["Open Targets", "PubMed"]

ExperimentPhase = Literal[
    "initial_evaluation",
    "learning_stream",
    "held_out_evaluation",
    "distribution_shift",
]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and validates assignment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class GoldRelationship(StrictModel):
    """A relationship the evaluator may expect from an assessment."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class GoldRecord(StrictModel):
    """Human-reviewed adjudication for one fixed dataset pair."""

    pair_id: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    split: ExperimentPhase
    gold_label: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_pattern: str = Field(min_length=1)
    expected_relationships: list[GoldRelationship]
    acceptable_source_ids: list[str]
    rationale: str = Field(min_length=1)
    known_contradictions: list[str]
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    adjudication_notes: str = Field(min_length=1)

    @field_validator("reviewed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @field_validator("acceptable_source_ids", "known_contradictions")
    @classmethod
    def require_unique_values(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "gold record list values", allow_empty=True)


class DiseaseDrugPair(StrictModel):
    """One fixed experiment input."""

    pair_id: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    disease_name: str = Field(min_length=1)
    disease_id: str = Field(min_length=1)
    drug_name: str = Field(min_length=1)
    drug_id: str = Field(min_length=1)


class EvidenceItem(StrictModel):
    """One traceable item retrieved from an allowed evidence source."""

    evidence_id: str = Field(min_length=1)
    source: EvidenceSource
    source_identifier: str = Field(min_length=1)
    url: HttpUrl
    title: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    extracted_claim: str = Field(min_length=1)
    retrieval_timestamp: datetime

    @field_validator("retrieval_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieval_timestamp must include a timezone")
        return value


class Relationship(StrictModel):
    """A cited relationship extracted from the available evidence."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def require_unique_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "evidence_ids")


class ClaimCitation(StrictModel):
    """Evidence references supporting one material assessment claim."""

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def require_unique_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "evidence_ids")


class Assessment(StrictModel):
    """The assessor's complete, source-grounded output."""

    pair_id: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    label: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_items: list[EvidenceItem]
    relationships: list[Relationship]
    explanation: str = Field(min_length=1)
    limitations: list[str]
    citations: list[ClaimCitation] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def enforce_frozen_labels(cls, value: str) -> str:
        if value not in CLASSIFICATION_LABELS:
            raise ValueError("label is not part of the frozen experiment contract")
        return value

    @field_validator("limitations")
    @classmethod
    def require_nonempty_limitations(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "limitations")

    @model_validator(mode="after")
    def validate_evidence_references(self) -> Assessment:
        evidence_ids = [item.evidence_id for item in self.evidence_items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_items must have unique evidence_id values")

        known_ids = set(evidence_ids)
        cited_ids = {
            evidence_id
            for citation in self.citations
            for evidence_id in citation.evidence_ids
        }
        relationship_ids = {
            evidence_id
            for relationship in self.relationships
            for evidence_id in relationship.evidence_ids
        }
        unresolved = (cited_ids | relationship_ids) - known_ids
        if unresolved:
            raise ValueError(
                f"assessment references unknown evidence IDs: {sorted(unresolved)}"
            )

        relationship_claims = {relationship.claim for relationship in self.relationships}
        citation_claims = {citation.claim for citation in self.citations}
        if not self.evidence_items:
            if self.label != "insufficient_evidence":
                raise ValueError(
                    "assessments without evidence must use insufficient_evidence"
                )
            if self.relationships or self.citations:
                raise ValueError(
                    "assessments without evidence cannot assert relationships or citations"
                )
            return self

        material_claims = relationship_claims | {self.explanation}
        uncited_claims = material_claims - citation_claims
        if uncited_claims:
            raise ValueError(
                f"material assessment claims require citations: {sorted(uncited_claims)}"
            )
        return self


class FailureType(StrEnum):
    EVIDENCE_STRENGTH_OVERESTIMATION = "evidence_strength_overestimation"
    CITATION_MISMATCH = "citation_mismatch"
    RELATIONSHIP_EXTRACTION_ERROR = "relationship_extraction_error"
    FAILED_ABSTENTION = "failed_abstention"
    CONTRADICTION_IGNORED = "contradiction_ignored"
    RETRIEVAL_FAILURE = "retrieval_failure"
    OTHER = "other"


class EvaluatorFeedback(StrictModel):
    """Pair-level evaluator result used for scoring and learning-stream feedback."""

    pair_id: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    expected_label: ClassificationLabel
    observed_label: ClassificationLabel
    classification_correct: bool
    relationship_extraction_score: float = Field(ge=0.0, le=1.0)
    citation_correctness_score: float = Field(ge=0.0, le=1.0)
    correct_abstention: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    failure_types: list[FailureType] = Field(default_factory=list)
    feedback: str = Field(min_length=1)
    phase: ExperimentPhase | None = None
    learning_feedback_allowed: bool = False
    latency_seconds: float = Field(ge=0.0, default=0.0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    token_cost_usd: float = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def validate_classification_result(self) -> EvaluatorFeedback:
        labels_match = self.expected_label == self.observed_label
        if self.classification_correct != labels_match:
            raise ValueError(
                "classification_correct must match expected_label versus observed_label"
            )
        if self.learning_feedback_allowed and self.phase != "learning_stream":
            raise ValueError(
                "learning feedback is allowed only during the learning stream"
            )
        return self


class LessonProvenance(StrictModel):
    """Audit trail proving why a validated lesson was accepted."""

    evaluator_feedback_id: str = Field(min_length=1)
    approval_record_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: datetime

    @field_validator("approved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approved_at must include a timezone")
        return value


class SupersessionMetadata(StrictModel):
    """Links between accumulated ADD-only memories."""

    supersedes_lesson_ids: list[str] = Field(default_factory=list)
    superseded_by_lesson_ids: list[str] = Field(default_factory=list)

    @field_validator("supersedes_lesson_ids", "superseded_by_lesson_ids")
    @classmethod
    def require_unique_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "supersession lesson IDs", allow_empty=True)


class LessonCandidate(StrictModel):
    """Unapproved procedural lesson proposed from one learning-stream failure."""

    candidate_id: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    failure_type: FailureType
    applicable_evidence_types: list[str] = Field(min_length=1)
    source_task: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    confidence: float = Field(ge=0.0, le=1.0)
    feedback_basis: str = Field(min_length=1)
    supersession: SupersessionMetadata = Field(default_factory=SupersessionMetadata)
    scope: Literal["procedural"] = "procedural"

    @field_validator("applicable_evidence_types")
    @classmethod
    def require_unique_evidence_types(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "applicable_evidence_types")


class LessonRegressionCase(StrictModel):
    """A deterministic rule that an accepted candidate must satisfy."""

    regression_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    check: Literal[
        "procedural_scope",
        "feedback_supported",
        "not_pair_specific",
        "not_treatment_recommendation",
        "not_duplicate",
    ]


class LessonRegressionResult(StrictModel):
    """Outcome of running one lesson regression case."""

    regression_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    passed: bool
    details: str = Field(min_length=1)


class LessonApprovalRecord(StrictModel):
    """Explicit human/evaluator approval after all regressions pass."""

    approval_record_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    evaluator_feedback_id: str = Field(min_length=1)
    approved: Literal[True]
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    regression_results: list[LessonRegressionResult] = Field(min_length=1)

    @field_validator("approved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_passing_regressions(self) -> LessonApprovalRecord:
        if any(result.candidate_id != self.candidate_id for result in self.regression_results):
            raise ValueError("approval regression results must match the candidate")
        if not all(result.passed for result in self.regression_results):
            raise ValueError("all lesson regression cases must pass before approval")
        return self


class LessonRejectionRecord(StrictModel):
    """Audit record for a candidate rejected before validated memory."""

    candidate_id: str = Field(min_length=1)
    source_task: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    rejected_at: datetime
    reasons: list[str] = Field(min_length=1)
    candidate_payload: dict[str, Any]

    @field_validator("rejected_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("rejected_at must include a timezone")
        return value

    @field_validator("reasons")
    @classmethod
    def require_unique_reasons(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "rejection reasons")


class Lesson(StrictModel):
    """An approved procedural lesson eligible for validated memory."""

    lesson_id: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    failure_type: FailureType
    validated: Literal[True]
    applicable_evidence_types: list[str] = Field(min_length=1)
    source_task: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: LessonProvenance
    supersession: SupersessionMetadata
    scope: Literal["procedural"] = "procedural"

    @field_validator("applicable_evidence_types")
    @classmethod
    def require_unique_evidence_types(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "applicable_evidence_types")

    @field_validator("lesson")
    @classmethod
    def reject_biomedical_or_treatment_claims(cls, value: str) -> str:
        normalized = " ".join(value.lower().split())
        prohibited_patterns = (
            r"\b(?:drug|medication|therapy)\s+.+\s+(?:is|was|are|were)\s+"
            r"(?:effective|efficacious|beneficial|safe)\s+for\b",
            r"\b(?:recommend|prescribe|administer|use)\s+.+\s+(?:to treat|for treating)\b",
            r"\bshould\s+(?:be\s+)?(?:prescribed|administered|used)\s+(?:to treat|for)\b",
            r"\b(?:cures?|treats?|prevents?)\s+[\w -]+(?:disease|syndrome|disorder)\b",
        )
        if any(re.search(pattern, normalized) for pattern in prohibited_patterns):
            raise ValueError(
                "validated lessons must not contain disease-specific efficacy claims "
                "or treatment recommendations"
            )
        return value


class RunRecord(StrictModel):
    """Auditable result of processing one pair in one condition and phase."""

    run_id: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    pair: DiseaseDrugPair
    condition: MemoryCondition
    phase: ExperimentPhase
    seed: int
    started_at: datetime
    completed_at: datetime
    assessment: Assessment
    evaluator_feedback: EvaluatorFeedback | None = None
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    written_memory_ids: list[str] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("condition")
    @classmethod
    def enforce_frozen_conditions(cls, value: str) -> str:
        if value not in MEMORY_CONDITIONS:
            raise ValueError("condition is not part of the frozen experiment contract")
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_record_consistency(self) -> RunRecord:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.pair.pair_id != self.assessment.pair_id:
            raise ValueError("pair and assessment pair_id values must match")
        if (
            self.evaluator_feedback is not None
            and self.evaluator_feedback.pair_id != self.pair.pair_id
        ):
            raise ValueError("pair and evaluator_feedback pair_id values must match")
        if self.condition == "no_memory" and (
            self.retrieved_memory_ids or self.written_memory_ids
        ):
            raise ValueError("no_memory runs cannot retrieve or write memories")
        if self.phase != "learning_stream" and self.written_memory_ids:
            raise ValueError("memory writes are allowed only during the learning stream")
        return self


class SessionEventType(StrEnum):
    """Allowed events in an experiment session's append-only log."""

    SESSION_STARTED = "session_started"
    PAIR_STARTED = "pair_started"
    PAIR_COMPLETED = "pair_completed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_WRITTEN = "memory_written"
    HUMAN_APPROVAL_RECORDED = "human_approval_recorded"
    ERROR_RECORDED = "error_recorded"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_COMPLETED = "session_completed"


class SessionEvent(StrictModel):
    """One immutable entry in an experiment session's append-only event log."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    previous_event_id: str | None = None
    event_type: SessionEventType
    timestamp: datetime
    pair_id: str | None = Field(default=None, pattern=r"^disease-drug-pair-\d{3}$")
    run_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session event timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_event_shape(self) -> SessionEvent:
        if self.sequence == 1 and self.previous_event_id is not None:
            raise ValueError("the first event cannot reference a previous event")
        if self.sequence > 1 and self.previous_event_id is None:
            raise ValueError("events after sequence 1 must reference the previous event")
        if self.event_type is SessionEventType.SESSION_STARTED and self.sequence != 1:
            raise ValueError("session_started must be the first event")
        pair_events = {
            SessionEventType.PAIR_STARTED,
            SessionEventType.PAIR_COMPLETED,
        }
        if self.event_type in pair_events and (self.pair_id is None or self.run_id is None):
            raise ValueError("pair events require pair_id and run_id")
        return self


class SessionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentSession(StrictModel):
    """A resumable snapshot over an append-only sequence of pair-level runs."""

    session_id: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    condition: MemoryCondition
    seed: int
    status: SessionStatus
    started_at: datetime
    updated_at: datetime
    next_pair_ordinal: int = Field(ge=1, le=101)
    completed_run_ids: list[str] = Field(default_factory=list)
    last_checkpoint_sequence: int | None = Field(default=None, ge=1)
    events: list[SessionEvent] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("condition")
    @classmethod
    def enforce_frozen_conditions(cls, value: str) -> str:
        if value not in MEMORY_CONDITIONS:
            raise ValueError("condition is not part of the frozen experiment contract")
        return value

    @field_validator("started_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session timestamps must include a timezone")
        return value

    @field_validator("completed_run_ids")
    @classmethod
    def require_unique_run_ids(cls, value: list[str]) -> list[str]:
        return _unique_nonempty(value, "completed_run_ids", allow_empty=True)

    @model_validator(mode="after")
    def validate_session_snapshot(self) -> ExperimentSession:
        if self.updated_at < self.started_at:
            raise ValueError("updated_at must not precede started_at")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("session event IDs must be unique")
        if any(event.session_id != self.session_id for event in self.events):
            raise ValueError("all events must belong to the experiment session")

        for index, event in enumerate(self.events, start=1):
            if event.sequence != index:
                raise ValueError("session event sequences must be contiguous from 1")
            if index == 1:
                if event.event_type is not SessionEventType.SESSION_STARTED:
                    raise ValueError("the first event must be session_started")
            elif event.previous_event_id != self.events[index - 2].event_id:
                raise ValueError("session events must form an unbroken previous-event chain")
            if index > 1 and event.timestamp < self.events[index - 2].timestamp:
                raise ValueError("session event timestamps must be nondecreasing")

        if self.events[0].timestamp != self.started_at:
            raise ValueError("started_at must match the session_started event timestamp")
        if self.updated_at < self.events[-1].timestamp:
            raise ValueError("updated_at must not precede the latest event")

        completed_event_run_ids = [
            event.run_id
            for event in self.events
            if event.event_type is SessionEventType.PAIR_COMPLETED
        ]
        if self.completed_run_ids != completed_event_run_ids:
            raise ValueError(
                "completed_run_ids must match pair_completed events in append order"
            )
        if self.next_pair_ordinal != len(self.completed_run_ids) + 1:
            raise ValueError(
                "next_pair_ordinal must follow the completed atomic run count"
            )

        if self.last_checkpoint_sequence is not None:
            checkpoint = self.events[self.last_checkpoint_sequence - 1]
            if checkpoint.event_type is not SessionEventType.CHECKPOINT_SAVED:
                raise ValueError(
                    "last_checkpoint_sequence must reference a checkpoint_saved event"
                )

        expected_final_event = {
            SessionStatus.PAUSED: SessionEventType.SESSION_PAUSED,
            SessionStatus.COMPLETED: SessionEventType.SESSION_COMPLETED,
            SessionStatus.FAILED: SessionEventType.ERROR_RECORDED,
        }.get(self.status)
        if expected_final_event is not None and self.events[-1].event_type is not expected_final_event:
            raise ValueError(
                f"{self.status.value} sessions must end with {expected_final_event.value}"
            )
        return self


def _unique_nonempty(
    values: list[str],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain empty values")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    return values
