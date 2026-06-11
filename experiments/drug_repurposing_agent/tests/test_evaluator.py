from __future__ import annotations

from datetime import datetime, timezone

from experiments.drug_repurposing_agent.src.evaluator import evaluate_assessment
from experiments.drug_repurposing_agent.src.models import (
    Assessment,
    ClaimCitation,
    EvidenceItem,
    FailureType,
    GoldRecord,
    GoldRelationship,
    Relationship,
)

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)
CLAIM = "IVACAFTOR has direct positive human evidence for cystic fibrosis."


def evidence(*, pmid: str = "37278811", extracted_claim: str = CLAIM) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"pubmed:{pmid}",
        source="PubMed",
        source_identifier=pmid,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        title="Exact-pair controlled human study",
        evidence_type="literature",
        extracted_claim=extracted_claim,
        retrieval_timestamp=NOW,
    )


def gold(*, split: str = "learning_stream", label: str = "supported") -> GoldRecord:
    relationships = (
        [
            GoldRelationship(
                subject="IVACAFTOR",
                predicate="has_direct_positive_human_evidence_for",
                object="cystic fibrosis",
            )
        ]
        if label != "insufficient_evidence"
        else []
    )
    return GoldRecord(
        pair_id="disease-drug-pair-001",
        split=split,
        gold_label=label,
        confidence=0.9,
        evidence_pattern="test",
        expected_relationships=relationships,
        acceptable_source_ids=["OpenTargets:MONDO_0009061", "PMID:37278811"],
        rationale="Evaluator-only rationale.",
        known_contradictions=["No stronger negative human evidence was identified."],
        reviewer="Reviewer",
        reviewed_at=NOW,
        adjudication_notes="Test record.",
    )


def assessment(
    *,
    label: str = "supported",
    item: EvidenceItem | None = None,
    predicate: str = "has_direct_positive_human_evidence_for",
    explanation: str = CLAIM,
) -> Assessment:
    item = item or evidence()
    relationship = Relationship(
        subject="IVACAFTOR",
        predicate=predicate,
        object="cystic fibrosis",
        claim=CLAIM,
        evidence_ids=[item.evidence_id],
    )
    return Assessment(
        pair_id="disease-drug-pair-001",
        label=label,
        confidence=0.9,
        evidence_items=[item],
        relationships=[relationship],
        explanation=explanation,
        limitations=["No stronger negative human evidence was identified."],
        citations=[
            ClaimCitation(claim=CLAIM, evidence_ids=[item.evidence_id]),
            ClaimCitation(claim=explanation, evidence_ids=[item.evidence_id]),
        ],
    )


def test_evaluator_scores_correct_assessment_and_runtime_metrics() -> None:
    result = evaluate_assessment(
        assessment(),
        gold(),
        latency_seconds=1.5,
        input_tokens=100,
        output_tokens=25,
        token_cost_usd=0.001,
    )

    assert result.classification_correct
    assert result.relationship_extraction_score == 1.0
    assert result.citation_correctness_score == 1.0
    assert result.unsupported_claims == []
    assert result.failure_types == []
    assert result.learning_feedback_allowed
    assert result.latency_seconds == 1.5
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.token_cost_usd == 0.001


def test_evaluator_detects_citation_mismatch_and_unsupported_claim() -> None:
    mismatched = evidence(pmid="999", extracted_claim="This article discusses an unrelated pathway.")

    result = evaluate_assessment(assessment(item=mismatched), gold())

    assert result.citation_correctness_score == 0.0
    assert FailureType.CITATION_MISMATCH in result.failure_types
    assert CLAIM in result.unsupported_claims


def test_evaluator_detects_relationship_extraction_error() -> None:
    result = evaluate_assessment(assessment(predicate="treats"), gold())

    assert result.relationship_extraction_score == 0.0
    assert FailureType.RELATIONSHIP_EXTRACTION_ERROR in result.failure_types
    assert result.unsupported_claims == []


def test_evaluator_detects_failed_abstention_and_overestimation() -> None:
    result = evaluate_assessment(assessment(label="supported"), gold(label="insufficient_evidence"))

    assert result.correct_abstention is False
    assert FailureType.FAILED_ABSTENTION in result.failure_types
    assert FailureType.EVIDENCE_STRENGTH_OVERESTIMATION in result.failure_types


def test_evaluator_detects_prohibited_treatment_or_safety_claim() -> None:
    unsafe = "IVACAFTOR should be used to treat cystic fibrosis because it is safe."
    item = evidence(extracted_claim=unsafe)

    result = evaluate_assessment(
        assessment(item=item, explanation=unsafe),
        gold(),
    )

    assert unsafe in result.unsupported_claims


def test_evaluator_detects_ignored_contradiction() -> None:
    payload = assessment().model_dump(mode="json")
    payload["limitations"] = ["This label does not establish treatment suitability."]

    result = evaluate_assessment(Assessment.model_validate(payload), gold())

    assert FailureType.CONTRADICTION_IGNORED in result.failure_types


def test_held_out_feedback_does_not_expose_corrective_answer() -> None:
    result = evaluate_assessment(
        assessment(label="weakly_supported"),
        gold(split="held_out_evaluation"),
    )

    assert not result.learning_feedback_allowed
    assert result.expected_label not in result.feedback
    assert "withheld" in result.feedback


def test_zero_evidence_abstention_records_retrieval_failure() -> None:
    empty = Assessment(
        pair_id="disease-drug-pair-001",
        label="insufficient_evidence",
        confidence=0.9,
        evidence_items=[],
        relationships=[],
        explanation="Retrieval failed materially.",
        limitations=["No evidence was available."],
        citations=[],
    )

    result = evaluate_assessment(empty, gold(label="insufficient_evidence"))

    assert result.correct_abstention is True
    assert FailureType.RETRIEVAL_FAILURE in result.failure_types
