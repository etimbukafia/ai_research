"""Deterministic evaluator for pair-level evidence-triage assessments."""

from __future__ import annotations

import re
from collections.abc import Iterable

from experiments.drug_repurposing_agent.src.models import (
    Assessment,
    EvaluatorFeedback,
    FailureType,
    GoldRecord,
)

_LABEL_STRENGTH = {
    "insufficient_evidence": 0,
    "unsupported": 1,
    "weakly_supported": 2,
    "supported": 3,
}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with",
}
_PROHIBITED_CLAIM_PATTERNS = (
    r"\b(?:recommend|prescribe|administer)\b",
    r"\b(?:safe|safety|dose|dosing)\b",
    r"\b(?:should|must)\s+(?:be\s+)?used\s+(?:for|to treat)\b",
)


def evaluate_assessment(
    assessment: Assessment,
    gold: GoldRecord,
    *,
    latency_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    token_cost_usd: float = 0.0,
) -> EvaluatorFeedback:
    """Score one assessment against evaluator-only gold truth."""

    if assessment.pair_id != gold.pair_id:
        raise ValueError("assessment and gold pair_id values must match")

    classification_correct = assessment.label == gold.gold_label
    relationship_score, _unexpected_relationship_claims = _score_relationships(
        assessment, gold
    )
    citation_score, citation_mismatches = _score_citations(assessment, gold)
    unsupported_claims = _unique(
        citation_mismatches
        + _prohibited_claims(assessment)
    )

    correct_abstention = (
        assessment.label == "insufficient_evidence"
        if gold.gold_label == "insufficient_evidence"
        else None
    )
    failure_types = _failure_types(
        assessment=assessment,
        gold=gold,
        classification_correct=classification_correct,
        relationship_score=relationship_score,
        citation_score=citation_score,
    )
    feedback = _feedback_text(
        assessment=assessment,
        gold=gold,
        classification_correct=classification_correct,
        relationship_score=relationship_score,
        citation_score=citation_score,
        unsupported_claims=unsupported_claims,
        failure_types=failure_types,
    )
    return EvaluatorFeedback(
        pair_id=assessment.pair_id,
        expected_label=gold.gold_label,
        observed_label=assessment.label,
        classification_correct=classification_correct,
        relationship_extraction_score=relationship_score,
        citation_correctness_score=citation_score,
        correct_abstention=correct_abstention,
        unsupported_claims=unsupported_claims,
        failure_types=failure_types,
        feedback=feedback,
        phase=gold.split,
        learning_feedback_allowed=gold.split == "learning_stream",
        latency_seconds=latency_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_cost_usd=token_cost_usd,
    )


def _score_relationships(
    assessment: Assessment, gold: GoldRecord
) -> tuple[float, list[str]]:
    expected = {
        _relationship_key(item.subject, item.predicate, item.object)
        for item in gold.expected_relationships
    }
    observed = {
        _relationship_key(item.subject, item.predicate, item.object)
        for item in assessment.relationships
    }
    if not expected:
        score = 1.0 if not observed else 0.0
    else:
        true_positive = len(expected & observed)
        precision = true_positive / len(observed) if observed else 0.0
        recall = true_positive / len(expected)
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    unexpected = [
        relationship.claim
        for relationship in assessment.relationships
        if _relationship_key(
            relationship.subject, relationship.predicate, relationship.object
        )
        not in expected
    ]
    return round(score, 6), unexpected


def _score_citations(
    assessment: Assessment, gold: GoldRecord
) -> tuple[float, list[str]]:
    evidence_by_id = {item.evidence_id: item for item in assessment.evidence_items}
    results: list[bool] = []
    mismatches: list[str] = []
    for citation in assessment.citations:
        valid = all(
            _source_is_acceptable(evidence_by_id[evidence_id], gold)
            and _claim_is_supported(citation.claim, evidence_by_id[evidence_id].extracted_claim)
            for evidence_id in citation.evidence_ids
        )
        results.append(valid)
        if not valid:
            mismatches.append(citation.claim)
    if not results:
        return (1.0 if not assessment.evidence_items else 0.0), mismatches
    return round(sum(results) / len(results), 6), mismatches


def _source_is_acceptable(evidence, gold: GoldRecord) -> bool:
    if evidence.source == "PubMed":
        aliases = {f"PMID:{evidence.source_identifier}"}
    else:
        identifiers = evidence.source_identifier.split(":")
        aliases = {f"OpenTargets:{identifier}" for identifier in identifiers}
        aliases.add(f"OpenTargets:{evidence.source_identifier}")
    return bool(aliases & set(gold.acceptable_source_ids))


def _claim_is_supported(claim: str, extracted_claim: str) -> bool:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(extracted_claim)
    if not claim_tokens:
        return False
    return len(claim_tokens & evidence_tokens) / len(claim_tokens) >= 0.25


def _prohibited_claims(assessment: Assessment) -> list[str]:
    claims = [assessment.explanation]
    claims.extend(relationship.claim for relationship in assessment.relationships)
    return [
        claim
        for claim in claims
        if any(re.search(pattern, claim, flags=re.IGNORECASE) for pattern in _PROHIBITED_CLAIM_PATTERNS)
    ]


def _failure_types(
    *,
    assessment: Assessment,
    gold: GoldRecord,
    classification_correct: bool,
    relationship_score: float,
    citation_score: float,
) -> list[FailureType]:
    failures: list[FailureType] = []
    if (
        not classification_correct
        and _LABEL_STRENGTH[assessment.label] > _LABEL_STRENGTH[gold.gold_label]
    ):
        failures.append(FailureType.EVIDENCE_STRENGTH_OVERESTIMATION)
    if citation_score < 1.0:
        failures.append(FailureType.CITATION_MISMATCH)
    if relationship_score < 1.0:
        failures.append(FailureType.RELATIONSHIP_EXTRACTION_ERROR)
    if gold.gold_label == "insufficient_evidence" and assessment.label != gold.gold_label:
        failures.append(FailureType.FAILED_ABSTENTION)
    if gold.known_contradictions and not _mentions_contradiction(assessment, gold):
        failures.append(FailureType.CONTRADICTION_IGNORED)
    if not assessment.evidence_items:
        failures.append(FailureType.RETRIEVAL_FAILURE)
    if not classification_correct and not failures:
        failures.append(FailureType.OTHER)
    return _unique(failures)


def _mentions_contradiction(assessment: Assessment, gold: GoldRecord) -> bool:
    assessment_tokens = _tokens(" ".join(assessment.limitations))
    return any(
        len(tokens & assessment_tokens) / len(tokens) >= 0.25
        for contradiction in gold.known_contradictions
        if (tokens := _tokens(contradiction))
    )


def _feedback_text(
    *,
    assessment: Assessment,
    gold: GoldRecord,
    classification_correct: bool,
    relationship_score: float,
    citation_score: float,
    unsupported_claims: list[str],
    failure_types: list[FailureType],
) -> str:
    if gold.split in {"held_out_evaluation", "distribution_shift"}:
        return (
            "Evaluator-only scoring completed. Detailed corrective feedback is "
            "withheld from learning to prevent answer leakage."
        )
    issues: list[str] = []
    if not classification_correct:
        issues.append("The classification does not match the adjudicated label.")
    if relationship_score < 1.0:
        issues.append("One or more relationships do not match the adjudicated relationships.")
    if citation_score < 1.0:
        issues.append("One or more citations do not support their material claim.")
    if unsupported_claims:
        issues.append("Unsupported material claims were detected.")
    if FailureType.CONTRADICTION_IGNORED in failure_types:
        issues.append("A material contradiction was not reported.")
    return " ".join(issues) or "Assessment matches the adjudicated record."


def _relationship_key(subject: str, predicate: str, object_: str) -> tuple[str, str, str]:
    return (_normalize(subject), _normalize(predicate), _normalize(object_))


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _unique(values: Iterable):
    return list(dict.fromkeys(values))
