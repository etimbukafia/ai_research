from __future__ import annotations

from datetime import datetime, timezone

import pytest

from experiments.drug_repurposing_agent.src.agent import AssessorError, BaselineAssessor
from experiments.drug_repurposing_agent.src.models import DiseaseDrugPair, EvidenceItem
from experiments.drug_repurposing_agent.src.retrieval import RetrievalErrorRecord, RetrievalResult

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


class MockModel:
    def __init__(self, response):
        self.response = response
        self.prompt = ""

    def complete(self, prompt: str):
        self.prompt = prompt
        return self.response


class RepairingMockModel:
    def __init__(self, first, repaired):
        self.responses = iter([first, repaired])
        self.calls = 0

    def complete(self, prompt: str):
        self.calls += 1
        return next(self.responses)


def pair() -> DiseaseDrugPair:
    return DiseaseDrugPair(
        pair_id="disease-drug-pair-001",
        disease_name="Disease A",
        disease_id="EFO_0000001",
        drug_name="Drug A",
        drug_id="CHEMBL1",
    )


def evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="pubmed:123",
        source="PubMed",
        source_identifier="123",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
        title="Exact-pair study",
        evidence_type="controlled_human_study",
        extracted_claim="A controlled human study reported a positive efficacy signal.",
        retrieval_timestamp=NOW,
    )


def valid_payload() -> dict:
    claim = "Direct controlled human evidence supports investigation."
    return {
        "pair_id": pair().pair_id,
        "label": "supported",
        "confidence": 0.9,
        "evidence_items": [evidence().model_dump(mode="json")],
        "relationships": [
            {
                "subject": "Drug A",
                "predicate": "evaluated_in",
                "object": "Disease A",
                "claim": claim,
                "evidence_ids": ["pubmed:123"],
            }
        ],
        "explanation": claim,
        "limitations": ["The evidence-support label does not establish safety."],
        "citations": [{"claim": claim, "evidence_ids": ["pubmed:123"]}],
    }


def retrieval() -> RetrievalResult:
    return RetrievalResult(pair_id=pair().pair_id, evidence_items=[evidence()])


def test_assessor_returns_validated_assessment_and_supplies_runtime_input() -> None:
    model = MockModel(valid_payload())

    assessment = BaselineAssessor(model).assess(pair=pair(), retrieval=retrieval())

    assert assessment.label == "supported"
    assert '"condition": "no_memory"' in model.prompt
    assert '"memories": []' in model.prompt
    assert "character-for-character" in model.prompt


def test_assessor_rejects_out_of_schema_label() -> None:
    payload = valid_payload()
    payload["label"] = "promising"

    with pytest.raises(AssessorError, match="invalid Assessment"):
        BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=retrieval())


def test_assessor_rejects_uncited_material_claim() -> None:
    payload = valid_payload()
    payload["citations"] = []

    with pytest.raises(AssessorError, match="material assessment claims require citations"):
        BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=retrieval())


def test_assessor_repairs_one_invalid_structured_response() -> None:
    invalid = valid_payload()
    invalid["citations"] = []
    model = RepairingMockModel(invalid, valid_payload())

    result = BaselineAssessor(model).assess(pair=pair(), retrieval=retrieval())

    assert result.label == "supported"
    assert model.calls == 2


def test_assessor_rejects_invented_or_modified_evidence() -> None:
    payload = valid_payload()
    payload["evidence_items"][0]["title"] = "Invented title"

    with pytest.raises(AssessorError, match="exactly match"):
        BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=retrieval())


def test_assessor_rejects_wrong_pair_id() -> None:
    payload = valid_payload()
    payload["pair_id"] = "disease-drug-pair-002"

    with pytest.raises(AssessorError, match="pair_id does not match"):
        BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=retrieval())


def test_assessor_rejects_memories_for_no_memory_condition() -> None:
    with pytest.raises(AssessorError, match="cannot receive memories"):
        BaselineAssessor(MockModel(valid_payload())).assess(
            pair=pair(),
            retrieval=retrieval(),
            memories=[{"lesson": "Use direct evidence first."}],
        )


def test_assessor_supplies_memories_for_memory_condition() -> None:
    model = MockModel(valid_payload())

    BaselineAssessor(model).assess(
        pair=pair(),
        retrieval=retrieval(),
        condition="validated_lessons",
        memories=[{"lesson": "Use direct evidence first."}],
    )

    assert "Use direct evidence first." in model.prompt
    assert '"condition": "validated_lessons"' in model.prompt


def test_material_retrieval_failure_requires_zero_evidence_abstention() -> None:
    error = RetrievalErrorRecord(
        source="PubMed",
        request_hash="a" * 64,
        query_parameters={"term": "test"},
        retrieval_timestamp=NOW,
        error_type="Timeout",
        message="timed out",
        retryable=True,
    )
    failed = RetrievalResult(pair_id=pair().pair_id, errors=[error])
    payload = {
        "pair_id": pair().pair_id,
        "label": "insufficient_evidence",
        "confidence": 0.95,
        "evidence_items": [],
        "relationships": [],
        "explanation": "Retrieval failed materially, so no directional judgment is possible.",
        "limitations": ["No evidence was available to assess."],
        "citations": [],
    }

    assessment = BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=failed)

    assert assessment.label == "insufficient_evidence"
    assert assessment.citations == []


def test_material_retrieval_failure_rejects_non_abstention() -> None:
    empty = RetrievalResult(pair_id=pair().pair_id)
    payload = valid_payload()
    payload["evidence_items"] = []
    payload["relationships"] = []
    payload["citations"] = []
    payload["label"] = "unsupported"

    with pytest.raises(AssessorError, match="invalid Assessment"):
        BaselineAssessor(MockModel(payload)).assess(pair=pair(), retrieval=empty)
