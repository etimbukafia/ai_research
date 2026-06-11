from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from experiments.drug_repurposing_agent.src.models import DiseaseDrugPair
from experiments.drug_repurposing_agent.src.open_targets import parse_open_targets_response
from experiments.drug_repurposing_agent.src.retrieval import (
    CachedHttpClient,
    RetrievalResult,
    error_from_envelope,
)

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def pair() -> DiseaseDrugPair:
    return DiseaseDrugPair(
        pair_id="disease-drug-pair-001",
        disease_name="cystic fibrosis",
        disease_id="MONDO_0009061",
        drug_name="IVACAFTOR",
        drug_id="CHEMBL2010601",
    )


def test_parse_open_targets_allowed_evidence_and_exact_known_drug() -> None:
    response = json.loads((FIXTURES / "open_targets_pair.json").read_text())

    evidence = parse_open_targets_response(response, pair=pair(), retrieved_at=NOW)

    assert {item.evidence_type for item in evidence} == {
        "disease",
        "target_association",
        "known_drug",
        "drug_target_mechanism",
    }
    known_drugs = [item for item in evidence if item.evidence_type == "known_drug"]
    assert len(known_drugs) == 1
    assert known_drugs[0].source_identifier == "known-pair-1"


class TimeoutSession:
    def request(self, *args, **kwargs):
        raise requests.Timeout("timed out")


def test_http_timeout_creates_cached_explicit_error(tmp_path: Path) -> None:
    http = CachedHttpClient(
        cache_dir=tmp_path,
        session=TimeoutSession(),
        now=lambda: NOW,
    )

    response, envelope = http.post_json(
        source="Open Targets",
        url="https://example.test/graphql",
        payload={"query": "query { test }"},
    )
    error = error_from_envelope(envelope)

    assert response is None
    assert error is not None
    assert error.error_type == "Timeout"
    assert error.retryable is True
    assert len(list(tmp_path.rglob("*.json"))) == 1

    result = RetrievalResult(pair_id=pair().pair_id, errors=[error])
    assert result.materially_failed is True
    assert result.required_abstention_label == "insufficient_evidence"
