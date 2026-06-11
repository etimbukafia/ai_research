from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.drug_repurposing_agent.src.pubmed import (
    parse_pubmed_search,
    parse_pubmed_summary,
    parse_pubmed_xml,
)
from experiments.drug_repurposing_agent.src.retrieval import CachedHttpClient

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


def test_parse_pubmed_search_returns_stable_pmids() -> None:
    response = json.loads((FIXTURES / "pubmed_search.json").read_text())

    assert parse_pubmed_search(response) == ["123", "456"]


def test_parse_pubmed_summary_creates_traceable_evidence() -> None:
    response = json.loads((FIXTURES / "pubmed_summary.json").read_text())

    evidence = parse_pubmed_summary(response, retrieved_at=NOW)

    assert [item.evidence_id for item in evidence] == ["pubmed:123", "pubmed:456"]
    assert all(item.source == "PubMed" for item in evidence)
    assert str(evidence[0].url) == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert "must be inspected" in evidence[0].extracted_claim


def test_parse_pubmed_fetch_includes_abstract_and_safe_missing_abstract() -> None:
    response_text = (FIXTURES / "pubmed_fetch.xml").read_text()

    evidence = parse_pubmed_xml(response_text, retrieved_at=NOW)

    assert "positive primary efficacy signal" in evidence[0].extracted_claim
    assert "Do not infer evidence direction from the title alone" in evidence[1].extracted_claim


class JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload

    @property
    def text(self):
        return json.dumps(self.payload)


class CountingSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        return JsonResponse(self.payload)


def test_raw_response_cache_reuses_request_hash(tmp_path: Path) -> None:
    session = CountingSession({"esearchresult": {"idlist": ["123"]}})
    http = CachedHttpClient(cache_dir=tmp_path, session=session, now=lambda: NOW)
    params = {"db": "pubmed", "term": "test", "retmode": "json"}

    first_response, first_envelope = http.get_json(
        source="PubMed",
        url="https://example.test/esearch",
        params=params,
    )
    second_response, second_envelope = http.get_json(
        source="PubMed",
        url="https://example.test/esearch",
        params=params,
    )

    assert first_response == second_response
    assert first_envelope["request_hash"] == second_envelope["request_hash"]
    assert first_envelope["cache_hit"] is False
    assert second_envelope["cache_hit"] is True
    assert session.calls == 1
