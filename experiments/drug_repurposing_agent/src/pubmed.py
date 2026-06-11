"""PubMed literature retrieval using documented NCBI E-utilities endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from xml.etree import ElementTree

from experiments.drug_repurposing_agent.src.models import DiseaseDrugPair, EvidenceItem
from experiments.drug_repurposing_agent.src.retrieval import (
    CachedHttpClient,
    RetrievalResult,
    error_from_envelope,
)

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedClient:
    """Retrieve exact-pair PubMed records and normalize stable PMID evidence."""

    def __init__(self, http: CachedHttpClient | None = None, *, retmax: int = 10) -> None:
        self.http = http or CachedHttpClient()
        self.retmax = retmax

    def retrieve(self, pair: DiseaseDrugPair) -> RetrievalResult:
        search_params = {
            "db": "pubmed",
            "term": f'"{pair.disease_name}" AND "{pair.drug_name}"',
            "retmode": "json",
            "retmax": self.retmax,
            "sort": "relevance",
        }
        search_response, search_envelope = self.http.get_json(
            source="PubMed",
            url=PUBMED_SEARCH_URL,
            params=search_params,
        )
        envelopes = [search_envelope]
        errors = [error for error in [error_from_envelope(search_envelope)] if error]
        pmids = parse_pubmed_search(search_response) if search_response else []
        fetch_response: str | None = None

        if pmids:
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            fetch_response, fetch_envelope = self.http.get_text(
                source="PubMed",
                url=PUBMED_FETCH_URL,
                params=fetch_params,
            )
            envelopes.append(fetch_envelope)
            error = error_from_envelope(fetch_envelope)
            if error:
                errors.append(error)

        retrieved_at = datetime.fromisoformat(envelopes[-1]["retrieval_timestamp"])
        evidence = parse_pubmed_xml(fetch_response, retrieved_at=retrieved_at)
        return RetrievalResult(
            pair_id=pair.pair_id,
            evidence_items=evidence,
            errors=errors,
            request_hashes=[envelope["request_hash"] for envelope in envelopes],
            cache_hits=sum(int(envelope["cache_hit"]) for envelope in envelopes),
            cache_misses=sum(int(not envelope["cache_hit"]) for envelope in envelopes),
        )


def parse_pubmed_search(response: dict[str, Any] | None) -> list[str]:
    """Return stable PubMed identifiers from an ESearch response."""

    if not response:
        return []
    return [
        str(pmid)
        for pmid in (response.get("esearchresult") or {}).get("idlist", [])
        if str(pmid).strip()
    ]


def parse_pubmed_summary(
    response: dict[str, Any] | None,
    *,
    retrieved_at: datetime,
) -> list[EvidenceItem]:
    """Normalize ESummary records into traceable literature evidence."""

    if not response:
        return []
    result = response.get("result") or {}
    evidence: list[EvidenceItem] = []
    for pmid in result.get("uids", []):
        record = result.get(str(pmid)) or {}
        title = record.get("title") or f"PubMed record {pmid}"
        publication_types = ", ".join(record.get("pubtype") or []) or "unspecified publication type"
        publication_date = record.get("pubdate") or "unknown publication date"
        evidence.append(
            EvidenceItem(
                evidence_id=f"pubmed:{pmid}",
                source="PubMed",
                source_identifier=str(pmid),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=title,
                evidence_type="literature",
                extracted_claim=(
                    f"PubMed indexes '{title}' as {publication_types}, published "
                    f"{publication_date}. The underlying record must be inspected "
                    "before inferring direction or efficacy."
                ),
                retrieval_timestamp=retrieved_at,
            )
        )
    return evidence


def parse_pubmed_xml(
    response_text: str | None,
    *,
    retrieved_at: datetime,
) -> list[EvidenceItem]:
    """Normalize EFetch XML articles, including abstracts when available."""

    if not response_text:
        return []
    root = ElementTree.fromstring(response_text)
    evidence: list[EvidenceItem] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _element_text(article.find(".//PMID"))
        if not pmid:
            continue
        title = _element_text(article.find(".//ArticleTitle")) or f"PubMed record {pmid}"
        abstract_parts = [
            _element_text(element)
            for element in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)
        publication_types = [
            _element_text(element)
            for element in article.findall(".//PublicationTypeList/PublicationType")
        ]
        publication_type_text = ", ".join(part for part in publication_types if part)
        evidence.append(
            EvidenceItem(
                evidence_id=f"pubmed:{pmid}",
                source="PubMed",
                source_identifier=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=title,
                evidence_type="literature",
                extracted_claim=abstract or (
                    f"PubMed indexes '{title}' as "
                    f"{publication_type_text or 'an article'}, but no abstract was available. "
                    "Do not infer evidence direction from the title alone."
                ),
                retrieval_timestamp=retrieved_at,
            )
        )
    return evidence


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()
