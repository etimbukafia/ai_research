"""Open Targets disease, target, association, and known-drug retrieval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.drug_repurposing_agent.src.models import DiseaseDrugPair, EvidenceItem
from experiments.drug_repurposing_agent.src.retrieval import (
    CachedHttpClient,
    RetrievalResult,
    error_from_envelope,
)

OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_QUERY = """
query PairEvidence($diseaseId: String!, $drugId: String!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: {index: 0, size: 10}) {
      rows {
        score
        target { id approvedSymbol }
      }
    }
    drugAndClinicalCandidates {
      rows {
        id
        maxClinicalStage
        drug { id name }
      }
    }
  }
  drug(chemblId: $drugId) {
    id
    name
    mechanismsOfAction {
      rows {
        actionType
        mechanismOfAction
        targetName
        targets { id approvedSymbol }
      }
    }
  }
}
""".strip()


class OpenTargetsClient:
    """Retrieve and normalize the allowed Open Targets evidence types."""

    def __init__(self, http: CachedHttpClient | None = None) -> None:
        self.http = http or CachedHttpClient()

    def retrieve(self, pair: DiseaseDrugPair) -> RetrievalResult:
        payload = {
            "query": OPEN_TARGETS_QUERY,
            "variables": {"diseaseId": pair.disease_id, "drugId": pair.drug_id},
        }
        response, envelope = self.http.post_json(
            source="Open Targets",
            url=OPEN_TARGETS_URL,
            payload=payload,
        )
        error = error_from_envelope(envelope)
        errors = [error] if error else []
        if response and response.get("errors"):
            errors.append(
                _graphql_error(envelope, response["errors"])
            )
        evidence = (
            parse_open_targets_response(
                response,
                pair=pair,
                retrieved_at=datetime.fromisoformat(envelope["retrieval_timestamp"]),
            )
            if response and not response.get("errors")
            else []
        )
        return RetrievalResult(
            pair_id=pair.pair_id,
            evidence_items=evidence,
            errors=errors,
            request_hashes=[envelope["request_hash"]],
            cache_hits=int(envelope["cache_hit"]),
            cache_misses=int(not envelope["cache_hit"]),
        )


def parse_open_targets_response(
    response: dict[str, Any],
    *,
    pair: DiseaseDrugPair,
    retrieved_at: datetime,
) -> list[EvidenceItem]:
    """Normalize a cached Open Targets GraphQL response."""

    data = response.get("data") or {}
    disease = data.get("disease")
    drug = data.get("drug")
    evidence: list[EvidenceItem] = []

    if disease:
        evidence.append(
            EvidenceItem(
                evidence_id=f"opentargets:disease:{disease['id']}",
                source="Open Targets",
                source_identifier=disease["id"],
                url=f"https://platform.opentargets.org/disease/{disease['id']}",
                title=f"Open Targets disease: {disease['name']}",
                evidence_type="disease",
                extracted_claim=f"{disease['id']} identifies {disease['name']}.",
                retrieval_timestamp=retrieved_at,
            )
        )
        for row in (disease.get("associatedTargets") or {}).get("rows", []):
            target = row.get("target") or {}
            if not target.get("id"):
                continue
            score = row.get("score")
            evidence.append(
                EvidenceItem(
                    evidence_id=f"opentargets:association:{disease['id']}:{target['id']}",
                    source="Open Targets",
                    source_identifier=f"{disease['id']}:{target['id']}",
                    url=f"https://platform.opentargets.org/disease/{disease['id']}/associations",
                    title=f"{disease['name']} association with {target.get('approvedSymbol', target['id'])}",
                    evidence_type="target_association",
                    extracted_claim=(
                        f"Open Targets associates {disease['name']} with "
                        f"{target.get('approvedSymbol', target['id'])} at score {score}."
                    ),
                    retrieval_timestamp=retrieved_at,
                )
            )
        for row in (disease.get("drugAndClinicalCandidates") or {}).get("rows", []):
            candidate = row.get("drug") or {}
            if candidate.get("id") != pair.drug_id:
                continue
            evidence.append(
                EvidenceItem(
                    evidence_id=f"opentargets:known-drug:{row['id']}",
                    source="Open Targets",
                    source_identifier=row["id"],
                    url=f"https://platform.opentargets.org/disease/{disease['id']}",
                    title=f"{candidate.get('name', pair.drug_name)} clinical candidate for {disease['name']}",
                    evidence_type="known_drug",
                    extracted_claim=(
                        f"Open Targets lists {candidate.get('name', pair.drug_name)} "
                        f"for {disease['name']} at maximum clinical stage "
                        f"{row.get('maxClinicalStage', 'UNKNOWN')}."
                    ),
                    retrieval_timestamp=retrieved_at,
                )
            )

    if drug:
        for index, row in enumerate((drug.get("mechanismsOfAction") or {}).get("rows", []), start=1):
            targets = row.get("targets") or []
            target_text = ", ".join(
                target.get("approvedSymbol") or target.get("id", "unknown target")
                for target in targets
            ) or row.get("targetName", "unknown target")
            evidence.append(
                EvidenceItem(
                    evidence_id=f"opentargets:mechanism:{drug['id']}:{index}",
                    source="Open Targets",
                    source_identifier=drug["id"],
                    url=f"https://platform.opentargets.org/drug/{drug['id']}",
                    title=row.get("mechanismOfAction") or f"{drug['name']} mechanism of action",
                    evidence_type="drug_target_mechanism",
                    extracted_claim=(
                        f"{drug['name']} has action type {row.get('actionType', 'UNKNOWN')} "
                        f"on {target_text}."
                    ),
                    retrieval_timestamp=retrieved_at,
                )
            )
    return evidence


def _graphql_error(envelope: dict[str, Any], errors: list[dict[str, Any]]):
    from experiments.drug_repurposing_agent.src.retrieval import RetrievalErrorRecord

    return RetrievalErrorRecord(
        source="Open Targets",
        request_hash=envelope["request_hash"],
        query_parameters=envelope["query_parameters"],
        retrieval_timestamp=envelope["retrieval_timestamp"],
        error_type="GraphQLError",
        message="; ".join(error.get("message", "Unknown GraphQL error") for error in errors),
        retryable=False,
    )
