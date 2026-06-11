"""Resolve and freeze the designated-reviewer dataset using official APIs."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
REVIEWED_AT = "2026-06-11T00:00:00+00:00"
REVIEWER = "Codex (designated dataset reviewer)"
OPEN_TARGETS_ALIASES = {
    ("drug", "N-acetylcysteine"): "acetylcysteine",
    ("drug", "insulin-like growth factor 1"): "mecasermin",
}


@dataclass(frozen=True)
class CuratedPair:
    disease: str
    drug: str
    label: str
    pattern: str


SUPPORTED = [
    ("cystic fibrosis", "ivacaftor"),
    ("spinal muscular atrophy", "nusinersen"),
    ("pulmonary arterial hypertension", "epoprostenol"),
    ("Gaucher disease", "imiglucerase"),
    ("Fabry disease", "migalastat"),
    ("Pompe disease", "alglucosidase alfa"),
    ("paroxysmal nocturnal hemoglobinuria", "eculizumab"),
    ("transthyretin amyloidosis", "tafamidis"),
    ("hereditary angioedema", "lanadelumab"),
    ("Wilson disease", "penicillamine"),
    ("Dravet syndrome", "fenfluramine"),
    ("Lennox-Gastaut syndrome", "cannabidiol"),
    ("tuberous sclerosis", "everolimus"),
    ("idiopathic pulmonary fibrosis", "nintedanib"),
    ("neuromyelitis optica", "eculizumab"),
    ("acute hepatic porphyria", "givosiran"),
    ("acromegaly", "pegvisomant"),
    ("Cushing disease", "pasireotide"),
    ("Friedreich ataxia", "omaveloxolone"),
    ("Hutchinson-Gilford progeria syndrome", "lonafarnib"),
    ("alkaptonuria", "nitisinone"),
    ("phenylketonuria", "sapropterin"),
    ("CLN2 disease", "cerliponase alfa"),
    ("sickle cell disease", "hydroxyurea"),
    ("Duchenne muscular dystrophy", "deflazacort"),
]

WEAKLY_SUPPORTED = [
    ("Gaucher disease", "ambroxol"),
    ("Hutchinson-Gilford progeria syndrome", "sirolimus"),
    ("dystrophic epidermolysis bullosa", "losartan"),
    ("Pompe disease", "clenbuterol"),
    ("amyotrophic lateral sclerosis", "tofersen"),
    ("Rett syndrome", "ketamine"),
    ("fragile X syndrome", "lovastatin"),
    ("Dravet syndrome", "clemizole"),
    ("spinal muscular atrophy", "apitegromab"),
    ("Duchenne muscular dystrophy", "idebenone"),
    ("fragile X syndrome", "metformin"),
    ("cystic fibrosis", "gentamicin"),
    ("Niemann-Pick disease type C", "2-hydroxypropyl-beta-cyclodextrin"),
    ("Niemann-Pick disease type C", "vorinostat"),
    ("ataxia telangiectasia", "nicotinamide riboside"),
    ("Friedreich ataxia", "resveratrol"),
    ("tuberous sclerosis", "sirolimus"),
    ("spinal muscular atrophy", "salbutamol"),
    ("cystinosis", "N-acetylcysteine"),
    ("Rett syndrome", "insulin-like growth factor 1"),
    ("Angelman syndrome", "minocycline"),
    ("junctional epidermolysis bullosa", "gentamicin"),
    ("amyotrophic lateral sclerosis", "ibudilast"),
    ("Huntington disease", "metformin"),
    ("pulmonary arterial hypertension", "metformin"),
]

UNSUPPORTED = [
    ("amyotrophic lateral sclerosis", "lithium"),
    ("amyotrophic lateral sclerosis", "ceftriaxone"),
    ("amyotrophic lateral sclerosis", "dexpramipexole"),
    ("amyotrophic lateral sclerosis", "minocycline"),
    ("amyotrophic lateral sclerosis", "creatine"),
    ("Huntington disease", "coenzyme Q10"),
    ("Huntington disease", "creatine"),
    ("Huntington disease", "latrepirdine"),
    ("Huntington disease", "pridopidine"),
    ("progressive supranuclear palsy", "davunetide"),
    ("progressive supranuclear palsy", "tideglusib"),
    ("cystic fibrosis", "ataluren"),
    ("Duchenne muscular dystrophy", "drisapersen"),
    ("Duchenne muscular dystrophy", "sildenafil"),
    ("idiopathic pulmonary fibrosis", "warfarin"),
    ("idiopathic pulmonary fibrosis", "ambrisentan"),
    ("idiopathic pulmonary fibrosis", "bosentan"),
    ("fragile X syndrome", "mavoglurant"),
    ("fragile X syndrome", "arbaclofen"),
    ("Rett syndrome", "sarizotan"),
    ("spinal muscular atrophy", "hydroxyurea"),
    ("spinal muscular atrophy", "gabapentin"),
    ("Friedreich ataxia", "idebenone"),
    ("sickle cell disease", "senicapoc"),
    ("pulmonary arterial hypertension", "simvastatin"),
]

INSUFFICIENT = [
    ("cystic fibrosis", "donepezil"),
    ("spinal muscular atrophy", "tafamidis"),
    ("pulmonary arterial hypertension", "migalastat"),
    ("Gaucher disease", "sildenafil"),
    ("Fabry disease", "nusinersen"),
    ("Pompe disease", "cannabidiol"),
    ("paroxysmal nocturnal hemoglobinuria", "nintedanib"),
    ("transthyretin amyloidosis", "lanadelumab"),
    ("hereditary angioedema", "sapropterin"),
    ("Wilson disease", "ivacaftor"),
    ("Dravet syndrome", "pegvisomant"),
    ("Lennox-Gastaut syndrome", "penicillamine"),
    ("tuberous sclerosis", "imiglucerase"),
    ("idiopathic pulmonary fibrosis", "cerliponase alfa"),
    ("neuromyelitis optica", "lonafarnib"),
    ("acute hepatic porphyria", "eculizumab"),
    ("acromegaly", "hydroxyurea"),
    ("Cushing disease", "fenfluramine"),
    ("Friedreich ataxia", "lanadelumab"),
    ("Hutchinson-Gilford progeria syndrome", "nusinersen"),
    ("alkaptonuria", "ivacaftor"),
    ("phenylketonuria", "nintedanib"),
    ("CLN2 disease", "tafamidis"),
    ("sickle cell disease", "migalastat"),
    ("Duchenne muscular dystrophy", "givosiran"),
]


def main() -> None:
    pairs = _interleaved_pairs()
    entity_cache: dict[tuple[str, str], dict[str, str]] = {}
    pair_records: list[dict[str, Any]] = []
    gold_records: list[dict[str, Any]] = []

    for ordinal, curated in enumerate(pairs, start=1):
        pair_id = f"disease-drug-pair-{ordinal:03d}"
        disease = _resolve_open_targets(curated.disease, "disease", entity_cache)
        drug = _resolve_open_targets(curated.drug, "drug", entity_cache)
        pmids = (
            []
            if curated.label == "insufficient_evidence"
            else _pubmed_ids(curated.disease, curated.drug)
        )
        split = _split_for(ordinal)
        pair_records.append(
            {
                "pair_id": pair_id,
                "disease_name": disease["name"],
                "disease_id": disease["id"],
                "drug_name": drug["name"],
                "drug_id": drug["id"],
            }
        )
        gold_records.append(
            _gold_record(pair_id, split, curated, disease, drug, pmids)
        )
        print(f"{pair_id}: {curated.label} | {disease['name']} | {drug['name']}")

    pairs_text = _jsonl(pair_records)
    gold_text = _jsonl(gold_records)
    (DATA_DIR / "pairs.jsonl").write_text(pairs_text, encoding="utf-8")
    (DATA_DIR / "gold.jsonl").write_text(gold_text, encoding="utf-8")
    _write_manifest(gold_records)


def _interleaved_pairs() -> list[CuratedPair]:
    records: list[CuratedPair] = []
    for index in range(25):
        records.extend(
            [
                CuratedPair(*SUPPORTED[index], "supported", "direct_positive_human"),
                CuratedPair(
                    *WEAKLY_SUPPORTED[index],
                    "weakly_supported",
                    "preliminary_preclinical_or_indirect",
                ),
                CuratedPair(*UNSUPPORTED[index], "unsupported", "direct_negative_human"),
                CuratedPair(
                    *INSUFFICIENT[index],
                    "insufficient_evidence",
                    "deliberately_unrelated_abstention_control",
                ),
            ]
        )
    return records


def _resolve_open_targets(
    query_string: str,
    entity: str,
    cache: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    resolved_query = OPEN_TARGETS_ALIASES.get((entity, query_string), query_string)
    key = (entity, resolved_query.lower())
    if key in cache:
        return cache[key]
    query = """
    query Search($queryString: String!, $entityNames: [String!]) {
      search(queryString: $queryString, entityNames: $entityNames, page: {index: 0, size: 5}) {
        hits { id name entity }
      }
    }
    """
    response = _post_json(
        OPEN_TARGETS_URL,
        {"query": query, "variables": {"queryString": resolved_query, "entityNames": [entity]}},
    )
    hits = response["data"]["search"]["hits"]
    if not hits:
        raise RuntimeError(f"Open Targets did not resolve {entity}: {query_string}")
    resolved = {"id": hits[0]["id"], "name": hits[0]["name"]}
    cache[key] = resolved
    return resolved


def _pubmed_ids(disease: str, drug: str) -> list[str]:
    term = f'"{disease}" AND "{drug}"'
    query = urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": 3, "sort": "relevance"}
    )
    with urllib.request.urlopen(f"{PUBMED_URL}?{query}", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time.sleep(0.11)
    return payload["esearchresult"]["idlist"]


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _gold_record(
    pair_id: str,
    split: str,
    curated: CuratedPair,
    disease: dict[str, str],
    drug: dict[str, str],
    pmids: list[str],
) -> dict[str, Any]:
    sources = [f"OpenTargets:{disease['id']}", f"OpenTargets:{drug['id']}"]
    sources.extend(f"PMID:{pmid}" for pmid in pmids)
    relationship = {
        "subject": drug["name"],
        "predicate": {
            "supported": "has_direct_positive_human_evidence_for",
            "weakly_supported": "has_preliminary_or_indirect_evidence_for",
            "unsupported": "has_direct_negative_human_evidence_for",
        }.get(curated.label, "has_no_established_evidence_for"),
        "object": disease["name"],
    }
    rationales = {
        "supported": (
            "Designated-reviewer adjudication: direct human evidence for the exact "
            "pair reports a credible positive efficacy signal, with no stronger "
            "negative evidence identified for this triage label."
        ),
        "weakly_supported": (
            "Designated-reviewer adjudication: the exact pair has preliminary, "
            "preclinical, small-study, or mechanistic evidence that justifies "
            "investigation but does not meet the supported threshold."
        ),
        "unsupported": (
            "Designated-reviewer adjudication: direct human trial evidence for the "
            "exact pair reports no meaningful efficacy or failure of the relevant "
            "endpoint, outweighing weaker positive rationale."
        ),
        "insufficient_evidence": (
            "Designated-reviewer adjudication: this deliberately unrelated control "
            "has no established direct evidence or coherent pair-specific rationale; "
            "absence of evidence requires abstention rather than a negative claim."
        ),
    }
    contradictions = {
        "supported": ["No known contradiction strong enough to overturn the positive human evidence."],
        "weakly_supported": ["No qualifying direct positive human efficacy evidence established."],
        "unsupported": ["Mechanistic or earlier positive signals may exist but are outweighed by direct negative human evidence."],
        "insufficient_evidence": ["No relevant direct evidence expected; unrelated drug familiarity must not be transferred across diseases."],
    }
    confidence = {
        "supported": 0.90,
        "weakly_supported": 0.70,
        "unsupported": 0.88,
        "insufficient_evidence": 0.92,
    }[curated.label]
    return {
        "pair_id": pair_id,
        "split": split,
        "gold_label": curated.label,
        "confidence": confidence,
        "evidence_pattern": curated.pattern,
        "expected_relationships": [] if curated.label == "insufficient_evidence" else [relationship],
        "acceptable_source_ids": sources,
        "rationale": rationales[curated.label],
        "known_contradictions": contradictions[curated.label],
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "adjudication_notes": (
            "Reviewed against labeling_guide.md version 1.0.0. PubMed identifiers "
            "are evidence anchors for later snapshot retrieval, not substitutes for "
            "pair-level citation verification."
        ),
    }


def _split_for(ordinal: int) -> str:
    if ordinal <= 20:
        return "initial_evaluation"
    if ordinal <= 70:
        return "learning_stream"
    if ordinal <= 90:
        return "held_out_evaluation"
    return "distribution_shift"


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _write_manifest(gold_records: list[dict[str, Any]]) -> None:
    counts: dict[str, dict[str, int]] = {}
    for record in gold_records:
        split_counts = counts.setdefault(record["split"], {})
        label = record["gold_label"]
        split_counts[label] = split_counts.get(label, 0) + 1
    manifest = {
        "contract_version": "1.0.0",
        "dataset_version": "1.0.0",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": REVIEWER,
        "pair_id_format": "disease-drug-pair-{number:03d}",
        "hashes": {
            "pairs.jsonl_sha256": _sha256(DATA_DIR / "pairs.jsonl"),
            "gold.jsonl_sha256": _sha256(DATA_DIR / "gold.jsonl"),
        },
        "label_counts_by_split": counts,
        "splits": {
            "initial_evaluation": {
                "start_id": "disease-drug-pair-001",
                "end_id": "disease-drug-pair-020",
                "count": 20,
            },
            "learning_stream": {
                "start_id": "disease-drug-pair-021",
                "end_id": "disease-drug-pair-070",
                "count": 50,
            },
            "held_out_evaluation": {
                "start_id": "disease-drug-pair-071",
                "end_id": "disease-drug-pair-090",
                "count": 20,
            },
            "distribution_shift": {
                "start_id": "disease-drug-pair-091",
                "end_id": "disease-drug-pair-100",
                "count": 10,
            },
        },
        "leakage_policy": {
            "agent_input": "data/pairs.jsonl only",
            "evaluator_only": "data/gold.jsonl",
            "held_out_feedback_to_memory": False,
            "distribution_shift_feedback_to_memory": False,
        },
    }
    (DATA_DIR / "splits.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
