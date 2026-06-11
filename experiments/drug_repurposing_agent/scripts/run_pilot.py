"""Build and run the 12-pair development pilot outside the frozen dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from experiments.drug_repurposing_agent.src.agent import BaselineAssessor
from experiments.drug_repurposing_agent.src.lessons import LessonWorkflow
from experiments.drug_repurposing_agent.src.models import EvidenceItem
from experiments.drug_repurposing_agent.src.retrieval import RetrievalResult
from experiments.drug_repurposing_agent.src.runner import (
    DEFAULT_CONFIG_PATH,
    GeminiStructuredModel,
    report_runs,
    run_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "pilot"
DATA_DIR = PILOT_ROOT / "data"
RUNS_DIR = PILOT_ROOT / "runs"
NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)
LABELS = ("supported", "weakly_supported", "unsupported", "insufficient_evidence")


def main() -> None:
    build_pilot_dataset()
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for condition in ("no_memory", "raw_memory", "validated_lessons"):
        model = GeminiStructuredModel(
            model_name=config["model"]["name"],
            temperature=float(config["model"]["temperature"]),
            seed=42,
        )
        run_experiment(
            condition=condition,
            seed=42,
            assessor=BaselineAssessor(model),
            lesson_workflow=LessonWorkflow(model) if condition == "validated_lessons" else None,
            data_dir=DATA_DIR,
            runs_dir=RUNS_DIR,
            retrieval_fn=retrieve_pilot_evidence,
        )
    write_review()
    print(json.dumps(report_runs(RUNS_DIR), indent=2, sort_keys=True))


def build_pilot_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pairs, gold, evidence = [], [], {}
    for index in range(12):
        ordinal = 201 + index
        label = LABELS[index % 4]
        pair_id = f"disease-drug-pair-{ordinal:03d}"
        disease = f"Pilot Rare Disease {ordinal}"
        drug = f"PILOT-DRUG-{ordinal}"
        pmid = str(90000000 + ordinal)
        pair = {
            "pair_id": pair_id,
            "disease_name": disease,
            "disease_id": f"MONDO_PILOT_{ordinal}",
            "drug_name": drug,
            "drug_id": f"CHEMBL_PILOT_{ordinal}",
        }
        items = _evidence_for(pair, label, pmid)
        predicate = {
            "supported": "has_direct_positive_human_evidence_for",
            "weakly_supported": "has_preliminary_or_indirect_evidence_for",
            "unsupported": "has_direct_negative_human_evidence_for",
            "insufficient_evidence": None,
        }[label]
        pairs.append(pair)
        gold.append(
            {
                "pair_id": pair_id,
                "split": "learning_stream",
                "gold_label": label,
                "confidence": 0.9,
                "evidence_pattern": f"pilot_{label}",
                "expected_relationships": (
                    [{"subject": drug, "predicate": predicate, "object": disease}]
                    if predicate
                    else []
                ),
                "acceptable_source_ids": [
                    f"OpenTargets:{pair['disease_id']}",
                    f"OpenTargets:{pair['drug_id']}",
                    *([f"PMID:{pmid}"] if items else []),
                ],
                "rationale": f"Pilot fixture for the {label} decision rule.",
                "known_contradictions": [],
                "reviewer": "Codex (designated pilot reviewer)",
                "reviewed_at": NOW.isoformat(),
                "adjudication_notes": "Synthetic development pilot; excluded from frozen dataset.",
            }
        )
        evidence[pair_id] = [item.model_dump(mode="json") for item in items]
    _write_jsonl(DATA_DIR / "pairs.jsonl", pairs)
    _write_jsonl(DATA_DIR / "gold.jsonl", gold)
    (DATA_DIR / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def retrieve_pilot_evidence(pair) -> RetrievalResult:
    evidence = json.loads((DATA_DIR / "evidence.json").read_text(encoding="utf-8"))
    return RetrievalResult(
        pair_id=pair.pair_id,
        evidence_items=[EvidenceItem.model_validate(item) for item in evidence[pair.pair_id]],
        request_hashes=[f"pilot-snapshot:{pair.pair_id}"],
        cache_hits=1,
    )


def write_review() -> None:
    rows = []
    label_counts = {}
    for records_path in sorted(RUNS_DIR.glob("*/records.jsonl")):
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        label_counts[records_path.parent.name] = {}
        for record in records:
            label = record["assessment"]["label"]
            label_counts[records_path.parent.name][label] = label_counts[records_path.parent.name].get(label, 0) + 1
            citation_ids = {
                evidence_id
                for citation in record["assessment"]["citations"]
                for evidence_id in citation["evidence_ids"]
            }
            known_ids = {item["evidence_id"] for item in record["assessment"]["evidence_items"]}
            rows.append(
                {
                    "session": records_path.parent.name,
                    "pair_id": record["pair"]["pair_id"],
                    "label": label,
                    "classification_correct": record["evaluator_feedback"]["classification_correct"],
                    "citation_ids_valid": citation_ids <= known_ids,
                    "unsupported_claims": record["evaluator_feedback"]["unsupported_claims"],
                    "retrieved_memory_ids": record["retrieved_memory_ids"],
                    "written_memory_ids": record["written_memory_ids"],
                }
            )
    review = {
        "pilot_status": "completed",
        "dataset_is_outside_frozen_100": True,
        "pairs": 12,
        "conditions": 3,
        "label_counts": label_counts,
        "manual_reviewer": "Codex (designated pilot reviewer)",
        "manual_review": {
            "all_citation_ids_traceable": all(row["citation_ids_valid"] for row in rows),
            "unsupported_claim_rows": [row for row in rows if row["unsupported_claims"]],
            "raw_memory_records": _memory_count("raw_memory-seed-42"),
            "validated_lesson_records": _memory_count("validated_lessons-seed-42"),
        },
        "records": rows,
    }
    (PILOT_ROOT / "PILOT_REVIEW.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _memory_count(session: str) -> int:
    path = RUNS_DIR / session / "memory.json"
    return len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0


def _evidence_for(pair: dict, label: str, pmid: str) -> list[EvidenceItem]:
    if label == "insufficient_evidence":
        return []
    claim = {
        "supported": (
            f"A peer-reviewed controlled human study directly evaluated {pair['drug_name']} "
            f"in {pair['disease_name']} and reported a credible positive primary efficacy signal."
        ),
        "weakly_supported": (
            f"A disease-model study reported a preliminary positive preclinical signal for "
            f"{pair['drug_name']} in {pair['disease_name']}; no direct human efficacy study was available."
        ),
        "unsupported": (
            f"Two controlled human studies directly evaluated {pair['drug_name']} in "
            f"{pair['disease_name']} and consistently reported no meaningful efficacy."
        ),
    }[label]
    return [
        EvidenceItem(
            evidence_id=f"pubmed:{pmid}",
            source="PubMed",
            source_identifier=pmid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            title=f"Synthetic pilot evidence for {pair['pair_id']}",
            evidence_type=(
                "controlled_human_study"
                if label in {"supported", "unsupported"}
                else "animal_disease_model"
            ),
            extracted_claim=claim,
            retrieval_timestamp=NOW,
        )
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
