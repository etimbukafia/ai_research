"""Metrics for retrieval, answers, compression, and latency."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Iterable

from answer import Answer


def ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": int(correct),
        "total": int(total),
        "rate": round(correct / total, 8) if total else None,
    }


def bool_metric(values: Iterable[bool | None]) -> dict[str, Any]:
    usable = [value for value in values if value is not None]
    return ratio(sum(bool(value) for value in usable), len(usable))


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 6)


def rank_of(candidate_ids: list[str], expected_id: str | None) -> int | None:
    if expected_id is None or expected_id not in candidate_ids:
        return None
    return candidate_ids.index(expected_id) + 1


def evaluate_row(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    answer: Answer,
    compression: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_status = case["expected_answer_status"]
    expected_id = case["applicable_document_id"]
    candidate_ids = [item["document_id"] for item in retrieval["candidate_documents"]]
    selected_id = retrieval.get("selected_document_id")
    selected = documents_by_id.get(selected_id) if selected_id else None
    expected_answer_case = expected_status == "answer" and expected_id is not None
    expected_abstention_case = expected_status != "answer"

    rank = rank_of(candidate_ids, expected_id)
    exact_fact = None
    unit = None
    source_precision = None
    source_recall = None
    if expected_answer_case:
        expected_fact = case["expected_answer"]
        exact_fact = (
            answer.status == "answer"
            and answer.value == expected_fact["value"]
            and answer.unit == expected_fact["unit"]
        )
        unit = answer.status == "answer" and answer.unit == expected_fact["unit"]
        expected_sources = {expected_id}
        predicted_sources = set(answer.source_ids)
        source_precision = (
            len(expected_sources & predicted_sources) / len(predicted_sources)
            if predicted_sources
            else 0.0
        )
        source_recall = len(expected_sources & predicted_sources) / len(expected_sources)

    if expected_abstention_case:
        status_correct = answer.status == expected_status
        strict_applicability = selected_id is None and status_correct
    else:
        status_correct = answer.status == "answer"
        strict_applicability = selected_id == expected_id

    version_correct = None
    region_correct = None
    authority_correct = None
    supersession_correct = None
    path_correct = None
    if expected_answer_case:
        scope = case["required_scope"]
        version_correct = bool(selected and selected["firmware_versions"] == [scope["firmware_version"]])
        region_correct = bool(selected and selected["regions"] == [scope["region"]])
        authority_correct = bool(selected and selected["authority"] == scope["authority"])
        supersession_correct = bool(
            selected
            and selected["status"] == "Approved"
            and selected["document_type"] == "specification"
        )
        path_correct = answer.evidence_path == case["required_evidence_path"]

    unsupported = bool(answer.unsupported_claims) or (
        expected_answer_case and selected_id != expected_id
    )
    return {
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "case_class": case["case_class"],
        "is_dense_overlap": case["is_dense_overlap"],
        "is_holdout": case["is_holdout"],
        "expected_status": expected_status,
        "candidate_document_ids": candidate_ids,
        "selected_document_id": selected_id,
        "candidate_recall_at_5": expected_id in candidate_ids if expected_answer_case else None,
        "reciprocal_rank": 1.0 / rank if rank else (0.0 if expected_answer_case else None),
        "strict_applicability": strict_applicability,
        "version_accuracy": version_correct,
        "region_accuracy": region_correct,
        "authority_accuracy": authority_correct,
        "supersession_accuracy": supersession_correct,
        "evidence_path_accuracy": path_correct,
        "correct_abstention": strict_applicability if expected_abstention_case else None,
        "exact_fact_accuracy": exact_fact,
        "unit_accuracy": unit,
        "source_precision": source_precision,
        "source_recall": source_recall,
        "unsupported_claim": unsupported if expected_answer_case else None,
        "answer_status_accuracy": status_correct,
        "retrieval_latency_ms": retrieval["retrieval_latency_ms"],
        "full_context_tokens": compression.get("full_context_tokens"),
        "evidence_packet_tokens": compression.get("evidence_packet_tokens"),
        "compression_ratio": compression.get("compression_ratio"),
        "required_evidence_retained": compression.get("required_evidence_retained"),
        "selected_evidence_path": retrieval.get("selected_evidence_path", []),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_latency = [float(row["retrieval_latency_ms"]) for row in rows]
    source_precisions = [row["source_precision"] for row in rows if row["source_precision"] is not None]
    source_recalls = [row["source_recall"] for row in rows if row["source_recall"] is not None]
    compression_ratios = [row["compression_ratio"] for row in rows if row["compression_ratio"] is not None]
    full_tokens = [row["full_context_tokens"] for row in rows if row["full_context_tokens"] is not None]
    packet_tokens = [row["evidence_packet_tokens"] for row in rows if row["evidence_packet_tokens"] is not None]
    return {
        "case_count": len(rows),
        "candidate_recall_at_5": bool_metric(row["candidate_recall_at_5"] for row in rows),
        "mean_reciprocal_rank": round(
            statistics.mean(
                [row["reciprocal_rank"] for row in rows if row["reciprocal_rank"] is not None]
            ),
            8,
        )
        if any(row["reciprocal_rank"] is not None for row in rows)
        else None,
        "strict_applicability_accuracy": bool_metric(row["strict_applicability"] for row in rows if row["expected_status"] == "answer"),
        "version_accuracy": bool_metric(row["version_accuracy"] for row in rows),
        "region_accuracy": bool_metric(row["region_accuracy"] for row in rows),
        "authority_accuracy": bool_metric(row["authority_accuracy"] for row in rows),
        "supersession_accuracy": bool_metric(row["supersession_accuracy"] for row in rows),
        "evidence_path_accuracy": bool_metric(row["evidence_path_accuracy"] for row in rows),
        "correct_abstention_rate": bool_metric(row["correct_abstention"] for row in rows),
        "exact_fact_accuracy": bool_metric(row["exact_fact_accuracy"] for row in rows),
        "unit_accuracy": bool_metric(row["unit_accuracy"] for row in rows),
        "source_precision": {
            "mean": round(statistics.mean(source_precisions), 8) if source_precisions else None,
            "count": len(source_precisions),
        },
        "source_recall": {
            "mean": round(statistics.mean(source_recalls), 8) if source_recalls else None,
            "count": len(source_recalls),
        },
        "unsupported_claim_rate": bool_metric(row["unsupported_claim"] for row in rows),
        "answer_status_accuracy": bool_metric(row["answer_status_accuracy"] for row in rows),
        "retrieval_latency_ms": {
            "count": len(retrieval_latency),
            "p50": percentile(retrieval_latency, 50),
            "p95": percentile(retrieval_latency, 95),
            "mean": round(statistics.mean(retrieval_latency), 6) if retrieval_latency else None,
        },
        "compression": {
            "count": len(compression_ratios),
            "mean_full_context_tokens": round(statistics.mean(full_tokens), 6) if full_tokens else None,
            "mean_evidence_packet_tokens": round(statistics.mean(packet_tokens), 6) if packet_tokens else None,
            "mean_compression_ratio": round(statistics.mean(compression_ratios), 8) if compression_ratios else None,
            "required_evidence_retention": bool_metric(
                row["required_evidence_retained"] for row in rows
            ),
        },
    }


def rows_for(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = {}
    for case_class in sorted({row["case_class"] for row in rows}):
        by_class[case_class] = summarize_rows(
            rows_for(rows, lambda row, value=case_class: row["case_class"] == value)
        )
    return {
        "overall": summarize_rows(rows),
        "clean": summarize_rows(rows_for(rows, lambda row: not row["is_dense_overlap"])),
        "dense_overlap": summarize_rows(rows_for(rows, lambda row: row["is_dense_overlap"])),
        "holdout": summarize_rows(rows_for(rows, lambda row: row["is_holdout"])),
        "non_holdout": summarize_rows(rows_for(rows, lambda row: not row["is_holdout"])),
        "by_case_class": by_class,
    }


def paired_changes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]]["dense" if row["is_dense_overlap"] else "clean"] = row
    output = {}
    for pair_id in sorted(pairs):
        clean = pairs[pair_id].get("clean", {})
        dense = pairs[pair_id].get("dense", {})
        output[pair_id] = {
            "clean_case_id": clean.get("case_id"),
            "dense_case_id": dense.get("case_id"),
            "strict_applicability_clean": clean.get("strict_applicability"),
            "strict_applicability_dense": dense.get("strict_applicability"),
            "strict_applicability_change": (
                int(bool(dense.get("strict_applicability")))
                - int(bool(clean.get("strict_applicability")))
                if clean and dense and clean.get("expected_status") == "answer"
                else None
            ),
            "candidate_rank_clean": (
                1.0 / clean["reciprocal_rank"]
                if clean.get("reciprocal_rank")
                else None
            ),
            "candidate_rank_dense": (
                1.0 / dense["reciprocal_rank"]
                if dense.get("reciprocal_rank")
                else None
            ),
        }
    return output
