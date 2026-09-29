"""Run the Gemini relation-extraction audit.

The audit reads ``GOOGLE_API_KEY`` through the provider only.  It never writes
or prints the value.  A missing key produces a truthful ``not_run`` artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from build_graph import EDGE_TYPES, NODE_TYPES
from retrieval import graph_from_export, load_json


MODEL_NAME = "google:gemini-3.5-flash-lite"
INTERVAL_SECONDS = 5.0
AUDIT_DOCUMENT_COUNT = 16


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractedEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_node_id: str
    target_node_id: str
    edge_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)


def proposal_prompt(document: dict[str, Any]) -> str:
    schema = {
        "node_types": sorted(NODE_TYPES),
        "edge_types": sorted(EDGE_TYPES),
        "required_edge_attributes": [
            "source_id",
            "valid_from",
            "valid_to",
            "status",
            "authority",
            "extraction_method",
            "confidence",
        ],
    }
    return (
        "Extract only entities and relations stated by this enterprise document. "
        "Use stable IDs that match the supplied IDs when possible. Keep the exact "
        "number and unit. Every edge must include source_id, validity, status, "
        "authority, extraction_method, and confidence. Return only the typed JSON "
        f"object. Graph schema: {json.dumps(schema, sort_keys=True)}\n"
        f"Document: {json.dumps(document, sort_keys=True)}"
    )


def select_audit_documents(
    documents: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Select four documents for each case class without tuning on results."""

    by_class: dict[str, list[str]] = {}
    globally_selected: set[str] = set()
    for case_class in sorted({case["case_class"] for case in cases}):
        selected: list[str] = []
        class_cases = [case for case in cases if case["case_class"] == case_class]
        for case in class_cases:
            values = []
            if case["applicable_document_id"]:
                values.append(case["applicable_document_id"])
            values.extend(case["forbidden_document_ids"])
            for document_id in values:
                if document_id not in selected and document_id not in globally_selected:
                    selected.append(document_id)
                if len(selected) == 4:
                    break
            if len(selected) == 4:
                break
        if len(selected) < 4:
            for document in documents:
                document_id = document["document_id"]
                if document_id not in selected and document_id not in globally_selected:
                    selected.append(document_id)
                if len(selected) == 4:
                    break
        if len(selected) != 4:
            raise AssertionError(f"Could not select four audit documents for {case_class}")
        by_class[case_class] = selected
        globally_selected.update(selected)
    all_ids = [document_id for ids in by_class.values() for document_id in ids]
    known_ids = {document["document_id"] for document in documents}
    if len(all_ids) != AUDIT_DOCUMENT_COUNT or len(set(all_ids)) != AUDIT_DOCUMENT_COUNT or not set(all_ids) <= known_ids:
        raise AssertionError("Relation audit must select 16 known documents")
    return by_class


def graph_slice(graph: Any, document_id: str) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    document_node = f"document-{document_id}"
    touched = {document_node}
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("source_id") == document_id and (
            source == document_node or target == document_node
        ):
            touched.add(source)
            touched.add(target)
    nodes = {
        (node_id, attrs.get("type", ""))
        for node_id, attrs in graph.nodes(data=True)
        if node_id in touched
    }
    edges = {
        (source, target, attrs.get("type", ""))
        for source, target, attrs in graph.edges(data=True)
        if attrs.get("source_id") == document_id
        and (source in touched or target in touched)
    }
    return nodes, edges


def validate_proposal(proposal: ExtractionProposal) -> tuple[bool, str]:
    node_ids = set()
    for node in proposal.nodes:
        if node.node_type not in NODE_TYPES:
            return False, "unknown_node_type"
        if node.node_id in node_ids:
            return False, "duplicate_node_id"
        node_ids.add(node.node_id)
    for edge in proposal.edges:
        if edge.edge_type not in EDGE_TYPES:
            return False, "unknown_edge_type"
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            return False, "edge_references_unknown_node"
        if not edge.attributes.get("source_id"):
            return False, "edge_missing_source_id"
        if bool(edge.attributes.get("valid_from")) != bool(edge.attributes.get("valid_to")):
            return False, "edge_missing_validity_range"
    return True, "accepted"


def compare_proposal(
    proposal: ExtractionProposal, graph: Any, document_id: str
) -> dict[str, Any]:
    expected_nodes, expected_edges = graph_slice(graph, document_id)
    proposed_nodes = {(node.node_id, node.node_type) for node in proposal.nodes}
    proposed_edges = {
        (edge.source_node_id, edge.target_node_id, edge.edge_type)
        for edge in proposal.edges
    }
    node_intersection = expected_nodes & proposed_nodes
    edge_intersection = expected_edges & proposed_edges
    scope_types = {"Product", "FirmwareVersion", "Region"}
    expected_scope = {item for item in expected_nodes if item[1] in scope_types}
    proposed_scope = {item for item in proposed_nodes if item[1] in scope_types}

    gold_edge_attributes: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source, target, attributes in graph.edges(data=True):
        key = (source, target, attributes.get("type", ""))
        if attributes.get("source_id") == document_id:
            gold_edge_attributes[key] = attributes

    matched_edges = [
        edge
        for edge in proposal.edges
        if (edge.source_node_id, edge.target_node_id, edge.edge_type)
        in gold_edge_attributes
    ]

    def matched_field_accuracy(fields: tuple[str, ...]) -> float:
        if not matched_edges:
            return 0.0
        correct = 0
        total = 0
        for edge in matched_edges:
            key = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            gold_attributes = gold_edge_attributes[key]
            for field in fields:
                total += 1
                if edge.attributes.get(field) == gold_attributes.get(field):
                    correct += 1
        return correct / total if total else 0.0

    return {
        "node": {
            "correct": len(node_intersection),
            "predicted": len(proposed_nodes),
            "gold": len(expected_nodes),
            "precision": len(node_intersection) / len(proposed_nodes) if proposed_nodes else 0.0,
            "recall": len(node_intersection) / len(expected_nodes) if expected_nodes else 0.0,
        },
        "edge": {
            "correct": len(edge_intersection),
            "predicted": len(proposed_edges),
            "gold": len(expected_edges),
            "precision": len(edge_intersection) / len(proposed_edges) if proposed_edges else 0.0,
            "recall": len(edge_intersection) / len(expected_edges) if expected_edges else 0.0,
        },
        "scope_field_accuracy": (
            len(expected_scope & proposed_scope) / len(expected_scope)
            if expected_scope
            else 1.0
        ),
        "validity_field_accuracy": matched_field_accuracy(("valid_from", "valid_to")),
        "authority_accuracy": matched_field_accuracy(("authority",)),
    }


async def call_extraction_agent(document: dict[str, Any]) -> ExtractionProposal:
    from pydantic_ai import Agent

    agent = Agent(
        MODEL_NAME,
        output_type=ExtractionProposal,
        system_prompt=(
            "You extract graph data from enterprise documents. "
            "Use only the document text and return the requested typed object."
        ),
    )
    result = await agent.run(proposal_prompt(document))
    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "data", None)
    if isinstance(output, ExtractionProposal):
        return output
    return ExtractionProposal.model_validate(output)


def empty_result(
    status: str,
    reason: str,
    selected_by_class: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": status,
        "reason": reason,
        "model": MODEL_NAME,
        "interval_seconds": INTERVAL_SECONDS,
        "document_count": AUDIT_DOCUMENT_COUNT,
        "selected_documents_by_case_class": selected_by_class,
        "request_count": 0,
        "error_count": 0,
        "review_count": 0,
        "metrics": {
            "node_precision": None,
            "node_recall": None,
            "edge_precision": None,
            "edge_recall": None,
            "scope_field_accuracy": None,
            "validity_field_accuracy": None,
            "authority_accuracy": None,
            "invalid_edge_rejection_rate": None,
            "review_rate": None,
        },
        "records": [],
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit(directory: Path) -> dict[str, Any]:
    documents = load_json(directory / "documents.json")["documents"]
    cases = load_json(directory / "cases.json")["cases"]
    graph = graph_from_export(load_json(directory / "graph.json"))
    selected_by_class = select_audit_documents(documents, cases)
    selected_ids = [document_id for ids in selected_by_class.values() for document_id in ids]
    by_id = {document["document_id"]: document for document in documents}
    output_path = directory / "extraction_results.json"

    if not os.environ.get("GOOGLE_API_KEY"):
        result = empty_result(
            "not_run",
            "GOOGLE_API_KEY is not set; relation extraction was not run.",
            selected_by_class,
        )
        write_result(output_path, result)
        return result

    records: list[dict[str, Any]] = []
    last_request = 0.0
    for document_id in selected_ids:
        wait = INTERVAL_SECONDS - (time.monotonic() - last_request)
        if last_request and wait > 0:
            time.sleep(wait)
        last_request = time.monotonic()
        document = by_id[document_id]
        started = time.perf_counter()
        try:
            proposal = asyncio.run(call_extraction_agent(document))
            valid, validation_reason = validate_proposal(proposal)
            comparison = compare_proposal(proposal, graph, document_id)
            records.append(
                {
                    "document_id": document_id,
                    "status": "accepted" if valid else "review",
                    "validation_reason": validation_reason,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
                    "comparison": comparison,
                }
            )
        except Exception:
            # Do not serialize provider messages.  They can contain request
            # metadata that is not useful for the result contract.
            records.append(
                {
                    "document_id": document_id,
                    "status": "error",
                    "error_type": "provider_or_validation_error",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
                }
            )
            break

    accepted = [record for record in records if record["status"] == "accepted"]
    review_count = sum(record["status"] == "review" for record in records)
    error_count = sum(record["status"] == "error" for record in records)
    comparisons = [record["comparison"] for record in records if record.get("comparison")]

    def mean_metric(path: tuple[str, str]) -> float | None:
        values = [float(item[path[0]][path[1]]) for item in comparisons]
        return round(sum(values) / len(values), 8) if values else None

    def mean_value(field: str) -> float | None:
        values = [float(item[field]) for item in comparisons]
        return round(sum(values) / len(values), 8) if values else None

    result = {
        "schema_version": "1.0",
        "status": "completed" if len(records) == len(selected_ids) and not error_count else "partial",
        "reason": None if len(records) == len(selected_ids) else "The provider run stopped after an error.",
        "model": MODEL_NAME,
        "interval_seconds": INTERVAL_SECONDS,
        "document_count": AUDIT_DOCUMENT_COUNT,
        "selected_documents_by_case_class": selected_by_class,
        "request_count": len(records),
        "error_count": error_count,
        "review_count": review_count,
        "metrics": {
            "node_precision": mean_metric(("node", "precision")),
            "node_recall": mean_metric(("node", "recall")),
            "edge_precision": mean_metric(("edge", "precision")),
            "edge_recall": mean_metric(("edge", "recall")),
            "scope_field_accuracy": mean_value("scope_field_accuracy"),
            "validity_field_accuracy": mean_value("validity_field_accuracy"),
            "authority_accuracy": mean_value("authority_accuracy"),
            "invalid_edge_rejection_rate": (
                round(review_count / review_count, 8) if review_count else None
            ),
            "review_rate": round(review_count / len(records), 8) if records else None,
        },
        "records": records,
    }
    write_result(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    result = audit(args.directory)
    print(f"Relation extraction audit: {result['status']} ({result['request_count']} requests).")


if __name__ == "__main__":
    main()
