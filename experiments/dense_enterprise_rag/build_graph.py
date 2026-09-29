"""Create the controlled documents, evaluation cases, and gold graph.

The experiment uses a small, local graph.  The data is synthetic by design.
The generator keeps the overlap and the scope errors under test explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


SEED = 20260828
QUESTION_DATE = "2026-08-28"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PRODUCTS = [
    {"id": "atlas_x100", "name": "Atlas X100", "family": "Atlas X"},
    {"id": "atlas_x200", "name": "Atlas X200", "family": "Atlas X"},
    {"id": "atlas_x210", "name": "Atlas X210", "family": "Atlas X"},
    {"id": "atlas_y200", "name": "Atlas Y200", "family": "Atlas Y"},
]
VERSIONS = ["3.8", "4.1", "4.2", "4.3"]
REGIONS = [
    {"id": "global", "name": "Global"},
    {"id": "europe", "name": "Europe"},
    {"id": "north_america", "name": "North America"},
]
AUTHORITIES = [
    {"id": "product_safety", "name": "Product Safety"},
    {"id": "engineering", "name": "Engineering"},
    {"id": "support", "name": "Support"},
    {"id": "sales", "name": "Sales"},
]
STATUSES = ["Approved", "Draft", "Expired", "Superseded"]
NODE_TYPES = {
    "Product",
    "FirmwareVersion",
    "Region",
    "Document",
    "Specification",
    "Authority",
    "Status",
    "Fact",
}
EDGE_TYPES = {
    "HAS_FIRMWARE",
    "SOLD_IN",
    "APPLIES_TO",
    "VALID_IN",
    "EFFECTIVE_DURING",
    "HAS_STATUS",
    "ISSUED_BY",
    "SUPERSEDES",
    "DEFINES",
    "SUPPORTED_BY",
}

RESERVED_SPEC_IDS = {
    ("atlas_x200", "3.8", "global"): "spec-12",
    ("atlas_x200", "4.2", "europe"): "spec-17",
    ("atlas_x200", "4.3", "europe"): "spec-19",
    ("atlas_x210", "4.2", "europe"): "spec-27",
}
RESERVED_BULLETIN_IDS = {
    ("atlas_x200", "4.2", "europe"): "bulletin-08",
}


def product_by_id(product_id: str) -> dict[str, str]:
    return next(product for product in PRODUCTS if product["id"] == product_id)


def region_by_id(region_id: str) -> dict[str, str]:
    return next(region for region in REGIONS if region["id"] == region_id)


def version_node_id(version: str) -> str:
    return f"firmware-{version.replace('.', '-')}"


def product_node_id(product_id: str) -> str:
    return f"product-{product_id.replace('_', '-')}"


def region_node_id(region_id: str) -> str:
    return f"region-{region_id.replace('_', '-')}"


def authority_node_id(authority_id: str) -> str:
    return f"authority-{authority_id.replace('_', '-')}"


def status_node_id(status: str) -> str:
    return f"status-{status.lower()}"


def fact_node_id(document_id: str, fact_name: str) -> str:
    return f"fact-{document_id}-{fact_name.replace('_', '-')}"


def date_range(version: str) -> tuple[str, str]:
    return {
        "3.8": ("2022-01-01", "2025-01-01"),
        "4.1": ("2024-01-01", "2025-06-30"),
        "4.2": ("2025-01-01", "2027-01-01"),
        "4.3": ("2026-09-01", "2027-12-31"),
    }[version]


def status_for_version(version: str) -> str:
    return {
        "3.8": "Superseded",
        "4.1": "Expired",
        "4.2": "Approved",
        "4.3": "Draft",
    }[version]


def specification_authority(product_id: str) -> str:
    # Product Safety owns X100 and X200 pressure limits.  Engineering owns the
    # other two models.  Support and Sales remain non-authoritative sources.
    if product_id in {"atlas_x100", "atlas_x200"}:
        return "product_safety"
    return "engineering"


def bulletin_authority(region_id: str) -> str:
    if region_id == "north_america":
        return "sales"
    return "support"


def pressure_value(product_id: str, version: str, region_id: str) -> int:
    overrides = {
        ("atlas_x200", "3.8", "global"): 250,
        ("atlas_x200", "4.2", "europe"): 220,
        ("atlas_x200", "4.3", "europe"): 230,
        ("atlas_x210", "4.2", "europe"): 240,
    }
    if (product_id, version, region_id) in overrides:
        return overrides[(product_id, version, region_id)]
    base = {
        "atlas_x100": 180,
        "atlas_x200": 210,
        "atlas_x210": 230,
        "atlas_y200": 205,
    }[product_id]
    version_delta = {"3.8": 20, "4.1": 10, "4.2": 0, "4.3": 10}[version]
    region_delta = {"global": 0, "europe": 10, "north_america": 5}[region_id]
    return base + version_delta + region_delta


def flow_value(product_id: str, version: str, region_id: str) -> int:
    base = {
        "atlas_x100": 38,
        "atlas_x200": 48,
        "atlas_x210": 52,
        "atlas_y200": 44,
    }[product_id]
    version_delta = {"3.8": 4, "4.1": 2, "4.2": 0, "4.3": 3}[version]
    region_delta = {"global": 0, "europe": 2, "north_america": 1}[region_id]
    return base + version_delta + region_delta


def allocate_id(prefix: str, used: set[str], limit: int = 48) -> str:
    for number in range(1, limit + 1):
        candidate = f"{prefix}-{number:02d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError(f"No free {prefix} identifier")


def make_document(
    document_id: str,
    product_id: str,
    version: str,
    region_id: str,
    document_type: str,
    status: str,
    authority_id: str,
    valid_from: str,
    valid_to: str,
    supersedes: list[str],
) -> dict[str, Any]:
    product = product_by_id(product_id)
    region = region_by_id(region_id)
    pressure = pressure_value(product_id, version, region_id)
    flow = flow_value(product_id, version, region_id)
    is_specification = document_type == "specification"
    role = "Specification" if is_specification else "Informational"
    authority = next(item["name"] for item in AUTHORITIES if item["id"] == authority_id)
    status_text = status.lower()
    title = (
        f"{product['name']} firmware {version} {region['name']} "
        f"{role.lower()} {document_id}"
    )
    supersession_text = (
        f" It supersedes {', '.join(supersedes)}."
        if supersedes
        else " It has no superseded source."
    )
    if is_specification:
        body = (
            f"{authority} specification {document_id} defines the operating limits "
            f"for {product['name']} with firmware {version} in {region['name']}. "
            f"The pressure limit is {pressure} kPa. The flow rate limit is "
            f"{flow} L/min. The document status is {status_text}. It is effective "
            f"from {valid_from} to {valid_to}. The source authority is {authority}."
            f"{supersession_text}"
        )
    else:
        body = (
            f"Support bulletin {document_id} records a field observation for "
            f"{product['name']} with firmware {version} in {region['name']}. "
            f"It reports a pressure limit of {pressure} kPa and a flow rate limit "
            f"of {flow} L/min. The bulletin is {status_text} and is effective from "
            f"{valid_from} to {valid_to}. The source authority is {authority}. "
            f"For {product['name']} with firmware {version} in {region['name']}, "
            f"the pressure limit applies as {pressure} kPa. "
            "This informational bulletin does not define the approved product "
            "specification."
        )
    facts = [
        {"name": "pressure_limit", "value": pressure, "unit": "kPa"},
        {"name": "flow_rate_limit", "value": flow, "unit": "L/min"},
    ]
    return {
        "document_id": document_id,
        "title": title,
        "body": body,
        "document_type": document_type,
        "document_role": role,
        "product_ids": [product_id],
        "firmware_versions": [version],
        "regions": [region_id],
        "status": status,
        "authority": authority_id,
        "authority_name": authority,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "supersedes": supersedes,
        "facts": facts,
        "specification_authority": is_specification,
        "extraction_method": "gold_template",
        "extraction_confidence": 1.0,
    }


def generate_documents() -> list[dict[str, Any]]:
    """Generate 48 specifications and 48 informational bulletins."""

    used_specs = set(RESERVED_SPEC_IDS.values())
    used_bulletins = set(RESERVED_BULLETIN_IDS.values())
    spec_ids: dict[tuple[str, str, str], str] = {}
    bulletin_ids: dict[tuple[str, str, str], str] = {}
    for product in PRODUCTS:
        for version in VERSIONS:
            for region in REGIONS:
                key = (product["id"], version, region["id"])
                spec_ids[key] = RESERVED_SPEC_IDS.get(key) or allocate_id(
                    "spec", used_specs
                )
                bulletin_ids[key] = RESERVED_BULLETIN_IDS.get(key) or allocate_id(
                    "bulletin", used_bulletins
                )

    documents: list[dict[str, Any]] = []
    for product in PRODUCTS:
        for version in VERSIONS:
            for region in REGIONS:
                key = (product["id"], version, region["id"])
                valid_from, valid_to = date_range(version)
                status = status_for_version(version)
                old_spec_id = spec_ids[(product["id"], "3.8", region["id"])]
                supersedes = [old_spec_id] if version == "4.2" else []
                if spec_ids[key] == "spec-17":
                    # Keep the named running case from the plan exact.  The
                    # current Europe source supersedes the older global source
                    # used as its controlled distractor.
                    supersedes = ["spec-12"]
                documents.append(
                    make_document(
                        spec_ids[key],
                        product["id"],
                        version,
                        region["id"],
                        "specification",
                        status,
                        specification_authority(product["id"]),
                        valid_from,
                        valid_to,
                        supersedes,
                    )
                )
                bulletin_from, bulletin_to = "2025-01-01", "2027-01-01"
                documents.append(
                    make_document(
                        bulletin_ids[key],
                        product["id"],
                        version,
                        region["id"],
                        "bulletin",
                        "Approved",
                        bulletin_authority(region["id"]),
                        bulletin_from,
                        bulletin_to,
                        [],
                    )
                )
    documents.sort(key=lambda document: document["document_id"])
    if len(documents) != 96 or len({doc["document_id"] for doc in documents}) != 96:
        raise AssertionError("Document generation must produce 96 unique documents")
    return documents


def document_lookup(documents: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for document in documents:
        lookup[
            (
                document["document_type"],
                document["product_ids"][0],
                document["firmware_versions"][0],
                document["regions"][0],
            )
        ] = document
    return lookup


def make_path(document: dict[str, Any]) -> list[str]:
    product_id = document["product_ids"][0]
    version = document["firmware_versions"][0]
    region_id = document["regions"][0]
    fact_id = fact_node_id(document["document_id"], "pressure_limit")
    return [
        product_node_id(product_id),
        version_node_id(version),
        region_node_id(region_id),
        f"document-{document['document_id']}",
        fact_id,
        authority_node_id(document["authority"]),
    ]


def make_answer_case(
    case_id: str,
    pair_id: str,
    case_class: str,
    dense: bool,
    holdout: bool,
    question: str,
    question_time: str,
    document: dict[str, Any] | None,
    *,
    product_id: str | None,
    firmware_version: str | None,
    region_id: str | None,
    product_candidates: list[str] | None = None,
    region_candidates: list[str] | None = None,
    expected_status: str = "answer",
    abstention_reason: str | None = None,
    hard_negative_ids: list[str] | None = None,
) -> dict[str, Any]:
    if document is not None:
        fact = next(fact for fact in document["facts"] if fact["name"] == "pressure_limit")
        required_scope = {
            "product_id": product_id,
            "firmware_version": firmware_version,
            "region": region_id,
            "authority": document["authority"],
            "document_type": "specification",
            "status": "Approved",
        }
        applicable_document_id = document["document_id"]
        required_fact = fact
        path = make_path(document)
        expected_answer = {"value": fact["value"], "unit": fact["unit"]}
    else:
        required_scope = {
            "product_id": product_id,
            "firmware_version": firmware_version,
            "region": region_id,
            "authority": "product_safety",
            "document_type": "specification",
            "status": "Approved",
        }
        applicable_document_id = None
        required_fact = {"name": "pressure_limit", "value": None, "unit": "kPa"}
        path = []
        expected_answer = None
    return {
        "case_id": case_id,
        "pair_id": pair_id,
        "case_class": case_class,
        "is_dense_overlap": dense,
        "is_holdout": holdout,
        "question": question,
        "question_time": question_time,
        "required_scope": required_scope,
        "product_candidates": product_candidates or [],
        "region_candidates": region_candidates or [],
        "applicable_document_id": applicable_document_id,
        "required_fact": required_fact,
        "required_evidence_path": path,
        "forbidden_document_ids": hard_negative_ids or [],
        "expected_answer_status": expected_status,
        "expected_answer": expected_answer,
        "abstention_reason": abstention_reason,
    }


def generate_cases(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = document_lookup(documents)

    def spec(product: str, version: str, region: str) -> dict[str, Any]:
        return lookup[("specification", product, version, region)]

    def bulletin(product: str, version: str, region: str) -> dict[str, Any]:
        return lookup[("bulletin", product, version, region)]

    cases: list[dict[str, Any]] = []

    answer_pairs = [
        (
            "version_and_supersession",
            "atlas_x200",
            "4.2",
            "europe",
            [spec("atlas_x200", "3.8", "global")["document_id"], spec("atlas_x200", "4.3", "europe")["document_id"], bulletin("atlas_x200", "4.2", "europe")["document_id"], spec("atlas_x210", "4.2", "europe")["document_id"]],
        ),
        (
            "version_and_supersession",
            "atlas_x100",
            "4.2",
            "global",
            [spec("atlas_x100", "3.8", "global")["document_id"], spec("atlas_x100", "4.3", "global")["document_id"], bulletin("atlas_x100", "4.2", "global")["document_id"]],
        ),
        (
            "version_and_supersession",
            "atlas_x210",
            "4.2",
            "europe",
            [spec("atlas_x210", "4.1", "europe")["document_id"], spec("atlas_x210", "4.3", "europe")["document_id"], bulletin("atlas_x210", "4.2", "europe")["document_id"]],
        ),
        (
            "version_and_supersession",
            "atlas_y200",
            "4.2",
            "north_america",
            [spec("atlas_y200", "3.8", "north_america")["document_id"], spec("atlas_y200", "4.3", "north_america")["document_id"], bulletin("atlas_y200", "4.2", "north_america")["document_id"]],
        ),
        (
            "region_and_entity_scope",
            "atlas_x200",
            "4.2",
            "europe",
            [spec("atlas_x200", "4.2", "global")["document_id"], bulletin("atlas_x200", "4.2", "europe")["document_id"]],
        ),
        (
            "region_and_entity_scope",
            "atlas_x100",
            "4.2",
            "north_america",
            [spec("atlas_x100", "4.2", "europe")["document_id"], bulletin("atlas_x100", "4.2", "north_america")["document_id"]],
        ),
        (
            "region_and_entity_scope",
            "atlas_x210",
            "4.2",
            "europe",
            [spec("atlas_x210", "4.2", "north_america")["document_id"], bulletin("atlas_x210", "4.2", "europe")["document_id"]],
        ),
        (
            "region_and_entity_scope",
            "atlas_y200",
            "4.2",
            "global",
            [spec("atlas_x200", "4.2", "global")["document_id"], bulletin("atlas_y200", "4.2", "global")["document_id"]],
        ),
        (
            "authority_and_status",
            "atlas_x200",
            "4.2",
            "europe",
            [bulletin("atlas_x200", "4.2", "europe")["document_id"], spec("atlas_x200", "4.3", "europe")["document_id"]],
        ),
        (
            "authority_and_status",
            "atlas_x100",
            "4.2",
            "global",
            [bulletin("atlas_x100", "4.2", "global")["document_id"], spec("atlas_x100", "4.3", "global")["document_id"]],
        ),
        (
            "authority_and_status",
            "atlas_x210",
            "4.2",
            "europe",
            [bulletin("atlas_x210", "4.2", "europe")["document_id"], spec("atlas_x210", "4.3", "europe")["document_id"]],
        ),
        (
            "authority_and_status",
            "atlas_y200",
            "4.2",
            "north_america",
            [bulletin("atlas_y200", "4.2", "north_america")["document_id"], spec("atlas_y200", "4.3", "north_america")["document_id"]],
        ),
    ]
    for index, (case_class, product_id, version, region_id, negatives) in enumerate(answer_pairs, 1):
        pair_id = f"pair-{index:02d}"
        document = spec(product_id, version, region_id)
        product_name = product_by_id(product_id)["name"]
        region_name = region_by_id(region_id)["name"]
        authority_name = document["authority_name"]
        clean_question = (
            f"What is the approved current pressure limit in the {authority_name} "
            f"specification for {product_name} in {region_name} with firmware {version}?"
        )
        dense_question = (
            f"What pressure limit applies to the {product_name} in {region_name} "
            f"with firmware {version}?"
        )
        cases.append(
            make_answer_case(
                f"case-{index:02d}-clean",
                pair_id,
                case_class,
                False,
                False,
                clean_question,
                QUESTION_DATE,
                document,
                product_id=product_id,
                firmware_version=version,
                region_id=region_id,
                hard_negative_ids=[],
            )
        )
        cases.append(
            make_answer_case(
                f"case-{index:02d}-dense",
                pair_id,
                case_class,
                True,
                False,
                dense_question,
                QUESTION_DATE,
                document,
                product_id=product_id,
                firmware_version=version,
                region_id=region_id,
                hard_negative_ids=negatives,
            )
        )

    holdout_specs = [
        (
            13,
            "missing_or_ambiguous_evidence",
            "What is the approved current pressure limit for Atlas X100 in Europe with firmware 5.0?",
            "What pressure limit applies to Atlas X100 in Europe with firmware 5.0?",
            {"product_id": "atlas_x100", "firmware_version": "5.0", "region_id": "europe"},
            "insufficient_evidence",
            "No approved specification exists for firmware 5.0.",
            [spec("atlas_x100", "4.2", "europe")["document_id"], spec("atlas_x100", "4.3", "europe")["document_id"], bulletin("atlas_x100", "4.2", "europe")["document_id"]],
        ),
        (
            14,
            "missing_or_ambiguous_evidence",
            "Which Atlas 200 product has the approved current pressure limit in Europe with firmware 4.2?",
            "What pressure limit applies to Atlas 200 in Europe with firmware 4.2?",
            {"product_id": None, "firmware_version": "4.2", "region_id": "europe"},
            "clarification",
            "Atlas X200 and Atlas Y200 both match the product name.",
            [spec("atlas_x200", "4.2", "europe")["document_id"], spec("atlas_y200", "4.2", "europe")["document_id"]],
        ),
        (
            15,
            "missing_or_ambiguous_evidence",
            "Which region has the approved current pressure limit for Atlas X200 with firmware 4.2?",
            "What pressure limit applies to Atlas X200 with firmware 4.2?",
            {"product_id": "atlas_x200", "firmware_version": "4.2", "region_id": None},
            "clarification",
            "The question does not name a region.",
            [spec("atlas_x200", "4.2", "global")["document_id"], spec("atlas_x200", "4.2", "europe")["document_id"], spec("atlas_x200", "4.2", "north_america")["document_id"]],
        ),
        (
            16,
            "missing_or_ambiguous_evidence",
            "What pressure limit applied to Atlas X200 in Europe with firmware 4.2 on 2024-01-01?",
            "What pressure limit applies to Atlas X200 in Europe with firmware 4.2 on 2024-01-01?",
            {"product_id": "atlas_x200", "firmware_version": "4.2", "region_id": "europe"},
            "insufficient_evidence",
            "The approved firmware 4.2 specification was not effective on 2024-01-01.",
            [spec("atlas_x200", "3.8", "global")["document_id"], spec("atlas_x200", "4.2", "europe")["document_id"], spec("atlas_x200", "4.3", "europe")["document_id"]],
        ),
    ]
    for index, case_class, clean_question, dense_question, scope, expected_status, reason, negatives in holdout_specs:
        pair_id = f"pair-{index:02d}"
        clean_document = None
        cases.append(
            make_answer_case(
                f"case-{index:02d}-clean",
                pair_id,
                case_class,
                False,
                True,
                clean_question,
                QUESTION_DATE if index != 16 else "2024-01-01",
                clean_document,
                product_id=scope["product_id"],
                firmware_version=scope["firmware_version"],
                region_id=scope["region_id"],
                product_candidates=["atlas_x200", "atlas_y200"] if index == 14 else [],
                region_candidates=["global", "europe", "north_america"] if index == 15 else [],
                expected_status=expected_status,
                abstention_reason=reason,
                hard_negative_ids=[],
            )
        )
        cases.append(
            make_answer_case(
                f"case-{index:02d}-dense",
                pair_id,
                case_class,
                True,
                True,
                dense_question,
                QUESTION_DATE if index != 16 else "2024-01-01",
                clean_document,
                product_id=scope["product_id"],
                firmware_version=scope["firmware_version"],
                region_id=scope["region_id"],
                product_candidates=["atlas_x200", "atlas_y200"] if index == 14 else [],
                region_candidates=["global", "europe", "north_america"] if index == 15 else [],
                expected_status=expected_status,
                abstention_reason=reason,
                hard_negative_ids=negatives,
            )
        )

    cases.sort(key=lambda case: case["case_id"])
    if len(cases) != 32:
        raise AssertionError("Case generation must produce 32 cases")
    if len({case["pair_id"] for case in cases}) != 16:
        raise AssertionError("Case generation must produce 16 pairs")
    if sum(case["is_holdout"] for case in cases) != 8:
        raise AssertionError("Case generation must produce 8 holdout cases")
    return cases


def edge_attributes(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": document["document_id"],
        "valid_from": document["valid_from"],
        "valid_to": document["valid_to"],
        "status": document["status"],
        "authority": document["authority"],
        "extraction_method": document["extraction_method"],
        "confidence": document["extraction_confidence"],
    }


def build_graph(documents: list[dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(name="dense_enterprise_rag_gold_graph")
    for product in PRODUCTS:
        graph.add_node(
            product_node_id(product["id"]),
            type="Product",
            product_id=product["id"],
            name=product["name"],
            family=product["family"],
        )
    for version in VERSIONS:
        graph.add_node(version_node_id(version), type="FirmwareVersion", version=version)
    for region in REGIONS:
        graph.add_node(
            region_node_id(region["id"]),
            type="Region",
            region_id=region["id"],
            name=region["name"],
        )
    for authority in AUTHORITIES:
        graph.add_node(
            authority_node_id(authority["id"]),
            type="Authority",
            authority_id=authority["id"],
            name=authority["name"],
        )
    for status in STATUSES:
        graph.add_node(status_node_id(status), type="Status", status=status)

    for document in documents:
        document_id = document["document_id"]
        document_node = f"document-{document_id}"
        graph.add_node(
            document_node,
            type="Document",
            document_id=document_id,
            document_type=document["document_type"],
            document_role=document["document_role"],
            status=document["status"],
            authority=document["authority"],
            valid_from=document["valid_from"],
            valid_to=document["valid_to"],
            product_ids=document["product_ids"],
            firmware_versions=document["firmware_versions"],
            regions=document["regions"],
            specification_authority=document["specification_authority"],
        )
        attrs = edge_attributes(document)
        product_id = document["product_ids"][0]
        version = document["firmware_versions"][0]
        region_id = document["regions"][0]
        graph.add_edge(product_node_id(product_id), document_node, type="APPLIES_TO", **attrs)
        graph.add_edge(version_node_id(version), document_node, type="APPLIES_TO", **attrs)
        graph.add_edge(document_node, region_node_id(region_id), type="VALID_IN", **attrs)
        graph.add_edge(document_node, status_node_id(document["status"]), type="HAS_STATUS", **attrs)
        graph.add_edge(document_node, authority_node_id(document["authority"]), type="ISSUED_BY", **attrs)
        graph.add_edge(product_node_id(product_id), version_node_id(version), type="HAS_FIRMWARE", **attrs)
        graph.add_edge(product_node_id(product_id), region_node_id(region_id), type="SOLD_IN", **attrs)
        for fact in document["facts"]:
            node_id = fact_node_id(document_id, fact["name"])
            graph.add_node(
                node_id,
                type="Fact",
                fact_name=fact["name"],
                value=fact["value"],
                unit=fact["unit"],
                authority=document["authority"],
                source_id=document_id,
            )
            graph.add_edge(document_node, node_id, type="DEFINES", **attrs)
        if document["document_type"] == "specification":
            graph.add_node(
                f"specification-{document_id}",
                type="Specification",
                document_id=document_id,
                source_id=document_id,
            )
            graph.add_edge(
                document_node,
                f"specification-{document_id}",
                type="SUPPORTED_BY",
                **attrs,
            )

    by_id = {document["document_id"]: document for document in documents}
    for document in documents:
        for old_id in document["supersedes"]:
            if old_id not in by_id:
                raise ValueError(f"Unknown superseded document: {old_id}")
            graph.add_edge(
                f"document-{document['document_id']}",
                f"document-{old_id}",
                type="SUPERSEDES",
                **edge_attributes(document),
            )
    validate_graph(graph)
    return graph


def validate_graph(graph: nx.MultiDiGraph) -> None:
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type")
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type for {node_id}: {node_type}")
        if node_type == "Fact" and isinstance(attrs.get("value"), (int, float)):
            if not attrs.get("unit"):
                raise ValueError(f"Numeric fact has no unit: {node_id}")
            if not attrs.get("authority"):
                raise ValueError(f"Specification fact has no authority: {node_id}")
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_type = attrs.get("type")
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {edge_type}")
        if not attrs.get("source_id"):
            raise ValueError(f"Edge has no source_id: {source} -> {target}")
        if bool(attrs.get("valid_from")) != bool(attrs.get("valid_to")):
            raise ValueError(f"Time-varying edge has an incomplete range: {source} -> {target}")
    supersession = nx.DiGraph()
    supersession.add_edges_from(
        (source, target)
        for source, target, attrs in graph.edges(data=True)
        if attrs.get("type") == "SUPERSEDES"
    )
    cycle = next(nx.simple_cycles(supersession), None)
    if cycle:
        raise ValueError(f"SUPERSEDES cycle: {cycle}")


def graph_export(graph: nx.MultiDiGraph) -> dict[str, Any]:
    nodes = []
    for node_id, attrs in sorted(graph.nodes(data=True), key=lambda item: item[0]):
        nodes.append({"id": node_id, "attributes": attrs})
    edges = []
    for source, target, key, attrs in sorted(
        graph.edges(keys=True, data=True),
        key=lambda item: (item[0], item[1], item[3].get("type", ""), str(item[2])),
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "key": str(key),
                "type": attrs["type"],
                "attributes": {name: value for name, value in attrs.items() if name != "type"},
            }
        )
    return {
        "schema_version": "1.0",
        "graph_name": "dense_enterprise_rag_gold_graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def schema_export() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "node_types": {
            "Product": ["product_id", "name", "family"],
            "FirmwareVersion": ["version"],
            "Region": ["region_id", "name"],
            "Document": ["document_id", "document_type", "status", "authority", "valid_from", "valid_to"],
            "Specification": ["document_id", "source_id"],
            "Authority": ["authority_id", "name"],
            "Status": ["status"],
            "Fact": ["fact_name", "value", "unit", "authority", "source_id"],
        },
        "edge_types": sorted(EDGE_TYPES),
        "edge_required_attributes": [
            "source_id",
            "valid_from",
            "valid_to",
            "status",
            "authority",
            "extraction_method",
            "confidence",
        ],
        "validation_rules": [
            "Reject unknown node or edge types.",
            "Reject edges without source_id.",
            "Reject time-varying edges without valid_from and valid_to.",
            "Reject numeric facts without a unit.",
            "Reject specification facts without an authority.",
            "Reject SUPERSEDES cycles.",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_generated(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = generate_documents()
    cases = generate_cases(documents)
    graph = build_graph(documents)
    write_json(output_dir / "documents.json", {"seed": SEED, "documents": documents})
    write_json(output_dir / "cases.json", {"seed": SEED, "cases": cases})
    write_json(output_dir / "graph_schema.json", schema_export())
    write_json(output_dir / "graph.json", graph_export(graph))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated JSON artifacts.",
    )
    args = parser.parse_args()
    write_generated(args.output_dir)
    print("Generated 96 documents, 32 cases, and the validated gold graph.")


if __name__ == "__main__":
    main()
