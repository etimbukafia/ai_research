"""Build compact evidence packets and measure context reduction."""

from __future__ import annotations

import json
import re
import sysconfig
from typing import Any

# This file has the planned name ``compression.py``.  Python's standard
# library also has a package named ``compression``.  Expose that package's
# submodule path so imports such as ``bz2 -> compression._common`` keep
# working when this experiment directory is first on ``sys.path``.
__path__ = [str(__import__("pathlib").Path(sysconfig.get_path("stdlib")) / "compression")]


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)


def token_count(text: str) -> int:
    """Return a stable word-token proxy used for local comparisons."""

    return len(TOKEN_RE.findall(text.lower()))


def full_context(
    candidate_ids: list[str], documents_by_id: dict[str, dict[str, Any]]
) -> str:
    chunks = []
    for document_id in candidate_ids:
        document = documents_by_id[document_id]
        chunks.append(
            f"SOURCE {document_id}\nTITLE: {document['title']}\n{document['body']}"
        )
    return "\n\n".join(chunks)


def evidence_packet(
    case: dict[str, Any], document: dict[str, Any], evidence_path: list[str]
) -> dict[str, Any]:
    fact = next(
        fact for fact in document["facts"] if fact["name"] == case["required_fact"]["name"]
    )
    product_id = document["product_ids"][0]
    firmware_version = document["firmware_versions"][0]
    region_id = document["regions"][0]
    return {
        "fact": {
            "name": fact["name"],
            "value": fact["value"],
            "unit": fact["unit"],
        },
        "scope": {
            "product": product_id,
            "firmware": firmware_version,
            "region": region_id,
        },
        "validity": {
            "status": document["status"].lower(),
            "effective_at": case["question_time"],
            "valid_from": document["valid_from"],
            "valid_to": document["valid_to"],
        },
        "authority": document["authority_name"],
        "evidence_path": evidence_path,
        "source_ids": [document["document_id"]],
    }


def packet_text(packet: dict[str, Any]) -> str:
    return json.dumps(packet, sort_keys=True, separators=(",", ":"))


def required_evidence_retained(
    packet: dict[str, Any] | None, case: dict[str, Any]
) -> bool | None:
    if packet is None:
        return None
    expected_document_id = case["applicable_document_id"]
    if expected_document_id is None:
        return None
    expected_fact = case["required_fact"]
    return (
        packet.get("source_ids") == [expected_document_id]
        and packet.get("fact", {}).get("name") == expected_fact["name"]
        and packet.get("fact", {}).get("value") == expected_fact["value"]
        and packet.get("fact", {}).get("unit") == expected_fact["unit"]
        and packet.get("scope", {}).get("firmware") == case["required_scope"]["firmware_version"]
        and packet.get("scope", {}).get("region") == case["required_scope"]["region"]
        and packet.get("evidence_path") == case["required_evidence_path"]
    )


def measure_compression(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_ids = [
        item["document_id"] for item in retrieval["candidate_documents"]
    ]
    full_text = full_context(candidate_ids, documents_by_id)
    selected_id = retrieval.get("selected_document_id")
    packet = None
    if selected_id is not None:
        packet = evidence_packet(
            case,
            documents_by_id[selected_id],
            retrieval.get("selected_evidence_path", []),
        )
    packet_string = packet_text(packet) if packet is not None else ""
    full_tokens = token_count(full_text)
    packet_tokens = token_count(packet_string)
    ratio = None
    if full_tokens:
        ratio = round(1.0 - packet_tokens / full_tokens, 8) if packet is not None else None
    return {
        "case_id": case["case_id"],
        "full_context_tokens": full_tokens,
        "evidence_packet_tokens": packet_tokens if packet is not None else None,
        "compression_ratio": ratio,
        "required_evidence_retained": required_evidence_retained(packet, case),
        "packet": packet,
    }
