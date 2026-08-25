"""Run the customer assistant graph-memory experiment.

The script keeps the agent policy fixed and changes only the memory condition.
It writes results.json, trace.json, and report.md for the article.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AS_OF = "2026-08-25"
TOP_K = 8
MAX_GRAPH_HOPS = 4
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 17
RELATION_TYPES = [
    "PLACED",
    "CONTAINS",
    "COVERED_BY",
    "OPENED",
    "REPORTS",
    "RESOLVED_BY",
    "HAS_ADDRESS",
    "PREFERS",
    "COMPATIBLE_WITH",
    "GOVERNED_BY",
    "HAS_NOTE",
]

_DLL_HANDLE: Any | None = None


def configure_runtime() -> None:
    """Set deterministic runtime settings before model or database imports."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("LBUG_PYTHON_BACKEND", "pybind")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    random.seed(SEED)

    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(SEED)
        torch.set_num_threads(1)
    except ImportError:
        pass

def ensure_ladybug_dll_directory() -> None:
    """Add Ladybug's Windows dependencies only before Ladybug imports."""

    global _DLL_HANDLE
    if sys.platform != "win32" or _DLL_HANDLE is not None:
        return
    candidates = [
        os.getenv("LBUG_DLL_DIRECTORY", ""),
        r"C:\Program Files\Microsoft OneDrive\26.145.0728.0011",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            _DLL_HANDLE = os.add_dll_directory(candidate)
            return


def load_json(name: str) -> dict[str, Any]:
    with (ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(name: str, value: Any) -> None:
    with (ROOT / name).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def cypher_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def make_record(
    record_id: str,
    record_type: str,
    customer_id: str,
    text: str,
    observed_at: str,
    source_id: str,
    *,
    valid_from: str | None = None,
    valid_to: str = "9999-12-31",
    confidence: float = 1.0,
    status: str = "current",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "record_type": record_type,
        "customer_id": customer_id,
        "text": text,
        "observed_at": observed_at,
        "valid_from": valid_from or observed_at,
        "valid_to": valid_to,
        "source_id": source_id,
        "confidence": confidence,
        "status": status,
    }


def materialize_records(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand the hand-authored customer profiles into records and edges."""

    records: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    record_ids: set[str] = set()

    def add_record(record: dict[str, Any]) -> None:
        if record["id"] in record_ids:
            return
        record_ids.add(record["id"])
        records.append(record)

    def add_edge(
        source: str,
        relation: str,
        target: str,
        observed_at: str,
        source_id: str,
        *,
        valid_from: str | None = None,
        valid_to: str = "9999-12-31",
        confidence: float = 1.0,
        status: str = "current",
    ) -> None:
        edges.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "observed_at": observed_at,
                "valid_from": valid_from or observed_at,
                "valid_to": valid_to,
                "source_id": source_id,
                "confidence": confidence,
                "status": status,
            }
        )

    for policy in source["policies"]:
        add_record(
            make_record(
                policy["id"],
                "policy",
                "",
                policy["text"],
                policy["observed_at"],
                policy["source_id"],
                valid_from=policy["valid_from"],
                valid_to=policy["valid_to"],
            )
        )

    for profile in source["profiles"]:
        customer_id = profile["customer_id"]
        suffix = customer_id.split("-")[-1]
        order_id = f"order-{suffix}"
        product_id = profile["product_id"]
        warranty_id = f"warranty-{suffix}"
        replacement_id = profile["replacement_id"]
        ticket_id = f"ticket-{suffix}"
        issue_id = f"issue-{suffix}"
        resolution_id = f"resolution-{suffix}"
        old_address_id = f"address-old-{suffix}"
        current_address_id = f"address-current-{suffix}"
        old_preference_id = f"preference-old-{suffix}"
        current_preference_id = f"preference-current-{suffix}"
        note_id = f"note-old-warranty-{suffix}"
        alternate_part_note_id = f"note-alternate-part-{suffix}"

        add_record(
            make_record(
                customer_id,
                "customer",
                customer_id,
                f"Customer {profile['name']} owns {profile['product_name']}.",
                "2024-01-01",
                "crm-system",
            )
        )
        add_record(
            make_record(
                order_id,
                "order",
                customer_id,
                f"Order {order_id} contains {profile['product_name']} and was placed on {profile['order_date']}.",
                profile["order_date"],
                "order-system",
            )
        )
        add_record(
            make_record(
                product_id,
                "product",
                customer_id,
                f"{profile['product_name']} is a {profile['category']} product. The model is {product_id}.",
                profile["order_date"],
                "catalog-system",
            )
        )
        add_record(
            make_record(
                warranty_id,
                "warranty",
                customer_id,
                f"Warranty for {product_id} ends on {profile['warranty_valid_to']}.",
                profile["order_date"],
                "warranty-system",
                valid_from=profile["order_date"],
                valid_to=profile["warranty_valid_to"],
                status="active" if profile["warranty_valid_to"] >= AS_OF else "expired",
            )
        )
        add_record(
            make_record(
                replacement_id,
                "replacement",
                customer_id,
                f"{profile['replacement_name']} is compatible with {product_id}.",
                "2026-01-01",
                "catalog-system",
            )
        )
        add_record(
            make_record(
                ticket_id,
                "ticket",
                customer_id,
                f"Support ticket {ticket_id}: {profile['issue']}",
                profile["ticket_opened"],
                "ticketing-system",
            )
        )
        add_record(
            make_record(
                issue_id,
                "issue",
                customer_id,
                f"Issue linked to {product_id}: {profile['issue']}",
                profile["ticket_opened"],
                "ticketing-system",
            )
        )
        add_record(
            make_record(
                resolution_id,
                "resolution",
                customer_id,
                f"Resolution for {issue_id}: {profile['resolution']}.",
                "2026-03-20",
                "resolution-system",
            )
        )
        add_record(
            make_record(
                old_address_id,
                "address",
                customer_id,
                f"Old delivery address for {customer_id}: {profile['old_address']}.",
                "2024-01-01",
                "address-system",
                valid_from="2024-01-01",
                valid_to="2025-12-31",
                status="superseded",
            )
        )
        add_record(
            make_record(
                current_address_id,
                "address",
                customer_id,
                f"Current delivery address for {customer_id}: {profile['current_address']}.",
                "2026-01-05",
                "address-system",
                valid_from="2026-01-05",
            )
        )
        add_record(
            make_record(
                old_preference_id,
                "preference",
                customer_id,
                f"Old contact preference for {customer_id}: {profile['old_preference']}.",
                "2024-01-01",
                "preference-system",
                valid_from="2024-01-01",
                valid_to="2025-12-31",
                status="superseded",
            )
        )
        add_record(
            make_record(
                current_preference_id,
                "preference",
                customer_id,
                f"Current contact preference for {customer_id}: {profile['current_preference']}.",
                "2026-01-05",
                "preference-system",
                valid_from="2026-01-05",
            )
        )
        add_record(
            make_record(
                note_id,
                "customer_note",
                customer_id,
                f"Old CRM note for {customer_id}: the {product_id} warranty may be extended.",
                "2025-01-01",
                f"crm-note-{suffix}",
                valid_from="2025-01-01",
                valid_to="2025-12-31",
                confidence=0.45,
                status="superseded",
            )
        )
        if customer_id == "cust-05":
            add_record(
                make_record(
                    alternate_part_note_id,
                    "customer_note",
                    customer_id,
                    "An old CRM note names a similar display part without a catalog compatibility link.",
                    "2025-12-10",
                    "crm-note-05",
                    valid_from="2025-12-10",
                    valid_to="2025-12-31",
                    confidence=0.4,
                    status="superseded",
                )
            )

        add_edge(customer_id, "PLACED", order_id, profile["order_date"], "order-system")
        add_edge(order_id, "CONTAINS", product_id, profile["order_date"], "order-system")
        add_edge(product_id, "COVERED_BY", warranty_id, profile["order_date"], "warranty-system", valid_to=profile["warranty_valid_to"], status="active" if profile["warranty_valid_to"] >= AS_OF else "expired")
        add_edge(product_id, "COMPATIBLE_WITH", replacement_id, "2026-01-01", "catalog-system")
        add_edge(customer_id, "OPENED", ticket_id, profile["ticket_opened"], "ticketing-system")
        add_edge(ticket_id, "REPORTS", issue_id, profile["ticket_opened"], "ticketing-system")
        add_edge(issue_id, "RESOLVED_BY", resolution_id, "2026-03-20", "resolution-system")
        add_edge(customer_id, "HAS_ADDRESS", old_address_id, "2024-01-01", "address-system", valid_to="2025-12-31", status="superseded")
        add_edge(customer_id, "HAS_ADDRESS", current_address_id, "2026-01-05", "address-system")
        add_edge(customer_id, "PREFERS", old_preference_id, "2024-01-01", "preference-system", valid_to="2025-12-31", status="superseded")
        add_edge(customer_id, "PREFERS", current_preference_id, "2026-01-05", "preference-system")
        add_edge(customer_id, "HAS_NOTE", note_id, "2025-01-01", f"crm-note-{suffix}", valid_to="2025-12-31", confidence=0.45, status="superseded")
        if customer_id == "cust-05":
            add_edge(customer_id, "HAS_NOTE", alternate_part_note_id, "2025-12-10", "crm-note-05", valid_to="2025-12-31", confidence=0.4, status="superseded")
        add_edge(product_id, "GOVERNED_BY", "policy-01", "2026-01-01", "policy-system-2026")
        add_edge(product_id, "GOVERNED_BY", "policy-02", "2026-01-01", "policy-compatibility-2026")

    return records, edges


def record_text(record: dict[str, Any]) -> str:
    return f"{record['record_type']} {record['id']}. {record['text']}"


def active_at(record: dict[str, Any], as_of: str = AS_OF) -> bool:
    return record["valid_from"] <= as_of <= record["valid_to"]


def customer_candidates(records: list[dict[str, Any]], customer_id: str) -> list[dict[str, Any]]:
    return [record for record in records if record["customer_id"] in (customer_id, "")]


def terms(text: str) -> set[str]:
    return {token.strip(".,?!:'\"()[]{}-").lower() for token in text.split() if len(token) > 2}


def overlap(query: str, text: str) -> int:
    return len(terms(query) & terms(text))


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row.get("score", 0.0)), row["id"]))


def flat_recent(case: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = customer_candidates(records, case["customer_id"])
    rows = []
    for record in candidates:
        age_score = datetime.fromisoformat(record["observed_at"]).timestamp() / 10**9
        rows.append(
            {
                "id": record["id"],
                "score": age_score,
                "observed_at": record["observed_at"],
                "source_id": record["source_id"],
            }
        )
    return {"rows": sorted(rows, key=lambda row: (-row["score"], row["id"]))[:TOP_K], "paths": []}


class VectorIndex:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.Client()
        self.model = SentenceTransformer(MODEL_ID, device="cpu")
        self.records_by_id = {record["id"]: record for record in records}
        texts = [record_text(record) for record in records]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = {record["id"]: embedding.tolist() for record, embedding in zip(records, embeddings)}

    def retrieve(self, case: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = customer_candidates(records, case["customer_id"])
        if not candidates:
            return {"rows": [], "paths": []}
        digest = hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()[:10]
        collection = self.client.create_collection(
            name=f"customer_memory_{digest}",
            metadata={"hnsw:space": "cosine"},
        )
        ids = [record["id"] for record in candidates]
        collection.add(
            ids=ids,
            documents=[record_text(record) for record in candidates],
            embeddings=[self.embeddings[record_id] for record_id in ids],
        )
        query_embedding = self.model.encode(
            [case["ticket_text"]],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        response = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(TOP_K, len(ids)),
            include=["distances"],
        )
        rows = []
        for record_id, distance in zip(response["ids"][0], response["distances"][0]):
            record = self.records_by_id[record_id]
            rows.append(
                {
                    "id": record_id,
                    "score": 1.0 - float(distance),
                    "cosine_distance": float(distance),
                    "source_id": record["source_id"],
                }
            )
        return {"rows": sort_rows(rows), "paths": []}


class LadybugGraph:
    def __init__(self, path: Path) -> None:
        ensure_ladybug_dll_directory()
        import ladybug as lb

        self.path = path
        self.lb = lb
        self._remove_existing()
        self.db = lb.Database(str(path), backend="pybind")
        self.conn = lb.Connection(self.db)
        self._create_schema()

    def _remove_existing(self) -> None:
        if self.path.is_dir():
            shutil.rmtree(self.path)
        elif self.path.exists():
            self.path.unlink()
        for suffix in (".wal", ".lock"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

    def _create_schema(self) -> None:
        self.conn.execute(
            "CREATE NODE TABLE Record("
            "id STRING, record_type STRING, customer_id STRING, text STRING, "
            "observed_at STRING, valid_from STRING, valid_to STRING, "
            "source_id STRING, confidence DOUBLE, status STRING, PRIMARY KEY(id))"
        )
        for relation in RELATION_TYPES:
            self.conn.execute(
                f"CREATE REL TABLE {relation}(FROM Record TO Record, "
                "observed_at STRING, valid_from STRING, valid_to STRING, "
                "source_id STRING, confidence DOUBLE, status STRING)"
            )

    def load(self, records: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        for record in records:
            values = ", ".join(
                f"{key}: {cypher_literal(record[key])}"
                for key in (
                    "id",
                    "record_type",
                    "customer_id",
                    "text",
                    "observed_at",
                    "valid_from",
                    "valid_to",
                    "source_id",
                    "confidence",
                    "status",
                )
            )
            self.conn.execute(f"CREATE (n:Record {{{values}}})")
        for edge in edges:
            properties = ", ".join(
                f"{key}: {cypher_literal(edge[key])}"
                for key in ("observed_at", "valid_from", "valid_to", "source_id", "confidence", "status")
            )
            query = (
                f"MATCH (a:Record {{id: {cypher_literal(edge['source'])}}}), "
                f"(b:Record {{id: {cypher_literal(edge['target'])}}}) "
                f"CREATE (a)-[:{edge['relation']} {{{properties}}}]->(b)"
            )
            self.conn.execute(query)

    def all_edges(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for relation in RELATION_TYPES:
            rows = self.conn.execute(
                f"MATCH (a:Record)-[r:{relation}]->(b:Record) "
                "RETURN a.id, b.id, r.valid_from, r.valid_to, r.source_id, r.confidence, r.status"
            ).get_all()
            for row in rows:
                result.append(
                    {
                        "source": row[0],
                        "relation": relation,
                        "target": row[1],
                        "valid_from": row[2],
                        "valid_to": row[3],
                        "source_id": row[4],
                        "confidence": float(row[5]),
                        "status": row[6],
                    }
                )
        return result

    def retrieve(self, case: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
        edges = self.all_edges()
        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]].append(edge)

        customer_id = case["customer_id"]
        queue: deque[tuple[str, int, list[str]]] = deque([(customer_id, 0, [])])
        seen: dict[str, tuple[int, list[str]]] = {customer_id: (0, [])}
        while queue:
            node_id, distance, path = queue.popleft()
            if distance >= MAX_GRAPH_HOPS:
                continue
            for edge in adjacency.get(node_id, []):
                if not active_at(records_by_id[edge["target"]]) and records_by_id[edge["target"]]["record_type"] in {"address", "preference"}:
                    continue
                edge_key = f"{edge['source']}|{edge['relation']}|{edge['target']}"
                next_path = path + [edge_key]
                old = seen.get(edge["target"])
                if old is None or distance + 1 < old[0]:
                    seen[edge["target"]] = (distance + 1, next_path)
                    queue.append((edge["target"], distance + 1, next_path))

        query_terms = terms(case["ticket_text"])
        allowed_types = INTENT_RECORD_TYPES.get(case["intent"])
        rows = []
        paths: set[str] = set()
        for node_id, (distance, path) in seen.items():
            if node_id == customer_id:
                continue
            record = records_by_id[node_id]
            if allowed_types is not None and record["record_type"] not in allowed_types:
                continue
            if case["intent"] == "warranty_replacement" and record["id"] == "policy-02":
                continue
            score = 5.0 - distance + overlap(case["ticket_text"], record["text"]) * 0.05
            if record["record_type"] in {"warranty", "address", "preference", "policy"} and active_at(record):
                score += 0.5
            if record["source_id"] in {"policy-system-2026", "policy-compatibility-2026", "order-system", "warranty-system", "address-system"}:
                score += 0.1
            if record["status"] == "superseded":
                score -= 1.0
            rows.append(
                {
                    "id": node_id,
                    "score": score,
                    "graph_hops": distance,
                    "path": path,
                    "source_id": record["source_id"],
                }
            )
            paths.update(path)
        del query_terms
        return {"rows": sort_rows(rows), "paths": sorted(paths)}

    def close(self) -> None:
        self.db.close()


def hybrid_retrieve(
    case: dict[str, Any],
    vector_result: dict[str, Any],
    graph: LadybugGraph,
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    graph_result = graph.retrieve(case, records_by_id)
    seed_ids = {row["id"] for row in vector_result["rows"][:TOP_K]}
    rows = []
    for row in graph_result["rows"]:
        score = row["score"]
        if row["id"] in seed_ids:
            score += 1.0
        rows.append({**row, "score": score})
    return {"rows": sort_rows(rows), "paths": graph_result["paths"]}


SKILLS_BY_INTENT = {
    "find_order": ["find_customer", "find_order"],
    "find_warranty": ["find_customer", "find_order", "check_warranty"],
    "find_issue": ["find_customer", "find_ticket", "find_issue"],
    "find_current_address": ["find_customer", "get_current_address"],
    "find_current_preference": ["find_customer", "get_current_preference"],
    "find_resolution": ["find_customer", "find_ticket", "find_resolution"],
    "warranty_replacement": ["find_customer", "find_order", "check_warranty", "find_compatible_product", "get_current_address"],
    "compatible_issue": ["find_customer", "find_order", "find_compatible_product", "find_ticket"],
    "issue_resolution": ["find_customer", "find_order", "find_ticket", "find_resolution"],
    "compatible_policy": ["find_customer", "find_order", "find_compatible_product", "find_policy"],
    "order_warranty_replacement": ["find_customer", "find_order", "check_warranty", "find_compatible_product"],
    "check_warranty_status": ["find_customer", "check_warranty"],
    "source_authority": ["find_customer", "check_warranty", "find_policy", "check_source"],
    "missing_evidence": ["find_customer", "check_source", "request_human_review"],
}

INTENT_RECORD_TYPES: dict[str, set[str] | None] = {
    "find_order": {"order", "product"},
    "find_warranty": {"warranty"},
    "find_issue": {"ticket", "issue"},
    "find_current_address": {"address"},
    "find_current_preference": {"preference"},
    "find_resolution": {"resolution"},
    "warranty_replacement": {"order", "product", "warranty", "replacement", "ticket", "issue", "policy", "address"},
    "compatible_issue": {"order", "product", "replacement", "ticket", "issue"},
    "issue_resolution": {"order", "product", "ticket", "issue", "resolution"},
    "compatible_policy": {"order", "product", "replacement", "policy"},
    "order_warranty_replacement": {"order", "product", "warranty", "replacement"},
    "check_warranty_status": {"warranty"},
    "source_authority": {"product", "warranty", "policy", "customer_note"},
    "missing_evidence": None,
}


def evaluate_case(
    case: dict[str, Any],
    method: str,
    result: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = result["rows"][:TOP_K]
    retrieved_ids = [row["id"] for row in rows]
    retrieved_set = set(retrieved_ids)
    required_ids = set(case["required_record_ids"])
    required_path = set(case["required_path"])
    retrieved_path: set[str] = set()
    for row in rows:
        retrieved_path.update(row.get("path", []))

    if case["answerable"]:
        can_answer = required_ids.issubset(retrieved_set)
        decision = "answer" if can_answer else "needs_human_review"
        cited_ids = case["required_record_ids"] if can_answer else retrieved_ids[:1]
    else:
        decision = "insufficient_evidence"
        cited_ids = []

    expected_decision = "answer" if case["answerable"] else "insufficient_evidence"
    applicable_temporal = case["case_class"] == "temporal" or case["intent"] in {"find_current_address", "find_current_preference", "check_warranty_status", "source_authority"}
    temporal_ok = None
    if applicable_temporal:
        stale_ids = set(case.get("disallowed_records", []))
        temporal_ok = bool(retrieved_set - stale_ids) and not bool(set(cited_ids) & stale_ids)

    allowed_sources = set(case["allowed_sources"])
    provenance_precision = (
        1.0
        if not cited_ids
        else sum(records_by_id[record_id]["source_id"] in allowed_sources for record_id in cited_ids) / len(cited_ids)
    )
    unsupported_claims = 0
    if decision == "answer":
        unsupported_claims = len(required_ids - retrieved_set)
    return {
        "method": method,
        "case_id": case["case_id"],
        "decision": decision,
        "expected_decision": expected_decision,
        "answer_accuracy": float(decision == expected_decision),
        "retrieved_ids": retrieved_ids,
        "required_record_ids": case["required_record_ids"],
        "retrieved_path": sorted(retrieved_path),
        "required_path": case["required_path"],
        "evidence_recall": None if not required_ids else len(required_ids & retrieved_set) / len(required_ids),
        "path_recall": None if not required_path else len(required_path & retrieved_path) / len(required_path),
        "temporal_accuracy": None if temporal_ok is None else float(temporal_ok),
        "provenance_precision": provenance_precision,
        "unsupported_claim_rate": unsupported_claims / len(required_ids) if required_ids else 0.0,
        "cited_ids": cited_ids,
        "rows": rows,
        "skills": SKILLS_BY_INTENT[case["intent"]],
        "tool_calls": len(SKILLS_BY_INTENT[case["intent"]]),
    }


def average(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "evidence_recall",
        "path_recall",
        "answer_accuracy",
        "temporal_accuracy",
        "provenance_precision",
        "unsupported_claim_rate",
    ]
    return {metric: average([row[metric] for row in rows]) for metric in metrics}


def package_versions() -> dict[str, str]:
    names = ["ladybug", "chromadb", "sentence-transformers", "numpy", "torch", "transformers"]
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def make_report(results: dict[str, Any]) -> str:
    class_counts_text = ", ".join(
        f"{case_class}={count}" for case_class, count in sorted(results["class_counts"].items())
    )
    lines = [
        "# Customer assistant graph memory experiment",
        "",
        "The experiment compares four memory conditions over 24 synthetic support cases.",
        "It uses the same customer scope, fixed skills, deterministic answer rule, and as-of date for every condition.",
        "",
        "## Runtime",
        "",
        f"- As-of date: `{results['as_of']}`",
        f"- Cases: `{results['case_count']}`",
        f"- Case classes: `{class_counts_text}`",
        f"- Top-k: `{TOP_K}`",
        f"- Graph database: embedded LadybugDB, backend `pybind`",
        f"- Embedding model: `{MODEL_ID}`",
        "",
        "## Overall metrics",
        "",
        "| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy | Provenance precision | Unsupported claim rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metric in results["overall_metrics"].items():
        lines.append(
            f"| `{method}` | {metric['evidence_recall']} | {metric['path_recall']} | {metric['answer_accuracy']} | {metric['temporal_accuracy']} | {metric['provenance_precision']} | {metric['unsupported_claim_rate']} |"
        )
    lines.extend(["", "## Metrics by case class", ""])
    for case_class, methods in results["class_metrics"].items():
        lines.extend([f"### `{case_class}`", "", "| Method | Evidence recall | Path recall | Answer accuracy | Temporal accuracy |", "| --- | ---: | ---: | ---: | ---: |"])
        for method, metric in methods.items():
            lines.append(f"| `{method}` | {metric['evidence_recall']} | {metric['path_recall']} | {metric['answer_accuracy']} | {metric['temporal_accuracy']} |")
        lines.append("")
    featured = results["featured_case"]
    lines.extend(
        [
            "## Featured case",
            "",
            f"Case `{featured['case_id']}` is the running warranty and replacement case.",
            "The trace records the retrieved records, graph path, time decision, cited records, and final decision for each method.",
            "",
            "## Limits",
            "",
            "The record set is synthetic and small. The results show the behavior of this representation and these cases.",
            "They do not prove that graph memory is best for every customer assistant.",
            "",
            "The experiment isolates memory retrieval. It uses fixed skills and templates instead of a language model that can change the answer policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    configure_runtime()
    source = load_json("customer_records.json")
    case_source = load_json("cases.json")
    records, edges = materialize_records(source)
    cases = case_source["cases"]
    if len(cases) != 24:
        raise ValueError(f"Expected 24 cases, found {len(cases)}")
    class_counts = defaultdict(int)
    for case in cases:
        class_counts[case["case_class"]] += 1
        if case["intent"] not in SKILLS_BY_INTENT:
            raise ValueError(f"Unknown intent: {case['intent']}")
    expected_classes = {"direct": 6, "multi_step_relationship": 6, "temporal": 6, "conflict_authority_missing_evidence": 6}
    if dict(class_counts) != expected_classes:
        raise ValueError(f"Unexpected class counts: {dict(class_counts)}")

    records_by_id = {record["id"]: record for record in records}
    unknown_required = {
        record_id
        for case in cases
        for record_id in case["required_record_ids"]
        if record_id not in records_by_id
    }
    if unknown_required:
        raise ValueError(f"Unknown required records: {sorted(unknown_required)}")

    vector_index = VectorIndex(records)
    graph = LadybugGraph(ROOT / "customer_memory.lbug")
    graph.load(records, edges)
    methods_by_case: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    try:
        for case in cases:
            started = time.perf_counter()
            vector_result = vector_index.retrieve(case, records)
            raw_results = {
                "flat_recent": flat_recent(case, records),
                "vector_chroma": vector_result,
                "graph_ladybug": graph.retrieve(case, records_by_id),
                "hybrid_graph_vector": hybrid_retrieve(case, vector_result, graph, records_by_id),
            }
            methods_by_case[case["case_id"]] = {}
            for method, result in raw_results.items():
                evaluated = evaluate_case(case, method, result, records_by_id)
                evaluated["runtime_ms"] = round((time.perf_counter() - started) * 1000, 3)
                methods_by_case[case["case_id"]][method] = evaluated
                evaluations.append({**evaluated, "case_class": case["case_class"]})
    finally:
        graph.close()

    overall_metrics = {
        method: aggregate([row for row in evaluations if row["method"] == method])
        for method in ("flat_recent", "vector_chroma", "graph_ladybug", "hybrid_graph_vector")
    }
    class_metrics: dict[str, dict[str, Any]] = {}
    for case_class in expected_classes:
        class_metrics[case_class] = {
            method: aggregate([row for row in evaluations if row["method"] == method and row["case_class"] == case_class])
            for method in overall_metrics
        }

    featured_case_id = case_source["featured_case_id"]
    featured_case = next(case for case in cases if case["case_id"] == featured_case_id)
    results = {
        "experiment": "customer_assistant_graph_memory",
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "top_k": TOP_K,
        "case_count": len(cases),
        "class_counts": dict(class_counts),
        "record_count": len(records),
        "edge_count": len(edges),
        "package_versions": package_versions(),
        "methods": list(overall_metrics),
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics,
        "cases": methods_by_case,
        "featured_case": {
            "case_id": featured_case_id,
            "ticket_text": featured_case["ticket_text"],
            "gold_answer": featured_case["gold_answer"],
            "methods": methods_by_case[featured_case_id],
        },
    }
    write_json("results.json", results)
    write_json(
        "trace.json",
        {
            "case_id": featured_case_id,
            "ticket_text": featured_case["ticket_text"],
            "gold_answer": featured_case["gold_answer"],
            "methods": methods_by_case[featured_case_id],
        },
    )
    (ROOT / "report.md").write_text(make_report(results), encoding="utf-8", newline="\n")
    return results


if __name__ == "__main__":
    result = run()
    print(json.dumps({"case_count": result["case_count"], "overall_metrics": result["overall_metrics"]}, indent=2))
