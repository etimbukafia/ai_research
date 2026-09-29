"""Retrieval conditions for the dense enterprise RAG experiment."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
CONDITIONS = ("bm25", "vector", "metadata_filtered_vector", "graph_guided_hybrid")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def document_text(document: dict[str, Any]) -> str:
    return f"{document['title']} {document['body']}"


class BM25Index:
    """A small BM25 index with no service or native extension requirement."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.document_ids = [document["document_id"] for document in documents]
        self.tokens = [tokenize(document_text(document)) for document in documents]
        self.lengths = np.asarray([len(tokens) for tokens in self.tokens], dtype=float)
        self.average_length = float(np.mean(self.lengths)) if documents else 0.0
        self.term_frequency: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in self.tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_frequency.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: np.log(1.0 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores: list[float] = []
        for index, counts in enumerate(self.term_frequency):
            score = 0.0
            denominator_length = self.average_length or 1.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                norm = 1.0 - self.b + self.b * self.lengths[index] / denominator_length
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * norm)
                )
            scores.append(float(score))
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.score(query)
        ranked = sorted(
            zip(self.document_ids, scores), key=lambda item: (-item[1], item[0])
        )
        return ranked[:k]


class VectorIndex:
    """Embedding index with an explicit local deterministic fallback.

    The preferred backend is the exact MiniLM model in the plan.  A clean
    checkout can run without a model cache, so the fallback uses stable hashed
    token vectors and records that fact in the run manifest.
    """

    def __init__(self, documents: list[dict[str, Any]], dimension: int = 256) -> None:
        self.documents = documents
        self.document_ids = [document["document_id"] for document in documents]
        self.dimension = dimension
        self.model_name = EMBEDDING_MODEL
        self.backend = "sentence-transformers"
        self.fallback_reason: str | None = None
        self.model: Any = None
        if os.environ.get("DENSE_RAG_HASH_ONLY") == "1":
            self.backend = "deterministic_hash_fallback"
            self.fallback_reason = "DENSE_RAG_HASH_ONLY=1"
            self.matrix = self._encode([document_text(document) for document in documents])
            return
        try:
            # Keep model loading local.  The experiment must not require a
            # network request to build deterministic artifacts.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device="cpu",
                    model_kwargs={"local_files_only": True},
                    tokenizer_kwargs={"local_files_only": True},
                )
        except Exception as error:  # pragma: no cover - depends on local cache
            self.backend = "deterministic_hash_fallback"
            self.fallback_reason = type(error).__name__
        self.matrix = self._encode([document_text(document) for document in documents])

    def _hashed_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.vstack([self._hashed_vector(text) for text in texts])
        values = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        query_vector = self._encode([query])[0]
        scores = self.matrix @ query_vector
        ranked = sorted(
            zip(self.document_ids, scores.tolist()),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [(document_id, float(score)) for document_id, score in ranked[:k]]


def graph_from_export(export: dict[str, Any]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(name=export.get("graph_name", "gold_graph"))
    for node in export["nodes"]:
        graph.add_node(node["id"], **node["attributes"])
    for edge in export["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            key=edge.get("key"),
            type=edge["type"],
            **edge["attributes"],
        )
    return graph


def edge_matches(
    graph: nx.MultiDiGraph, source: str, target: str, edge_type: str
) -> bool:
    if not graph.has_node(source) or not graph.has_node(target):
        return False
    return any(
        attrs.get("type") == edge_type
        for attrs in graph.get_edge_data(source, target, default={}).values()
    )


def node_for_document(document_id: str) -> str:
    return f"document-{document_id}"


def document_is_effective(document: dict[str, Any], question_time: str) -> bool:
    return document["valid_from"] <= question_time <= document["valid_to"]


def direct_scope_match(document: dict[str, Any], case: dict[str, Any]) -> bool:
    scope = case["required_scope"]
    product_id = scope.get("product_id")
    firmware_version = scope.get("firmware_version")
    region_id = scope.get("region")
    if product_id is None or firmware_version is None or region_id is None:
        return False
    return (
        product_id in document["product_ids"]
        and firmware_version in document["firmware_versions"]
        and region_id in document["regions"]
        and document["status"] == scope.get("status", "Approved")
        and document_is_effective(document, case["question_time"])
    )


def graph_scope_match(
    graph: nx.MultiDiGraph, document: dict[str, Any], case: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Check a candidate through explicit graph relations."""

    scope = case["required_scope"]
    product_id = scope.get("product_id")
    firmware_version = scope.get("firmware_version")
    region_id = scope.get("region")
    if product_id is None or firmware_version is None or region_id is None:
        return False, []
    if document["document_type"] != scope.get("document_type", "specification"):
        return False, []
    if document["status"] != scope.get("status", "Approved"):
        return False, []
    if not document_is_effective(document, case["question_time"]):
        return False, []
    if document["authority"] != scope.get("authority"):
        return False, []
    if not document["specification_authority"]:
        return False, []

    product_node = f"product-{product_id.replace('_', '-')}"
    firmware_node = f"firmware-{firmware_version.replace('.', '-')}"
    region_node = f"region-{region_id.replace('_', '-')}"
    document_node = node_for_document(document["document_id"])
    fact_node = f"fact-{document['document_id']}-pressure-limit"
    authority_node = f"authority-{document['authority'].replace('_', '-')}"
    checks = [
        edge_matches(graph, product_node, document_node, "APPLIES_TO"),
        edge_matches(graph, firmware_node, document_node, "APPLIES_TO"),
        edge_matches(graph, document_node, region_node, "VALID_IN"),
        edge_matches(graph, document_node, f"status-{document['status'].lower()}", "HAS_STATUS"),
        edge_matches(graph, document_node, authority_node, "ISSUED_BY"),
        edge_matches(graph, document_node, fact_node, "DEFINES"),
        edge_matches(graph, product_node, firmware_node, "HAS_FIRMWARE"),
        edge_matches(graph, product_node, region_node, "SOLD_IN"),
    ]
    if not all(checks):
        return False, []

    # A current approved specification can supersede an older source.  An
    # older document is rejected by status and by the relation check below.
    for source, _, attrs in graph.in_edges(document_node, data=True):
        if attrs.get("type") == "SUPERSEDES":
            source_attrs = graph.nodes[source]
            if source_attrs.get("status") == "Approved" and document["status"] != "Approved":
                return False, []

    path = [
        product_node,
        firmware_node,
        region_node,
        document_node,
        fact_node,
        authority_node,
    ]
    return True, path


class Retriever:
    """Run all four conditions over the same documents and cases."""

    def __init__(
        self, documents: list[dict[str, Any]], graph: nx.MultiDiGraph
    ) -> None:
        self.documents = documents
        self.graph = graph
        self.by_id = {document["document_id"]: document for document in documents}
        self.bm25 = BM25Index(documents)
        self.vector = VectorIndex(documents)

    @property
    def vector_backend(self) -> str:
        return self.vector.backend

    @property
    def vector_fallback_reason(self) -> str | None:
        return self.vector.fallback_reason

    def _candidate_records(self, ranked: Iterable[tuple[str, float]]) -> list[dict[str, Any]]:
        return [
            {"rank": rank, "document_id": document_id, "score": round(float(score), 8)}
            for rank, (document_id, score) in enumerate(ranked, 1)
        ]

    def _clarification_status(self, case: dict[str, Any]) -> bool:
        scope = case["required_scope"]
        return (
            (scope.get("product_id") is None and bool(case.get("product_candidates")))
            or (scope.get("region") is None and bool(case.get("region_candidates")))
        )

    def retrieve(self, case: dict[str, Any], condition: str) -> dict[str, Any]:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        start = time.perf_counter()
        if condition == "bm25":
            ranked = self.bm25.top_k(case["question"], 5)
            vector_top20: list[dict[str, Any]] = []
        else:
            vector_ranked = self.vector.top_k(case["question"], 20)
            vector_top20 = self._candidate_records(vector_ranked)
            if condition == "vector":
                ranked = vector_ranked[:5]
            elif condition == "metadata_filtered_vector":
                filtered = [
                    (document_id, score)
                    for document_id, score in vector_ranked
                    if direct_scope_match(self.by_id[document_id], case)
                ]
                ranked = filtered[:5]
            else:
                valid: list[tuple[str, float, list[str]]] = []
                checks: list[dict[str, Any]] = []
                for document_id, score in vector_ranked:
                    document = self.by_id[document_id]
                    is_valid, path = graph_scope_match(self.graph, document, case)
                    checks.append(
                        {
                            "document_id": document_id,
                            "valid": is_valid,
                            "path": path,
                        }
                    )
                    if is_valid:
                        valid.append((document_id, score, path))
                valid.sort(key=lambda item: (-item[1], item[0]))
                ranked = [(document_id, score) for document_id, score, _ in valid[:5]]
        selected_records = self._candidate_records(ranked)
        selected_ids = [record["document_id"] for record in selected_records]
        graph_checks = []
        if condition == "graph_guided_hybrid":
            graph_checks = checks
        selected_document_id = selected_ids[0] if selected_ids else None
        selected_path: list[str] = []
        if selected_document_id is not None:
            selected_document = self.by_id[selected_document_id]
            if condition == "graph_guided_hybrid":
                selected_path = next(
                    (
                        check["path"]
                        for check in graph_checks
                        if check["document_id"] == selected_document_id and check["valid"]
                    ),
                    [],
                )
            else:
                selected_path = [
                    f"document-{selected_document_id}",
                    f"fact-{selected_document_id}-pressure-limit",
                ]
        if not selected_ids:
            answer_status = "clarification" if self._clarification_status(case) else "insufficient_evidence"
        else:
            answer_status = "answer"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "case_id": case["case_id"],
            "condition": condition,
            "candidate_documents": selected_records,
            "selected_document_ids": selected_ids,
            "selected_document_id": selected_document_id,
            "selected_evidence_path": selected_path,
            "vector_top20": vector_top20,
            "graph_checks": graph_checks,
            "answer_status": answer_status,
            "retrieval_latency_ms": round(elapsed_ms, 6),
        }


def load_retriever(directory: Path) -> tuple[Retriever, list[dict[str, Any]], list[dict[str, Any]]]:
    documents = load_json(directory / "documents.json")["documents"]
    cases = load_json(directory / "cases.json")["cases"]
    graph = graph_from_export(load_json(directory / "graph.json"))
    return Retriever(documents, graph), documents, cases
