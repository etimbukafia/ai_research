"""Common BM25 and vector retrieval for the three lab conditions."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol, Sequence

from pydantic import Field

from .models import CatalogEntry, LabModel


class RetrievalDocument(LabModel):
    context_id: str
    condition: str
    kind: str
    text: str
    source_ids: list[str] = Field(default_factory=list)
    catalog_entry: CatalogEntry | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(LabModel):
    context_id: str
    condition: str
    kind: str
    text: str
    score: float
    source_ids: list[str] = Field(default_factory=list)
    catalog_entry: CatalogEntry | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticRetriever(Protocol):
    def search_business_context(
        self,
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = 8,
    ) -> list[RetrievedContext]:
        ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """Small deterministic BM25 implementation with no service dependency."""

    def __init__(self, documents: Sequence[RetrievalDocument]) -> None:
        self.documents = list(documents)
        self.term_counts = [Counter(_tokens(document.text)) for document in self.documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        self.document_frequency = document_frequency
        self.k1 = 1.2
        self.b = 0.75

    def score(self, query: str, document_index: int) -> float:
        query_terms = _tokens(query)
        counts = self.term_counts[document_index]
        length = self.lengths[document_index] or 1
        score = 0.0
        document_count = len(self.documents)
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if not term_frequency:
                continue
            frequency = self.document_frequency.get(term, 0)
            idf = math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * length / max(1.0, self.average_length)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


class HashEmbeddingEncoder:
    """Offline vector encoder used by default in tests and replay."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class SentenceTransformerEncoder:
    """Lazy local MiniLM encoder for runs that opt into the cached model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        encoded = self._load().encode(list(texts), normalize_embeddings=True)
        return encoded.tolist()


class HybridRetriever:
    """BM25 plus vector search with deterministic result ordering."""

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        encoder: HashEmbeddingEncoder | SentenceTransformerEncoder | None = None,
        vector_weight: float = 0.35,
    ) -> None:
        self.documents = list(documents)
        self.bm25 = BM25Index(self.documents)
        self.encoder = encoder or HashEmbeddingEncoder()
        self.vector_weight = vector_weight
        self._vectors = self.encoder.encode([document.text for document in self.documents])

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))

    def search_business_context(
        self,
        query: str,
        account_scope: str | None = None,
        concept_type: str | None = None,
        top_k: int = 8,
    ) -> list[RetrievedContext]:
        if top_k <= 0:
            return []
        candidate_indexes: list[int] = []
        for index, document in enumerate(self.documents):
            if concept_type and document.metadata.get("concept_type") != concept_type:
                continue
            account_ids = document.metadata.get("account_ids")
            if account_scope and account_ids and account_scope not in account_ids:
                continue
            candidate_indexes.append(index)
        if not candidate_indexes:
            return []
        query_vector = self.encoder.encode([query])[0]
        lexical_scores = {
            index: self.bm25.score(query, index) for index in candidate_indexes
        }
        max_lexical = max(lexical_scores.values(), default=0.0) or 1.0
        scored: list[tuple[float, int]] = []
        for index in candidate_indexes:
            lexical = lexical_scores[index] / max_lexical
            vector = max(0.0, self._cosine(query_vector, self._vectors[index]))
            score = (1.0 - self.vector_weight) * lexical + self.vector_weight * vector
            scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], self.documents[pair[1]].context_id))
        return [
            RetrievedContext(
                context_id=self.documents[index].context_id,
                condition=self.documents[index].condition,
                kind=self.documents[index].kind,
                text=self.documents[index].text,
                score=round(score, 8),
                source_ids=self.documents[index].source_ids,
                catalog_entry=self.documents[index].catalog_entry,
                metadata=self.documents[index].metadata,
            )
            for score, index in scored[:top_k]
        ]


def build_retrievers(
    *,
    use_minilm: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, HybridRetriever]:
    """Build the three conditions behind the common retrieval interface."""

    from .catalog import build_documents

    encoder = SentenceTransformerEncoder(model_name) if use_minilm else HashEmbeddingEncoder()
    return {
        condition: HybridRetriever(build_documents(condition), encoder=encoder)
        for condition in ("raw_schema", "prose_rag", "semantic_catalog")
    }

