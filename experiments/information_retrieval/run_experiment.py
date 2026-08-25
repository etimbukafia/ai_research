"""Run the local information retrieval comparison.

The script keeps the lexical scoring rules visible. It uses a new in-memory
Chroma collection for each run and writes results for the article writer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus.json"
RESULTS_PATH = BASE_DIR / "results.json"
REPORT_PATH = BASE_DIR / "report.md"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SEED = 42
K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "does",
        "from",
        "how",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "when",
        "which",
        "with",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase text, keep ASCII word tokens, and remove stopwords."""

    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    ]


def load_corpus() -> dict[str, Any]:
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    documents = data.get("documents", [])
    queries = data.get("queries", {})
    document_ids = [document["id"] for document in documents]
    if document_ids != sorted(document_ids):
        raise ValueError("Documents must be stored in ascending ID order.")
    if len(document_ids) != 12 or len(set(document_ids)) != 12:
        raise ValueError("The corpus must contain 12 unique documents.")
    if set(queries) != {"main", "validation_direct", "validation_paraphrase"}:
        raise ValueError("The corpus must contain the three planned queries.")
    for query_name, query in queries.items():
        if not query.get("text") or not query.get("relevant_ids"):
            raise ValueError(f"Query {query_name} needs text and relevant_ids.")
        unknown_ids = set(query["relevant_ids"]) - set(document_ids)
        if unknown_ids:
            raise ValueError(f"Query {query_name} has unknown labels: {unknown_ids}")
    return data


def document_text(document: dict[str, Any]) -> str:
    """Use title and body as the indexed text. Tags stay metadata only."""

    return f"{document['title']} {document['text']}"


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row["score"]), row["id"]))


def exact_keyword_rank(
    query: str, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    query_terms = set(tokenize(query))
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_terms = set(tokenize(document_text(document)))
        matched_terms = sorted(query_terms & document_terms)
        if matched_terms:
            rows.append(
                {
                    "id": document["id"],
                    "score": len(matched_terms),
                    "matched_terms": matched_terms,
                }
            )
    return sort_rows(rows)


def bm25_rank(query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenized_documents = [tokenize(document_text(document)) for document in documents]
    query_terms = sorted(set(tokenize(query)))
    document_count = len(tokenized_documents)
    average_length = sum(map(len, tokenized_documents)) / document_count
    document_frequency: Counter[str] = Counter()
    for terms in tokenized_documents:
        document_frequency.update(set(terms))

    rows: list[dict[str, Any]] = []
    for document, terms in zip(documents, tokenized_documents):
        term_frequency = Counter(terms)
        score = 0.0
        matched_terms: list[str] = []
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            matched_terms.append(term)
            df = document_frequency[term]
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            length_factor = 1.0 - B + B * len(terms) / average_length
            score += idf * (frequency * (K1 + 1.0)) / (
                frequency + K1 * length_factor
            )
        rows.append(
            {
                "id": document["id"],
                "score": score,
                "matched_terms": matched_terms,
            }
        )
    return sort_rows(rows)


def set_deterministic_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.set_num_threads(1)
    except ImportError:
        pass


def vector_chroma_rank(
    query: str, documents: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed and retrieve all documents through an in-memory Chroma client."""

    import chromadb
    from sentence_transformers import SentenceTransformer

    set_deterministic_runtime()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    texts = [document_text(document) for document in documents]
    document_ids = [document["id"] for document in documents]
    document_embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    client = chromadb.Client()
    collection_name = (
        "information_retrieval_"
        + hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    )
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=document_ids,
        documents=texts,
        embeddings=document_embeddings.tolist(),
    )
    response = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=len(document_ids),
        include=["distances"],
    )

    rows: list[dict[str, Any]] = []
    returned_ids = response["ids"][0]
    distances = response["distances"][0]
    for document_id, distance in zip(returned_ids, distances):
        rows.append(
            {
                "id": document_id,
                "score": 1.0 - float(distance),
                "cosine_distance": float(distance),
            }
        )
    rows = sort_rows(rows)
    return rows, {
        "client": "chromadb.Client()",
        "collection": collection_name,
        "space": "cosine",
        "embedding_dimensions": int(document_embeddings.shape[1]),
    }


def recall_at_5(ranking: list[dict[str, Any]], relevant_ids: list[str]) -> float:
    result_ids = {row["id"] for row in ranking[:TOP_K]}
    return len(result_ids & set(relevant_ids)) / len(relevant_ids)


def mrr_at_5(ranking: list[dict[str, Any]], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, row in enumerate(ranking[:TOP_K], start=1):
        if row["id"] in relevant:
            return 1.0 / rank
    return 0.0


def package_versions() -> dict[str, str]:
    package_names = [
        "chromadb",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    ]
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def run_experiment(data: dict[str, Any]) -> dict[str, Any]:
    documents = data["documents"]
    queries = data["queries"]
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    vector_details: dict[str, Any] | None = None
    vector_error: str | None = None

    for query_name, query in queries.items():
        query_text = query["text"]
        rankings[query_name] = {
            "exact_keyword": exact_keyword_rank(query_text, documents),
            "bm25": bm25_rank(query_text, documents),
        }
        if vector_error is None:
            try:
                vector_rows, vector_details = vector_chroma_rank(query_text, documents)
                rankings[query_name]["vector_chroma"] = vector_rows
            except Exception as exc:  # record the failure in the output
                vector_error = f"{type(exc).__name__}: {exc}"

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for query_name, query in queries.items():
        metrics[query_name] = {}
        for method, ranking in rankings[query_name].items():
            metrics[query_name][method] = {
                "recall_at_5": recall_at_5(ranking, query["relevant_ids"]),
                "mrr": mrr_at_5(ranking, query["relevant_ids"]),
            }

    status = "complete" if vector_error is None else "failed"
    corpus_bytes = CORPUS_PATH.read_bytes()
    return {
        "status": status,
        "model_id": MODEL_ID,
        "top_k": TOP_K,
        "metric_definition": {
            "recall_at_5": "Relevant document share in the first five results.",
            "mrr": "Reciprocal rank of the first relevant result in the first five results; zero when none is relevant.",
        },
        "tokenizer": {
            "pattern": "[a-z0-9]+",
            "lowercase": True,
            "stopword_removal": True,
            "stemming": False,
            "stopwords": sorted(STOPWORDS),
        },
        "bm25_parameters": {"k1": K1, "b": B},
        "vector_store": vector_details
        or {
            "client": "chromadb.Client()",
            "collection": "information_retrieval",
            "space": "cosine",
        },
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(BASE_DIR.parent.parent)),
            "sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "document_ids": [document["id"] for document in documents],
        },
        "queries": queries,
        "rankings": {
            query_name: {
                method: rows[:TOP_K]
                for method, rows in query_rankings.items()
            }
            for query_name, query_rankings in rankings.items()
        },
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": package_versions(),
        },
        "vector_error": vector_error,
    }


def score_text(row: dict[str, Any]) -> str:
    if "cosine_distance" in row:
        return f"{float(row['score']):.6f} (distance {float(row['cosine_distance']):.6f})"
    if isinstance(row["score"], int):
        return str(row["score"])
    return f"{float(row['score']):.6f}"


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Experiment report",
        "",
        f"Status: **{results['status']}**",
        "",
        "This report records the local comparison for the article. The labels are manual labels for the 12-document corpus.",
        "",
        f"- Model: `{results['model_id']}`",
        f"- Top-k: `{results['top_k']}`",
        f"- Corpus SHA-256: `{results['corpus']['sha256']}`",
        f"- Python: `{results['runtime']['python']}`",
        f"- Platform: `{results['runtime']['platform']}`",
        "",
        "## Method rules",
        "",
        "- `exact_keyword` counts unique query terms that occur in the title or body.",
        "- `bm25` uses the visible tokenizer and parameters in `run_experiment.py`.",
        "- `vector_chroma` uses normalized MiniLM vectors in an in-memory Chroma collection with cosine distance. The reported score is `1 - cosine distance`.",
        "- Ties use ascending document ID after score order.",
        "",
        "## Results",
        "",
    ]
    for query_name, query in results["queries"].items():
        lines.extend(
            [
                f"### `{query_name}`",
                "",
                f"> {query['text']}",
                "",
                f"Manual relevant IDs: `{', '.join(query['relevant_ids'])}`",
                "",
                "| Method | Top five (score) | Recall@5 | MRR |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for method in ("exact_keyword", "bm25", "vector_chroma"):
            if method not in results["rankings"].get(query_name, {}):
                continue
            rows = results["rankings"][query_name][method]
            top_five = "; ".join(
                f"`{row['id']}` {score_text(row)}" for row in rows
            )
            metric = results["metrics"][query_name][method]
            lines.append(
                f"| `{method}` | {top_five} | {metric['recall_at_5']:.6f} | {metric['mrr']:.6f} |"
            )
        lines.extend(["", "### Main-query notes", ""] if query_name == "main" else [""])
        if query_name == "main":
            main_rows = results["rankings"][query_name]
            exact_ids = [row["id"] for row in main_rows.get("exact_keyword", [])]
            bm25_ids = [row["id"] for row in main_rows.get("bm25", [])]
            vector_ids = [row["id"] for row in main_rows.get("vector_chroma", [])]
            lines.extend(
                [
                    f"- Exact keyword top-five IDs: `{', '.join(exact_ids)}`.",
                    f"- BM25 top-five IDs: `{', '.join(bm25_ids)}`.",
                    f"- Vector top-five IDs: `{', '.join(vector_ids)}`.",
                    "- Use the scores as observations for this corpus. They do not prove answer quality.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Package versions",
            "",
            "```text",
            *[
                f"{package}: {version}"
                for package, version in results["runtime"]["package_versions"].items()
            ],
            "```",
            "",
            "## Limitations",
            "",
            "- The corpus has 12 short documents and three queries.",
            "- The relevance labels are manual and binary. They do not capture partial usefulness.",
            "- Exact keyword search and BM25 use no stemming or synonym expansion.",
            "- Vector similarity measures representation closeness. It does not verify facts or evidence.",
            "- The embedding model is small and general. Results can change with another model.",
            "- Chroma runs in memory. The experiment does not test persistence, updates, latency, or scale.",
            "- Runtime depends on the local CPU, package versions, model cache, and model download state.",
            "",
        ]
    )
    if results.get("vector_error"):
        lines.extend(
            [
                "## Failed vector stage",
                "",
                f"`{results['vector_error']}`",
                "",
                "The lexical results are recorded, but this run is not a complete comparison.",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(results: dict[str, Any]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(render_report(results), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="Run only lexical methods. The output is marked incomplete.",
    )
    args = parser.parse_args()
    data = load_corpus()
    started = time.perf_counter()
    if args.skip_vector:
        # Keep this flag useful for local tokenizer and BM25 checks. It does not
        # claim to satisfy the complete experiment.
        original_vector_rank = vector_chroma_rank

        def unavailable_vector_rank(
            query: str, documents: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            del query, documents
            raise RuntimeError("vector stage skipped by --skip-vector")

        globals()["vector_chroma_rank"] = unavailable_vector_rank
        try:
            results = run_experiment(data)
        finally:
            globals()["vector_chroma_rank"] = original_vector_rank
    else:
        results = run_experiment(data)
    results["runtime"]["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    write_outputs(results)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if results["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
