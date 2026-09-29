"""Run the deterministic dense enterprise RAG experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from answer import compose_deterministic
from build_graph import SEED, build_graph, generate_cases, generate_documents, graph_export, validate_graph
from compression import measure_compression
from evaluators import evaluate_row, paired_changes, summarize_condition
from retrieval import CONDITIONS, Retriever, graph_from_export, load_json


SOURCE_FILES = [
    "README.md",
    "requirements.txt",
    "build_graph.py",
    "extract_relations.py",
    "retrieval.py",
    "compression.py",
    "answer.py",
    "evaluators.py",
    "run_experiment.py",
    "run_live.py",
    "test_dense_enterprise_rag.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_or_generate(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    required = [directory / "documents.json", directory / "cases.json", directory / "graph.json"]
    if not all(path.exists() for path in required):
        documents = generate_documents()
        cases = generate_cases(documents)
        gold_graph = build_graph(documents)
        write_json(directory / "documents.json", {"seed": SEED, "documents": documents})
        write_json(directory / "cases.json", {"seed": SEED, "cases": cases})
        write_json(directory / "graph.json", graph_export(gold_graph))
    documents = load_json(directory / "documents.json")["documents"]
    cases = load_json(directory / "cases.json")["cases"]
    graph = graph_from_export(load_json(directory / "graph.json"))
    validate_graph(graph)
    return documents, cases, graph


def answer_table_line(condition: str, metrics: dict[str, Any]) -> str:
    strict = metrics["strict_applicability_accuracy"]
    answer_status = metrics["answer_status_accuracy"]
    unsupported = metrics["unsupported_claim_rate"]
    compression = metrics["compression"]
    latency = metrics["retrieval_latency_ms"]
    return (
        f"| {condition} | {strict['rate']:.2%} ({strict['correct']}/{strict['total']}) | "
        f"{answer_status['rate']:.2%} ({answer_status['correct']}/{answer_status['total']}) | "
        f"{unsupported['rate']:.2%} ({unsupported['correct']}/{unsupported['total']}) | "
        f"{compression['mean_full_context_tokens']:.1f} / "
        f"{compression['mean_evidence_packet_tokens']:.1f} | "
        f"{latency['p95']:.3f} ms |"
    )


def rate_text(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "n/a"
    return f"{metric['rate']:.2%} ({metric['correct']}/{metric['total']})"


def write_report(
    directory: Path,
    result: dict[str, Any],
    rows_by_condition: dict[str, list[dict[str, Any]]],
) -> None:
    conditions = result["conditions"]
    graph = conditions["graph_guided_hybrid"]["metrics"]
    lines = [
        "# Dense Enterprise RAG Experiment Report",
        "",
        "## Run contract",
        "",
        "This report comes from `run_experiment.py`. The data is synthetic.",
        "The run uses 96 documents and 32 cases in 16 matched pairs.",
        "Eight cases are holdouts. The deterministic answer composer is the",
        "primary answer path.",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Seed: `{result['seed']}`",
        f"- Documents: `{result['document_count']}`",
        f"- Cases: `{result['case_count']}`",
        f"- Holdouts: `{result['holdout_count']}`",
        f"- Question date: `{result['question_date']}`",
        f"- Embedding model: `{result['embedding_model']}`",
        f"- Embedding backend: `{result['embedding_backend']}`",
        f"- Embedding fallback reason: `{result['embedding_fallback_reason'] or 'none'}`",
        f"- Python: `{result['python_version']}`",
        "",
        "## Overall retrieval result",
        "",
        "Strict applicability uses the exact applicable document for the 24",
        "answer cases. Answer status uses all 32 cases.",
        "",
        "| Condition | Strict applicability | Answer status | Unsupported claim | Full / packet tokens | Retrieval P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        lines.append(answer_table_line(condition, conditions[condition]["metrics"]["overall"]))
    lines.extend(
        [
            "",
            "## Clean, dense, and holdout cases",
            "",
            "The clean and dense rows are the two members of each matched pair.",
            "Holdout rows remain separate from threshold selection. This run uses",
            "fixed rules and does not tune on the holdouts.",
            "",
            "| Condition | Slice | Strict applicability | Exact fact | Correct abstention |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for condition in CONDITIONS:
        for slice_name in ("clean", "dense_overlap", "holdout"):
            metrics = conditions[condition]["metrics"][slice_name]
            lines.append(
                f"| {condition} | {slice_name} | "
                f"{rate_text(metrics['strict_applicability_accuracy'])} | "
                f"{rate_text(metrics['exact_fact_accuracy'])} | "
                f"{rate_text(metrics['correct_abstention_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Results by case class",
            "",
            "| Condition | Case class | Strict applicability | Answer status |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for condition in CONDITIONS:
        for case_class, metrics in conditions[condition]["metrics"]["by_case_class"].items():
            lines.append(
                f"| {condition} | {case_class} | "
                f"{rate_text(metrics['strict_applicability_accuracy'])} | "
                f"{rate_text(metrics['answer_status_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "## Compression",
            "",
            "The packet keeps the fact, unit, scope, validity, authority, source,",
            "and evidence path. Token counts use the stable word-token proxy in",
            "`compression.py`.",
            "",
            "| Condition | Mean full tokens | Mean packet tokens | Mean reduction | Required evidence retained |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in CONDITIONS:
        compression = conditions[condition]["metrics"]["overall"]["compression"]
        lines.append(
            f"| {condition} | {compression['mean_full_context_tokens']:.2f} | "
            f"{compression['mean_evidence_packet_tokens']:.2f} | "
            f"{compression['mean_compression_ratio']:.2%} | "
            f"{rate_text(compression['required_evidence_retention'])} |"
        )
    lines.extend(
        [
            "",
            "## Matched pairs",
            "",
            "The full pair record is in `results.json` under",
            "`matched_pairs.<condition>`.",
            "",
        ]
    )
    for condition in CONDITIONS:
        pair_values = conditions[condition]["matched_pairs"]
        changed = sum(
            value.get("strict_applicability_change") == -1
            for value in pair_values.values()
        )
        lines.append(
            f"- `{condition}`: `{changed}` dense cases lost strict applicability "
            "relative to their clean pair."
        )
    lines.extend(
        [
            "",
            "## Failure traces",
            "",
            "Each class has one dense trace below. The complete trace set is in",
            "`traces.jsonl`.",
            "",
        ]
    )
    for case_class in sorted({row["case_class"] for row in rows_by_condition["graph_guided_hybrid"]}):
        candidate = next(
            row
            for row in rows_by_condition["vector"]
            if row["case_class"] == case_class and row["is_dense_overlap"]
        )
        graph_row = next(
            row
            for row in rows_by_condition["graph_guided_hybrid"]
            if row["case_id"] == candidate["case_id"]
        )
        lines.extend(
            [
                f"### `{case_class}`",
                "",
                f"- Case: `{candidate['case_id']}`",
                f"- Vector selected: `{candidate['selected_document_id']}`",
                f"- Graph selected: `{graph_row['selected_document_id']}`",
                f"- Vector candidates: `{', '.join(candidate['candidate_document_ids'])}`",
                f"- Graph path: `{ ' -> '.join(graph_row['selected_evidence_path']) or 'none' }`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The graph condition is useful only when its source edges are correct",
            "and the question contains enough scope to form a valid path.",
            "Metadata filters can solve direct field checks. The graph condition",
            "adds relation checks for authority and supersession.",
            "",
            "This synthetic set does not estimate enterprise failure rates. Local",
            "latency is hardware-dependent. It is not comparable to a hosted",
            "vendor latency claim.",
            "",
            "## Result field paths for the article",
            "",
            "- Overall strict applicability: `results.json` -> `conditions.<condition>.metrics.overall.strict_applicability_accuracy`",
            "- Dense strict applicability: `results.json` -> `conditions.<condition>.metrics.dense_overlap.strict_applicability_accuracy`",
            "- Case-class result: `results.json` -> `conditions.<condition>.metrics.by_case_class.<case_class>.strict_applicability_accuracy`",
            "- Compression: `results.json` -> `conditions.<condition>.metrics.overall.compression`",
            "- Retrieval latency: `results.json` -> `conditions.<condition>.metrics.overall.retrieval_latency_ms`",
            "- Extraction audit: `extraction_results.json`",
            "- Live generation: `live_results.json`",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    documents, cases, graph = load_or_generate(directory)
    if len(documents) != 96 or len(cases) != 32:
        raise AssertionError("The experiment must run on 96 documents and 32 cases")
    holdout_count = sum(case["is_holdout"] for case in cases)

    graph_start = time.perf_counter()
    rebuilt_graph = build_graph(documents)
    validate_graph(rebuilt_graph)
    graph_build_ms = (time.perf_counter() - graph_start) * 1000.0

    index_start = time.perf_counter()
    retriever = Retriever(documents, graph)
    index_build_ms = (time.perf_counter() - index_start) * 1000.0
    documents_by_id = {document["document_id"]: document for document in documents}

    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    trace_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows: list[dict[str, Any]] = []
        for case in cases:
            retrieval = retriever.retrieve(case, condition)
            answer = compose_deterministic(case, retrieval, documents_by_id)
            compression = measure_compression(case, retrieval, documents_by_id)
            row = evaluate_row(case, retrieval, answer, compression, documents_by_id)
            rows.append(row)
            trace_records.append(
                {
                    "case_id": case["case_id"],
                    "pair_id": case["pair_id"],
                    "case_class": case["case_class"],
                    "condition": condition,
                    "question": case["question"],
                    "question_time": case["question_time"],
                    "required_scope": case["required_scope"],
                    "expected_document_id": case["applicable_document_id"],
                    "candidate_documents": retrieval["candidate_documents"],
                    "vector_top20": retrieval["vector_top20"],
                    "selected_document_id": retrieval["selected_document_id"],
                    "selected_evidence_path": retrieval["selected_evidence_path"],
                    "graph_checks": retrieval["graph_checks"],
                    "answer": answer.model_dump(mode="json"),
                    "metrics": row,
                    "compression": compression,
                }
            )
        rows_by_condition[condition] = rows

    config_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "seed": SEED,
                "embedding_model": retriever.vector.model_name,
                "conditions": CONDITIONS,
                "document_count": len(documents),
                "case_count": len(cases),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    conditions_result = {}
    paired_result = {}
    for condition in CONDITIONS:
        conditions_result[condition] = {
            "metrics": summarize_condition(rows_by_condition[condition]),
            "row_count": len(rows_by_condition[condition]),
        }
        paired_result[condition] = paired_changes(rows_by_condition[condition])
        conditions_result[condition]["matched_pairs"] = paired_result[condition]

    run_id = f"dense-enterprise-rag-{SEED}-{config_fingerprint}"
    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "question_date": "2026-08-28",
        "document_count": len(documents),
        "case_count": len(cases),
        "pair_count": len({case["pair_id"] for case in cases}),
        "holdout_count": holdout_count,
        "embedding_model": retriever.vector.model_name,
        "embedding_backend": retriever.vector.backend,
        "embedding_fallback_reason": retriever.vector.fallback_reason,
        "python_version": sys.version,
        "conditions": conditions_result,
        "matched_pairs": paired_result,
        "operational": {
            "graph_build_ms": round(graph_build_ms, 6),
            "index_build_ms": round(index_build_ms, 6),
            "deterministic_answer_requests": 0,
            "deterministic_answer_model": None,
            "repeat_count": 1,
            "warm_up_rule": "No query warm-up. Index build is outside retrieval latency.",
            "hardware": platform.platform(),
            "python": sys.version,
        },
        "source_files": SOURCE_FILES,
        "result_file_contract": {
            "condition_metrics": "conditions.<condition>.metrics",
            "overall_metrics": "conditions.<condition>.metrics.overall",
            "dense_metrics": "conditions.<condition>.metrics.dense_overlap",
            "class_metrics": "conditions.<condition>.metrics.by_case_class.<case_class>",
            "matched_pairs": "matched_pairs.<condition>",
        },
    }

    write_json(directory / "results.json", result)
    with (directory / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for trace in trace_records:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    write_report(directory, result, rows_by_condition)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "command": "python -B run_experiment.py",
        "seed": SEED,
        "document_count": len(documents),
        "case_count": len(cases),
        "pair_count": len({case["pair_id"] for case in cases}),
        "holdout_count": holdout_count,
        "model": retriever.vector.model_name,
        "embedding_backend": retriever.vector.backend,
        "embedding_fallback_reason": retriever.vector.fallback_reason,
        "conditions": list(CONDITIONS),
        "answer_composer": "answer.py:compose_deterministic",
        "graph_builder": "build_graph.py:build_graph",
        "graph": "NetworkX MultiDiGraph; local JSON export",
        "live_key_policy": "The live runner reads GOOGLE_API_KEY from the environment only. The deterministic runner does not read it.",
        "docker_required": False,
        "external_graph_database_required": False,
        "package_versions": {
            "numpy": package_version("numpy"),
            "networkx": package_version("networkx"),
            "pydantic": package_version("pydantic"),
            "pydantic-ai-slim": package_version("pydantic-ai-slim"),
            "sentence-transformers": package_version("sentence-transformers"),
            "google-genai": package_version("google-genai"),
        },
        "source_sha256": {
            filename: sha256_file(directory / filename)
            for filename in SOURCE_FILES
            if (directory / filename).exists()
        },
        "outputs": [
            "documents.json",
            "cases.json",
            "graph.json",
            "graph_schema.json",
            "results.json",
            "traces.jsonl",
            "report.md",
            "extraction_results.json",
            "live_results.json",
        ],
    }
    write_json(directory / "run_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    result = run(args.directory)
    graph_metrics = result["conditions"]["graph_guided_hybrid"]["metrics"]["overall"]
    print(
        "Deterministic run complete: "
        f"{result['document_count']} documents, {result['case_count']} cases, "
        f"graph strict applicability {graph_metrics['strict_applicability_accuracy']['rate']:.2%}."
    )


if __name__ == "__main__":
    main()
