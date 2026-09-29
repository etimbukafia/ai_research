"""Run the controlled Gemini answer slice with checkpoint and rate control."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from answer import Answer, live_contexts, live_system_prompt, live_user_prompt
from retrieval import CONDITIONS, load_retriever


MODEL_NAME = "google:gemini-3.5-flash-lite"
INTERVAL_SECONDS = 5.0
RPM_LIMIT = 15
TRIALS = 2
LIVE_CASE_IDS = [
    f"case-{number:02d}-{kind}"
    for number in range(1, 5)
    for kind in ("clean", "dense")
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def planned_keys() -> list[str]:
    return [
        f"{case_id}|{condition}|trial-{trial}"
        for case_id in LIVE_CASE_IDS
        for condition in CONDITIONS
        for trial in range(1, TRIALS + 1)
    ]


def empty_live_result(status: str, reason: str, retriever: Any | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": status,
        "reason": reason,
        "model": MODEL_NAME,
        "rpm_limit": RPM_LIMIT,
        "interval_seconds": INTERVAL_SECONDS,
        "case_ids": LIVE_CASE_IDS,
        "condition_names": list(CONDITIONS),
        "trial_count": TRIALS,
        "planned_requests": len(planned_keys()),
        "request_count": 0,
        "provider_request_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "stopped_for_rate_limit": False,
        "context_form_by_trial": {"1": "full_documents", "2": "evidence_packet"},
        "embedding_backend": getattr(retriever, "vector_backend", None),
        "metrics": {},
        "records": [],
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {"records": []}


def save_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "schema_version": "1.0",
            "model": MODEL_NAME,
            "interval_seconds": INTERVAL_SECONDS,
            "planned_requests": len(planned_keys()),
            "records": records,
        },
    )


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n{user}".encode("utf-8")).hexdigest()


def safe_usage(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    try:
        if callable(usage):
            usage = usage()
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        fields = {}
        for source, target in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source) if isinstance(usage, dict) else getattr(usage, source, None)
            if isinstance(value, int):
                fields[target] = value
        return fields or None
    except Exception:
        return None
    return None


async def ask_agent(agent: Any, system: str, user: str) -> tuple[Answer, dict[str, int] | None]:
    result = await agent.run(user)
    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "data", None)
    answer = output if isinstance(output, Answer) else Answer.model_validate(output)
    return answer, safe_usage(result)


def live_metric(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [record for record in records if record.get("status") == "completed"]
    if not usable:
        return {
            "request_count": 0,
            "exact_fact_accuracy": None,
            "answer_status_accuracy": None,
            "source_recall": None,
            "latency_ms": {"p50": None, "p95": None},
            "input_tokens": None,
            "output_tokens": None,
        }
    exact = [record["exact_fact"] for record in usable]
    status = [record["answer_status"] for record in usable]
    source = [record["source_recall"] for record in usable]
    latency = sorted(float(record["latency_ms"]) for record in usable)

    def percentile(percent: float) -> float:
        if len(latency) == 1:
            return round(latency[0], 6)
        position = (len(latency) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(latency) - 1)
        return round(latency[lower] + (latency[upper] - latency[lower]) * (position - lower), 6)

    input_tokens = [record["usage"]["input_tokens"] for record in usable if record.get("usage") and "input_tokens" in record["usage"]]
    output_tokens = [record["usage"]["output_tokens"] for record in usable if record.get("usage") and "output_tokens" in record["usage"]]
    return {
        "request_count": len(usable),
        "exact_fact_accuracy": {
            "correct": sum(exact),
            "total": len(exact),
            "rate": round(sum(exact) / len(exact), 8),
        },
        "answer_status_accuracy": {
            "correct": sum(status),
            "total": len(status),
            "rate": round(sum(status) / len(status), 8),
        },
        "source_recall": {
            "correct": sum(source),
            "total": len(source),
            "rate": round(sum(source) / len(source), 8),
        },
        "latency_ms": {
            "p50": percentile(50),
            "p95": percentile(95),
        },
        "input_tokens": sum(input_tokens) if input_tokens else None,
        "output_tokens": sum(output_tokens) if output_tokens else None,
    }


def summarize_live(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for context_form in ("full_documents", "evidence_packet"):
        result[context_form] = live_metric(
            [record for record in records if record.get("context_form") == context_form]
        )
    for condition in CONDITIONS:
        result[condition] = live_metric(
            [record for record in records if record.get("condition") == condition]
        )
    return result


def append_report_status(directory: Path, result: dict[str, Any]) -> None:
    report_path = directory / "report.md"
    if not report_path.exists():
        return
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Model check status\n"
    if marker in report:
        report = report.split(marker, 1)[0]
    report += marker + "\n"
    extraction_path = directory / "extraction_results.json"
    extraction = (
        json.loads(extraction_path.read_text(encoding="utf-8"))
        if extraction_path.exists()
        else {"status": "missing", "metrics": {}}
    )
    report += f"- Relation extraction: `{extraction['status']}`; see `extraction_results.json`.\n"
    report += f"- Live Gemini slice: `{result['status']}`; see `live_results.json`.\n"
    if result.get("reason"):
        report += f"- Live run reason: {result['reason']}\n"
    extraction_metrics = extraction.get("metrics", {})
    if extraction.get("status") == "completed":
        report += "\n### Relation-extraction audit\n\n"
        report += "| Measure | Value |\n| --- | ---: |\n"
        for label, field in (
            ("Node precision", "node_precision"),
            ("Node recall", "node_recall"),
            ("Edge precision", "edge_precision"),
            ("Edge recall", "edge_recall"),
            ("Scope-field accuracy", "scope_field_accuracy"),
            ("Validity-field accuracy", "validity_field_accuracy"),
            ("Authority accuracy", "authority_accuracy"),
            ("Invalid-edge rejection", "invalid_edge_rejection_rate"),
            ("Review rate", "review_rate"),
        ):
            value = extraction_metrics.get(field)
            report += f"| {label} | {value:.2%} |\n" if isinstance(value, float) else f"| {label} | n/a |\n"
    if result.get("status") == "completed":
        report += "\n### Live answer check\n\n"
        report += "| Slice | Exact fact | Answer status | Source recall | Requests |\n"
        report += "| --- | ---: | ---: | ---: | ---: |\n"
        for name in ("bm25", "vector", "metadata_filtered_vector", "graph_guided_hybrid", "full_documents", "evidence_packet"):
            metric = result["metrics"][name]
            exact = metric["exact_fact_accuracy"]
            status = metric["answer_status_accuracy"]
            source = metric["source_recall"]
            report += (
                f"| {name} | {exact['correct']}/{exact['total']} ({exact['rate']:.2%}) "
                f"| {status['correct']}/{status['total']} ({status['rate']:.2%}) "
                f"| {source['correct']}/{source['total']} ({source['rate']:.2%}) "
                f"| {metric['request_count']} |\n"
            )
    report_path.write_text(report, encoding="utf-8")


def run(directory: Path, resume: bool = True) -> dict[str, Any]:
    output_path = directory / "live_results.json"
    checkpoint_path = directory / ".live_checkpoint.json"

    if not os.environ.get("GOOGLE_API_KEY"):
        embedding_backend = None
        results_path = directory / "results.json"
        if results_path.exists():
            try:
                embedding_backend = json.loads(results_path.read_text(encoding="utf-8")).get(
                    "embedding_backend"
                )
            except (OSError, json.JSONDecodeError):
                embedding_backend = None
        result = empty_live_result(
            "not_run",
            "GOOGLE_API_KEY is not set; the live Gemini slice was not run.",
            None,
        )
        result["embedding_backend"] = embedding_backend
        write_json(output_path, result)
        append_report_status(directory, result)
        return result

    retriever, documents, cases = load_retriever(directory)
    cases_by_id = {case["case_id"]: case for case in cases}
    documents_by_id = {document["document_id"]: document for document in documents}

    records = load_checkpoint(checkpoint_path).get("records", []) if resume else []
    completed_keys = {record.get("request_key") for record in records}
    remaining = [key for key in planned_keys() if key not in completed_keys]
    try:
        from pydantic_ai import Agent
    except Exception:
        result = empty_live_result(
            "partial" if records else "not_run",
            "PydanticAI or the Google provider is unavailable; the live slice did not complete.",
            retriever,
        )
        result["records"] = records
        result["request_count"] = len(records)
        result["provider_request_count"] = len(records)
        result["metrics"] = summarize_live(records)
        write_json(output_path, result)
        append_report_status(directory, result)
        return result

    last_request = 0.0
    stopped_for_rate_limit = False
    error_count = 0
    last_error_type: str | None = None
    last_error_status: int | None = None
    for request_key in remaining:
        case_id, condition, trial_name = request_key.split("|")
        trial = int(trial_name.replace("trial-", ""))
        context_form = "full_documents" if trial == 1 else "evidence_packet"
        case = cases_by_id[case_id]
        retrieval = retriever.retrieve(case, condition)
        contexts = live_contexts(case, retrieval, documents_by_id)
        system = live_system_prompt(context_form)
        user = live_user_prompt(case, contexts[context_form])
        wait = INTERVAL_SECONDS - (time.monotonic() - last_request)
        if last_request and wait > 0:
            time.sleep(wait)
        started = time.perf_counter()
        last_request = time.monotonic()
        try:
            agent = Agent(
                MODEL_NAME,
                output_type=Answer,
                system_prompt=system,
                retries=2,
            )
            answer, usage = asyncio.run(ask_agent(agent, system, user))
            expected = case.get("expected_answer")
            exact_fact = bool(
                expected
                and answer.status == "answer"
                and answer.value == expected["value"]
                and answer.unit == expected["unit"]
            )
            expected_status = case["expected_answer_status"]
            status_correct = answer.status == ("answer" if expected_status == "answer" else expected_status)
            expected_source = case.get("applicable_document_id")
            source_recall = bool(expected_source and expected_source in answer.source_ids)
            records.append(
                {
                    "request_key": request_key,
                    "case_id": case_id,
                    "pair_id": case["pair_id"],
                    "condition": condition,
                    "trial": trial,
                    "context_form": context_form,
                    "status": "completed",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
                    "prompt_hash": prompt_hash(system, user),
                    "answer": answer.model_dump(mode="json"),
                    "exact_fact": exact_fact,
                    "answer_status": status_correct,
                    "source_recall": source_recall,
                    "usage": usage,
                }
            )
            save_checkpoint(checkpoint_path, records)
        except Exception as error:
            error_count += 1
            last_error_type = type(error).__name__
            status_code = getattr(error, "status_code", None)
            last_error_status = status_code if isinstance(status_code, int) else None
            error_name = type(error).__name__.lower()
            error_text = str(error).lower()
            stopped_for_rate_limit = (
                "rate" in error_name
                or "429" in error_text
                or last_error_status == 429
            )
            save_checkpoint(checkpoint_path, records)
            break

    complete = len({record.get("request_key") for record in records}) == len(planned_keys())
    result = {
        "schema_version": "1.0",
        "status": "completed" if complete else "partial",
        "reason": None if complete else (
            "The provider returned a rate-limit response; resume from .live_checkpoint.json."
            if stopped_for_rate_limit
            else "The provider request loop stopped before all planned requests completed."
        ),
        "model": MODEL_NAME,
        "rpm_limit": RPM_LIMIT,
        "interval_seconds": INTERVAL_SECONDS,
        "case_ids": LIVE_CASE_IDS,
        "condition_names": list(CONDITIONS),
        "trial_count": TRIALS,
        "planned_requests": len(planned_keys()),
        "request_count": len(records),
        "provider_request_count": len(records) + error_count,
        "retry_count": 0,
        "error_count": error_count,
        "last_error_type": last_error_type,
        "last_error_status": last_error_status,
        "stopped_for_rate_limit": stopped_for_rate_limit,
        "context_form_by_trial": {"1": "full_documents", "2": "evidence_packet"},
        "embedding_backend": retriever.vector_backend,
        "metrics": summarize_live(records),
        "records": records,
    }
    write_json(output_path, result)
    append_report_status(directory, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint.")
    args = parser.parse_args()
    result = run(args.directory, resume=not args.no_resume)
    print(f"Live Gemini slice: {result['status']} ({result['request_count']} requests).")


if __name__ == "__main__":
    main()
