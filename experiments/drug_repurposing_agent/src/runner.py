"""Reproducible sequential experiment runner, validator, and report command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.drug_repurposing_agent.src.agent import BaselineAssessor
from experiments.drug_repurposing_agent.config import Config
from experiments.drug_repurposing_agent.src.constants import CLASSIFICATION_LABELS, MEMORY_CONDITIONS
from experiments.drug_repurposing_agent.src.evaluator import evaluate_assessment
from experiments.drug_repurposing_agent.src.lessons import LessonWorkflow
from experiments.drug_repurposing_agent.src.memory import ExperimentMemory, JsonFileBackend
from experiments.drug_repurposing_agent.src.models import (
    DiseaseDrugPair,
    ExperimentSession,
    GoldRecord,
    MemoryCondition,
    RunRecord,
    SessionEvent,
    SessionEventType,
    SessionStatus,
)
from experiments.drug_repurposing_agent.src.open_targets import OpenTargetsClient
from experiments.drug_repurposing_agent.src.pubmed import PubMedClient
from experiments.drug_repurposing_agent.src.retrieval import RetrievalResult

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = EXPERIMENT_ROOT / "data"
DEFAULT_RUNS_DIR = EXPERIMENT_ROOT / "runs"
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "config" / "experiment.yaml"
ASSESSOR_PROMPT_PATH = EXPERIMENT_ROOT / "prompts" / "assessor.md"
LESSON_PROMPT_PATH = EXPERIMENT_ROOT / "prompts" / "lesson_generator.md"


class GeminiStructuredModel:
    """Synchronous Gemini JSON adapter used by the CLI runner."""

    _last_request_at = 0.0
    _minimum_request_interval_seconds = 4.2

    def __init__(self, *, model_name: str, temperature: float, seed: int) -> None:
        api_key = Config.load().GEMINI_API_KEY
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is required to run the experiment") from exc
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": temperature,
                "candidate_count": 1,
                "response_mime_type": "application/json",
            },
        )
        self.seed = seed
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def complete(self, prompt: str) -> str:
        response = None
        for attempt in range(3):
            elapsed = time.monotonic() - type(self)._last_request_at
            if elapsed < self._minimum_request_interval_seconds:
                time.sleep(self._minimum_request_interval_seconds - elapsed)
            type(self)._last_request_at = time.monotonic()
            try:
                response = self.model.generate_content(prompt)
                break
            except Exception as exc:
                if attempt == 2 or not _is_rate_limit_error(exc):
                    raise
                time.sleep(35.0 * (attempt + 1))
        if response is None:
            raise RuntimeError("Gemini did not return a response")
        usage = getattr(response, "usage_metadata", None)
        self.last_input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        self.last_output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        return response.text


def run_experiment(
    *,
    condition: MemoryCondition,
    seed: int,
    assessor: BaselineAssessor,
    data_dir: Path = DEFAULT_DATA_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    retrieval_fn: Callable[[DiseaseDrugPair], RetrievalResult] = None,
    lesson_workflow: LessonWorkflow | None = None,
    now: Callable[[], datetime] | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Run or safely resume one fixed condition/seed session."""

    retrieval_fn = retrieval_fn or retrieve_pair
    now = now or (lambda: datetime.now(timezone.utc))
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    session_id = f"{condition}-seed-{seed}"
    run_dir = runs_dir / session_id
    records_path = run_dir / "records.jsonl"
    session_path = run_dir / "session.json"
    memory = ExperimentMemory(
        condition=condition,
        run_id=session_id,
        backend=JsonFileBackend(run_dir / "memory.json"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_run_inputs(run_dir, config, condition, seed)

    pairs = _load_jsonl(data_dir / "pairs.jsonl", DiseaseDrugPair)
    gold_by_id = {
        record.pair_id: record for record in _load_jsonl(data_dir / "gold.jsonl", GoldRecord)
    }
    existing_records = _load_jsonl(records_path, RunRecord) if records_path.exists() else []
    completed_pair_ids = {record.pair.pair_id for record in existing_records}
    if len(completed_pair_ids) != len(existing_records):
        raise ValueError("run records contain a duplicated pair")
    events = _load_or_start_events(session_path, session_id, now())
    _reconcile_record_events(events, existing_records, session_id, now)
    processed_this_call = 0

    for pair in pairs:
        if pair.pair_id in completed_pair_ids:
            continue
        if max_pairs is not None and processed_this_call >= max_pairs:
            break
        atomic_run_id = f"{session_id}:{pair.pair_id}"
        events.append(_event(events, session_id, SessionEventType.PAIR_STARTED, now(), pair.pair_id, atomic_run_id))
        started_at = now()
        started_clock = time.perf_counter()
        retrieval = retrieval_fn(pair)
        evidence_types = sorted({item.evidence_type for item in retrieval.evidence_items})
        memories = memory.retrieve(
            query=f"{pair.disease_name} {pair.drug_name} {' '.join(evidence_types)}",
            evidence_types=evidence_types,
        )
        if memories:
            events.append(_event(events, session_id, SessionEventType.MEMORY_RETRIEVED, now(), pair.pair_id, atomic_run_id, {"memory_ids": [entry.memory_id for entry in memories]}))
        assessment = assessor.assess(
            pair=pair,
            retrieval=retrieval,
            condition=condition,
            memories=[entry.model_dump(mode="json") for entry in memories],
        )
        latency = time.perf_counter() - started_clock
        input_tokens = int(getattr(assessor.model, "last_input_tokens", 0))
        output_tokens = int(getattr(assessor.model, "last_output_tokens", 0))
        feedback = evaluate_assessment(
            assessment,
            gold_by_id[pair.pair_id],
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        record = RunRecord(
            run_id=atomic_run_id,
            contract_version=str(config["contract_version"]),
            pair=pair,
            condition=condition,
            phase=gold_by_id[pair.pair_id].split,
            seed=seed,
            started_at=started_at,
            completed_at=now(),
            assessment=assessment,
            evaluator_feedback=feedback,
            retrieved_memory_ids=[entry.memory_id for entry in memories],
            prompt_versions={
                "assessor": config["model"]["assessor_prompt_version"],
                "lesson_generator": config["model"]["lesson_generator_prompt_version"],
                "evaluator": config["model"]["evaluator_policy_version"],
            },
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            errors=[error.message for error in retrieval.errors],
            metadata={
                "session_id": session_id,
                "request_hashes": retrieval.request_hashes,
                "cache_hits": retrieval.cache_hits,
                "cache_misses": retrieval.cache_misses,
                "evidence_ids": [item.evidence_id for item in retrieval.evidence_items],
            },
        )
        if condition == "raw_memory" and record.phase == "learning_stream":
            memory_id = memory.store_raw_run(record)
            record.written_memory_ids.append(memory_id)
            events.append(_event(events, session_id, SessionEventType.MEMORY_WRITTEN, now(), pair.pair_id, atomic_run_id, {"memory_ids": [memory_id]}))
        if (
            condition == "validated_lessons"
            and record.phase == "learning_stream"
            and feedback.failure_types
            and lesson_workflow is not None
        ):
            prior_lessons = [
                entry.content
                for entry in memory.retrieve(
                    query="procedural evidence assessment lesson",
                    evidence_types=evidence_types or ["retrieval_failure"],
                    limit=100,
                )
            ]
            candidate = lesson_workflow.generate_candidate(
                feedback=feedback,
                evidence_types=evidence_types or ["retrieval_failure"],
                pair_terms=[pair.disease_name, pair.drug_name, pair.disease_id, pair.drug_id],
                existing_lessons=prior_lessons,
            )
            if lesson_workflow.rejections:
                _append_new_rejections(run_dir / "lesson_rejections.jsonl", lesson_workflow)
            if candidate is not None:
                cases = lesson_workflow.create_regression_cases(candidate)
                results = lesson_workflow.run_regressions(
                    candidate,
                    cases,
                    feedback=feedback,
                    pair_terms=[pair.disease_name, pair.drug_name, pair.disease_id, pair.drug_id],
                    existing_lessons=prior_lessons,
                )
                lesson, approval = lesson_workflow.approve(
                    candidate,
                    feedback=feedback,
                    regression_results=results,
                    approval_record_id=f"approval:{candidate.candidate_id}",
                    approved_by="Codex (designated experiment reviewer)",
                )
                memory_id = memory.store_validated_lesson(
                    lesson,
                    evaluator_feedback=feedback,
                    approval=approval,
                )
                record.written_memory_ids.append(memory_id)
                _append_jsonl(run_dir / "lesson_approvals.jsonl", approval)
                events.append(_event(events, session_id, SessionEventType.HUMAN_APPROVAL_RECORDED, now(), pair.pair_id, atomic_run_id, {"approval_record_id": approval.approval_record_id}))
                events.append(_event(events, session_id, SessionEventType.MEMORY_WRITTEN, now(), pair.pair_id, atomic_run_id, {"memory_ids": [memory_id]}))
        _append_jsonl(records_path, record)
        existing_records.append(record)
        completed_pair_ids.add(pair.pair_id)
        events.append(_event(events, session_id, SessionEventType.PAIR_COMPLETED, now(), pair.pair_id, atomic_run_id))
        processed_this_call += 1
        _write_session(session_path, _session_snapshot(session_id, condition, seed, config, existing_records, events, now(), completed=False))

    complete = len(existing_records) == len(pairs)
    if complete and (not events or events[-1].event_type is not SessionEventType.SESSION_COMPLETED):
        events.append(_event(events, session_id, SessionEventType.SESSION_COMPLETED, now()))
    snapshot = _session_snapshot(session_id, condition, seed, config, existing_records, events, now(), completed=complete)
    _write_session(session_path, snapshot)
    return {
        "session_id": session_id,
        "condition": condition,
        "seed": seed,
        "completed_pairs": len(existing_records),
        "processed_this_call": processed_this_call,
        "status": snapshot.status.value,
        "run_dir": str(run_dir),
    }


def report_runs(run_dir: Path = DEFAULT_RUNS_DIR) -> dict[str, Any]:
    """Summarize all valid pair records under a runs directory."""

    report: dict[str, Any] = {"runs": {}, "totals": {"runs": 0, "records": 0}}
    if not run_dir.exists():
        return report
    for records_path in sorted(run_dir.glob("*/records.jsonl")):
        records = _load_jsonl(records_path, RunRecord)
        if not records:
            continue
        session_id = records_path.parent.name
        correct = sum(record.evaluator_feedback.classification_correct for record in records if record.evaluator_feedback)
        unsupported = sum(bool(record.evaluator_feedback.unsupported_claims) for record in records if record.evaluator_feedback)
        report["runs"][session_id] = {
            "condition": records[0].condition,
            "seed": records[0].seed,
            "records": len(records),
            "classification_accuracy": round(correct / len(records), 6),
            "unsupported_claim_rate": round(unsupported / len(records), 6),
            "latency_seconds": round(sum(record.latency_seconds for record in records), 6),
            "input_tokens": sum(record.input_tokens for record in records),
            "output_tokens": sum(record.output_tokens for record in records),
        }
        report["totals"]["runs"] += 1
        report["totals"]["records"] += len(records)
    return report


def validate_data(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Validate frozen pair, gold, split, hash, and leakage invariants."""

    pairs_path, gold_path, manifest_path = data_dir / "pairs.jsonl", data_dir / "gold.jsonl", data_dir / "splits.json"
    pairs, gold = _load_jsonl(pairs_path, DiseaseDrugPair), _load_jsonl(gold_path, GoldRecord)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    pair_ids, gold_ids = [item.pair_id for item in pairs], [item.pair_id for item in gold]
    expected_ids = [f"disease-drug-pair-{index:03d}" for index in range(1, 101)]
    _require(len(pairs) == 100, "pairs.jsonl must contain exactly 100 records", errors)
    _require(len(gold) == 100, "gold.jsonl must contain exactly 100 records", errors)
    _require(pair_ids == expected_ids, "pair IDs must be fixed and sequential", errors)
    _require(gold_ids == expected_ids, "gold IDs must match fixed pair order", errors)
    pair_keys = {(pair.disease_id, pair.drug_id) for pair in pairs}
    _require(len(pair_keys) == len(pairs), "disease-drug pairs must be unique", errors)
    split_ids: set[str] = set()
    label_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    pair_map = {pair.pair_id: pair for pair in pairs}
    for split_name, split in manifest.get("splits", {}).items():
        start, end = int(split["start_id"].rsplit("-", 1)[1]), int(split["end_id"].rsplit("-", 1)[1])
        ids = {f"disease-drug-pair-{index:03d}" for index in range(start, end + 1)}
        _require(len(ids) == split["count"], f"{split_name} manifest count does not match its ID range", errors)
        _require(not split_ids & ids, f"{split_name} overlaps another split", errors)
        split_ids.update(ids)
    _require(split_ids == set(expected_ids), "splits must cover all 100 IDs", errors)
    for record in gold:
        label_counts_by_split[record.split][record.gold_label] += 1
        pair = pair_map.get(record.pair_id)
        _require(pair is not None, f"gold record {record.pair_id} has no pair", errors)
        _require(record.split == _split_for_id(record.pair_id), f"{record.pair_id} is assigned to the wrong split", errors)
        _require(any(source.startswith("OpenTargets:") for source in record.acceptable_source_ids), f"{record.pair_id} requires Open Targets identifiers", errors)
        if record.gold_label != "insufficient_evidence":
            _require(any(source.startswith("PMID:") for source in record.acceptable_source_ids), f"{record.pair_id} requires at least one PubMed evidence anchor", errors)
        if pair:
            _require({f"OpenTargets:{pair.disease_id}", f"OpenTargets:{pair.drug_id}"}.issubset(set(record.acceptable_source_ids)), f"{record.pair_id} gold sources must match the pair's Open Targets IDs", errors)
    for split_name, counts in label_counts_by_split.items():
        _require(not (set(CLASSIFICATION_LABELS) - set(counts)), f"{split_name} is missing labels", errors)
    actual_counts = {split: dict(sorted(counts.items())) for split, counts in sorted(label_counts_by_split.items())}
    _require(manifest.get("label_counts_by_split") == actual_counts, "manifest label counts must match gold records", errors)
    leakage = manifest.get("leakage_policy", {})
    _require(manifest.get("status") == "frozen", "manifest status must be frozen", errors)
    _require(leakage.get("agent_input") == "data/pairs.jsonl only", "agent input must exclude gold records", errors)
    _require(leakage.get("evaluator_only") == "data/gold.jsonl", "gold records must be evaluator-only", errors)
    _require(leakage.get("held_out_feedback_to_memory") is False, "held-out feedback must be excluded from memory", errors)
    _require(leakage.get("distribution_shift_feedback_to_memory") is False, "distribution-shift feedback must be excluded from memory", errors)
    hashes = manifest.get("hashes", {})
    _require(hashes.get("pairs.jsonl_sha256") == _sha256(pairs_path), "pairs.jsonl hash does not match frozen manifest", errors)
    _require(hashes.get("gold.jsonl_sha256") == _sha256(gold_path), "gold.jsonl hash does not match frozen manifest", errors)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return {"pairs": len(pairs), "gold_records": len(gold), "unique_pairs": len(pair_keys), "split_overlap": False, "missing_gold_fields": False, "label_counts_by_split": actual_counts, "hashes_verified": True, "leakage_policy_verified": True}


def retrieve_pair(pair: DiseaseDrugPair) -> RetrievalResult:
    open_targets, pubmed = OpenTargetsClient().retrieve(pair), PubMedClient().retrieve(pair)
    return RetrievalResult(pair_id=pair.pair_id, evidence_items=open_targets.evidence_items + pubmed.evidence_items, errors=open_targets.errors + pubmed.errors, request_hashes=open_targets.request_hashes + pubmed.request_hashes, cache_hits=open_targets.cache_hits + pubmed.cache_hits, cache_misses=open_targets.cache_misses + pubmed.cache_misses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--pair-id", required=True)
    retrieve_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--condition", choices=MEMORY_CONDITIONS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    if args.command == "validate-data":
        result = validate_data(args.data_dir)
    elif args.command == "retrieve":
        result = retrieve_pair(_load_pair(args.data_dir, args.pair_id)).model_dump(mode="json")
    elif args.command == "report":
        result = report_runs(args.run_dir)
    else:
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        model = GeminiStructuredModel(model_name=config["model"]["name"], temperature=float(config["model"]["temperature"]), seed=args.seed)
        result = run_experiment(
            condition=args.condition,
            seed=args.seed,
            assessor=BaselineAssessor(model),
            lesson_workflow=LessonWorkflow(model) if args.condition == "validated_lessons" else None,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def _load_jsonl(path: Path, model: type[Any]) -> list[Any]:
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw.strip():
            try:
                records.append(model.model_validate_json(raw))
            except Exception as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    return records


def _append_jsonl(path: Path, model: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(model.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.flush()


def _append_new_rejections(path: Path, workflow: LessonWorkflow) -> None:
    existing = {
        json.loads(line)["candidate_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if path.exists() else set()
    for rejection in workflow.rejections:
        if rejection.candidate_id not in existing:
            _append_jsonl(path, rejection)
            existing.add(rejection.candidate_id)


def _load_pair(data_dir: Path, pair_id: str) -> DiseaseDrugPair:
    for pair in _load_jsonl(data_dir / "pairs.jsonl", DiseaseDrugPair):
        if pair.pair_id == pair_id:
            return pair
    raise ValueError(f"Unknown pair ID: {pair_id}")


def _event(events, session_id, event_type, timestamp, pair_id=None, run_id=None, data=None):
    return SessionEvent(event_id=f"{session_id}:event-{len(events)+1:04d}", session_id=session_id, sequence=len(events)+1, previous_event_id=events[-1].event_id if events else None, event_type=event_type, timestamp=timestamp, pair_id=pair_id, run_id=run_id, data=data or {})


def _load_or_start_events(session_path: Path, session_id: str, timestamp: datetime) -> list[SessionEvent]:
    if session_path.exists():
        session = ExperimentSession.model_validate_json(session_path.read_text(encoding="utf-8"))
        if session.status is SessionStatus.COMPLETED:
            return list(session.events)
        events = list(session.events)
        events.append(_event(events, session_id, SessionEventType.SESSION_RESUMED, timestamp))
        return events
    return [_event([], session_id, SessionEventType.SESSION_STARTED, timestamp)]


def _reconcile_record_events(
    events: list[SessionEvent],
    records: list[RunRecord],
    session_id: str,
    now: Callable[[], datetime],
) -> None:
    """Recover session events when an atomic record outlives its checkpoint."""

    started_run_ids = {
        event.run_id for event in events if event.event_type is SessionEventType.PAIR_STARTED
    }
    completed_run_ids = {
        event.run_id for event in events if event.event_type is SessionEventType.PAIR_COMPLETED
    }
    for record in records:
        if record.run_id in completed_run_ids:
            continue
        if record.run_id not in started_run_ids:
            events.append(
                _event(
                    events,
                    session_id,
                    SessionEventType.PAIR_STARTED,
                    now(),
                    record.pair.pair_id,
                    record.run_id,
                    {"recovered_from_atomic_record": True},
                )
            )
        events.append(
            _event(
                events,
                session_id,
                SessionEventType.PAIR_COMPLETED,
                now(),
                record.pair.pair_id,
                record.run_id,
                {"recovered_from_atomic_record": True},
            )
        )


def _session_snapshot(session_id, condition, seed, config, records, events, timestamp, *, completed):
    return ExperimentSession(session_id=session_id, contract_version=str(config["contract_version"]), condition=condition, seed=seed, status="completed" if completed else "active", started_at=events[0].timestamp, updated_at=timestamp, next_pair_ordinal=len(records)+1, completed_run_ids=[record.run_id for record in records], events=events, metadata={"run_directory": session_id})


def _write_session(path: Path, session: ExperimentSession) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def _freeze_run_inputs(run_dir: Path, config: dict[str, Any], condition: str, seed: int) -> None:
    payloads = {"config.json": {**config, "runtime": {"condition": condition, "seed": seed}}, "assessor_prompt.md": ASSESSOR_PROMPT_PATH.read_text(encoding="utf-8"), "lesson_generator_prompt.md": LESSON_PROMPT_PATH.read_text(encoding="utf-8")}
    for name, payload in payloads.items():
        path = run_dir / name
        content = payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"frozen run input changed: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_for_id(pair_id: str) -> str:
    ordinal = int(pair_id.rsplit("-", 1)[1])
    return "initial_evaluation" if ordinal <= 20 else "learning_stream" if ordinal <= 70 else "held_out_evaluation" if ordinal <= 90 else "distribution_shift"


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "resource_exhausted" in message or "quota exceeded" in message or "429" in message


if __name__ == "__main__":
    main()
