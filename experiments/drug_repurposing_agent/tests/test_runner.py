from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from experiments.drug_repurposing_agent.src.agent import BaselineAssessor
from experiments.drug_repurposing_agent.src.lessons import LessonWorkflow
from experiments.drug_repurposing_agent.src.models import Assessment, ExperimentSession, RunRecord
from experiments.drug_repurposing_agent.src.retrieval import RetrievalResult
from experiments.drug_repurposing_agent.src.runner import report_runs, run_experiment

NOW = datetime(2026, 6, 11, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        self.value += timedelta(seconds=1)
        return self.value


class AbstainingModel:
    def complete(self, prompt: str):
        payload = json.loads(prompt.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
        pair_id = payload["pair"]["pair_id"]
        return {
            "pair_id": pair_id,
            "label": "insufficient_evidence",
            "confidence": 0.9,
            "evidence_items": [],
            "relationships": [],
            "explanation": "No evidence was available for a directional judgment.",
            "limitations": ["No evidence was available."],
            "citations": [],
        }


class AssessorAndLessonModel(AbstainingModel):
    def complete(self, prompt: str):
        if "Procedural Lesson Generator" in prompt:
            payload = json.loads(prompt.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
            pair_id = payload["feedback"]["pair_id"]
            return {
                "candidate_id": f"lesson-{pair_id}",
                "lesson": "Assessment procedure should account for retrieval failure.",
                "failure_type": payload["feedback"]["failure_types"][0],
                "applicable_evidence_types": payload["applicable_evidence_types"],
                "source_task": pair_id,
                "confidence": 0.8,
                "feedback_basis": payload["feedback"]["feedback"],
                "supersession": {
                    "supersedes_lesson_ids": [],
                    "superseded_by_lesson_ids": [],
                },
                "scope": "procedural",
            }
        return super().complete(prompt)


def write_dataset(path: Path) -> None:
    path.mkdir()
    pairs = [
        {
            "pair_id": "disease-drug-pair-020",
            "disease_name": "Disease Initial",
            "disease_id": "MONDO_20",
            "drug_name": "Drug Initial",
            "drug_id": "CHEMBL20",
        },
        {
            "pair_id": "disease-drug-pair-021",
            "disease_name": "Disease Learning",
            "disease_id": "MONDO_21",
            "drug_name": "Drug Learning",
            "drug_id": "CHEMBL21",
        },
    ]
    gold = [
        {
            "pair_id": pair["pair_id"],
            "split": "initial_evaluation" if index == 0 else "learning_stream",
            "gold_label": "insufficient_evidence",
            "confidence": 0.9,
            "evidence_pattern": "test abstention",
            "expected_relationships": [],
            "acceptable_source_ids": [f"OpenTargets:{pair['disease_id']}", f"OpenTargets:{pair['drug_id']}"],
            "rationale": "No relevant evidence.",
            "known_contradictions": [],
            "reviewer": "Reviewer",
            "reviewed_at": NOW.isoformat(),
            "adjudication_notes": "Test.",
        }
        for index, pair in enumerate(pairs)
    ]
    (path / "pairs.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in pairs), encoding="utf-8"
    )
    (path / "gold.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in gold), encoding="utf-8"
    )


def retrieve(pair):
    return RetrievalResult(
        pair_id=pair.pair_id,
        request_hashes=[f"hash-{pair.pair_id}"],
        cache_hits=1,
    )


def test_runner_resumes_without_duplicate_pair_or_memory_write(tmp_path: Path) -> None:
    data_dir, runs_dir = tmp_path / "data", tmp_path / "runs"
    write_dataset(data_dir)
    assessor = BaselineAssessor(AbstainingModel())
    clock = Clock()

    first = run_experiment(
        condition="raw_memory",
        seed=42,
        assessor=assessor,
        data_dir=data_dir,
        runs_dir=runs_dir,
        retrieval_fn=retrieve,
        now=clock,
        max_pairs=1,
    )
    second = run_experiment(
        condition="raw_memory",
        seed=42,
        assessor=assessor,
        data_dir=data_dir,
        runs_dir=runs_dir,
        retrieval_fn=retrieve,
        now=clock,
    )
    third = run_experiment(
        condition="raw_memory",
        seed=42,
        assessor=assessor,
        data_dir=data_dir,
        runs_dir=runs_dir,
        retrieval_fn=retrieve,
        now=clock,
    )

    assert first["completed_pairs"] == 1
    assert second["completed_pairs"] == 2
    assert second["status"] == "completed"
    assert third["processed_this_call"] == 0
    run_dir = runs_dir / "raw_memory-seed-42"
    records = [
        RunRecord.model_validate_json(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    assert [record.pair.pair_id for record in records] == [
        "disease-drug-pair-020",
        "disease-drug-pair-021",
    ]
    assert records[0].written_memory_ids == []
    assert len(records[1].written_memory_ids) == 1
    memory_records = json.loads((run_dir / "memory.json").read_text())
    assert len(memory_records) == 1


def test_runner_persists_audit_artifacts_and_report(tmp_path: Path) -> None:
    data_dir, runs_dir = tmp_path / "data", tmp_path / "runs"
    write_dataset(data_dir)
    result = run_experiment(
        condition="no_memory",
        seed=7,
        assessor=BaselineAssessor(AbstainingModel()),
        data_dir=data_dir,
        runs_dir=runs_dir,
        retrieval_fn=retrieve,
        now=Clock(),
    )
    run_dir = Path(result["run_dir"])
    session = ExperimentSession.model_validate_json(
        (run_dir / "session.json").read_text(encoding="utf-8")
    )
    report = report_runs(runs_dir)

    assert session.status.value == "completed"
    assert len(session.completed_run_ids) == 2
    assert (run_dir / "config.json").exists()
    assert (run_dir / "assessor_prompt.md").exists()
    assert (run_dir / "lesson_generator_prompt.md").exists()
    assert report["totals"] == {"runs": 1, "records": 2}
    assert report["runs"]["no_memory-seed-7"]["classification_accuracy"] == 1.0


def test_validated_runner_writes_only_approved_learning_lesson(tmp_path: Path) -> None:
    data_dir, runs_dir = tmp_path / "data", tmp_path / "runs"
    write_dataset(data_dir)
    model = AssessorAndLessonModel()

    run_experiment(
        condition="validated_lessons",
        seed=42,
        assessor=BaselineAssessor(model),
        lesson_workflow=LessonWorkflow(model, now=Clock()),
        data_dir=data_dir,
        runs_dir=runs_dir,
        retrieval_fn=retrieve,
        now=Clock(),
    )

    run_dir = runs_dir / "validated_lessons-seed-42"
    records = [
        RunRecord.model_validate_json(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    assert records[0].written_memory_ids == []
    assert len(records[1].written_memory_ids) == 1
    assert (run_dir / "lesson_approvals.jsonl").exists()
