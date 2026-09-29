"""Checks for the checked-in Evals 101 replay contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_improvement_lab import DatasetSplit

from .runtime import build_dataset


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_NAMES = (
    "cases.json",
    "evaluator_coverage.json",
    "evaluator_fixtures.json",
    "traces.jsonl",
    "results.json",
    "featured_trace.json",
    "run_manifest.json",
    "comparison.json",
    "promotion_evaluation.json",
    "promotion_decision.json",
    "report.md",
)


def _json(name: str) -> dict:
    return json.loads((EXPERIMENT_DIR / name).read_text(encoding="utf-8"))


def test_case_and_pair_contract() -> None:
    dataset = build_dataset()
    assert len(dataset.cases) == 24
    assert len({case.metadata["pair_id"] for case in dataset.cases}) == 12
    assert {
        (case.metadata["pair_id"], case.metadata["variant"])
        for case in dataset.cases
    } == {
        (f"pair-{pair:02d}", variant)
        for pair in range(1, 13)
        for variant in ("clean", "distractor")
    }


def test_split_contract() -> None:
    dataset = build_dataset()
    counts = {
        split.value: sum(case.split == split for case in dataset.cases)
        for split in DatasetSplit
    }
    assert counts == {
        "smoke": 4,
        "development": 8,
        "regression": 6,
        "holdout": 4,
        "security": 2,
    }


def test_protected_arguments_are_declared() -> None:
    dataset = build_dataset()
    protected_names = {"account_id", "account_scope", "at", "period_start", "period_end"}
    for case in dataset.cases:
        for expectation in case.tool_expectations:
            for name in protected_names & set(expectation.exact_arguments):
                assert name in expectation.protected_arguments


def test_distractors_have_review_contracts() -> None:
    dataset = build_dataset()
    distractors = [
        case
        for case in dataset.cases
        if case.metadata.get("variant") == "distractor"
    ]
    assert len(distractors) == 12
    required = {
        "source_id",
        "distractor_type",
        "reason",
        "source_authority",
        "validity_period",
        "reviewer_note",
        "expected_response",
    }
    for case in distractors:
        records = case.metadata.get("distractor_records", [])
        assert records
        for record in records:
            assert required <= set(record)
            assert record["source_id"]
            assert record["validity_period"]["start"]
            assert record["validity_period"]["end"]


def test_evaluator_fixtures_cover_false_passes_and_valid_paths() -> None:
    fixtures = _json("evaluator_fixtures.json")["fixtures"]
    by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}
    assert {
        "correct_answer_wrong_source",
        "correct_answer_skipped_approval",
        "correct_answer_wrong_period_argument",
        "fluent_wrong_account",
        "correct_abstention_missing_evidence",
        "valid_alternate_tool_path",
    } == set(by_id)
    for fixture_id in (
        "correct_answer_wrong_source",
        "correct_answer_skipped_approval",
        "correct_answer_wrong_period_argument",
        "fluent_wrong_account",
    ):
        assert by_id[fixture_id]["answer_only_pass"] is True
        assert by_id[fixture_id]["enterprise_pass"] is False
    assert by_id["correct_abstention_missing_evidence"]["enterprise_pass"] is True
    assert by_id["valid_alternate_tool_path"]["enterprise_pass"] is True


def test_results_show_false_passes_and_promotion_gates() -> None:
    results = _json("results.json")
    baseline = results["candidates"]["baseline_false_pass"]
    assert baseline["answer_only"]["pass_rate"] == 1.0
    assert baseline["enterprise"]["pass_rate"] < baseline["answer_only"]["pass_rate"]
    assert baseline["false_pass_count"] > 0
    assert results["promotion_decisions"]["safe_candidate"]["outcome"] == "approved"
    assert results["promotion_decisions"]["fast_answer_candidate"]["outcome"] == "rejected"
    assert results["comparison"]["primary"]["holdout_checked"] is True


def test_artifact_hashes_match_manifest() -> None:
    manifest = _json("run_manifest.json")
    assert set(manifest["artifact_hashes"]) == set(ARTIFACT_NAMES) - {"run_manifest.json"}
    for name, expected in manifest["artifact_hashes"].items():
        digest = hashlib.sha256((EXPERIMENT_DIR / name).read_bytes()).hexdigest()
        assert digest == expected, name
