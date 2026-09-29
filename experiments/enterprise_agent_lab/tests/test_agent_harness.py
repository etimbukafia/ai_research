from __future__ import annotations

import asyncio
from collections import Counter

from enterprise_agent_lab.config import load_config
from enterprise_agent_lab.experiments.agent_harness.run_experiment import (
    CONDITIONS,
    HarnessReplayRuntime,
    _candidate,
    _decision_metrics,
    build_dataset,
)


def test_agent_harness_dataset_has_paired_splits_and_labels() -> None:
    dataset = build_dataset()

    assert dataset.dataset_id == "enterprise_agent_harness"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 24
    assert len({case.metadata["pair_id"] for case in dataset.cases}) == 12
    assert Counter(case.split.value for case in dataset.cases) == {
        "smoke": 4,
        "development": 8,
        "regression": 6,
        "holdout": 4,
        "security": 2,
    }

    for pair_id in {case.metadata["pair_id"] for case in dataset.cases}:
        variants = [case for case in dataset.cases if case.metadata["pair_id"] == pair_id]
        assert {case.metadata["variant"] for case in variants} == {"clean", "distractor"}
        distractor = next(case for case in variants if case.metadata["variant"] == "distractor")
        assert distractor.metadata["distractor_types"]
        assert distractor.metadata["distractor_source_ids"]
        assert distractor.metadata["distractor_reason"]


def test_protected_arguments_include_time_boundaries() -> None:
    dataset = build_dataset()
    case = next(case for case in dataset.cases if case.metadata["base_case_id"] == "case-01")
    protected = {
        argument
        for expectation in case.tool_expectations
        for argument in expectation.protected_arguments
    }

    assert {"account_id", "period_start", "period_end"} <= protected


def test_raw_distractor_is_rejected_and_catalog_resists_it() -> None:
    dataset = build_dataset()
    case = next(
        case
        for case in dataset.cases
        if case.metadata["pair_id"] == "pair-04" and case.metadata["variant"] == "distractor"
    )
    runtime = HarnessReplayRuntime(load_config())

    raw_trace = asyncio.run(runtime.execute(case, _candidate("test-raw", "raw_schema")))
    catalog_trace = asyncio.run(
        runtime.execute(case, _candidate("test-catalog", "semantic_catalog"))
    )

    raw_metrics = _decision_metrics(case, raw_trace)
    catalog_metrics = _decision_metrics(case, catalog_trace)
    assert raw_metrics["forbidden_sources_used"]
    assert raw_metrics["unsafe_action"] is True
    assert raw_metrics["metrics"]["enterprise.distractor_resistance"] == 0.0
    assert catalog_metrics["forbidden_sources_used"] == []
    assert catalog_metrics["metrics"]["enterprise.distractor_resistance"] == 1.0


def test_experiment_conditions_are_fixed() -> None:
    assert CONDITIONS == ("raw_schema", "prose_rag", "semantic_catalog")

