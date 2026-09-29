from pathlib import Path

from assistant_harness import ResponseStatus

from enterprise_agent_lab.cases import build_cases
from enterprise_agent_lab.config import load_config
from enterprise_agent_lab.dependencies import build_components
from enterprise_agent_lab.integrations.agent_improvement import (
    build_dataset,
    run_replay_evaluation,
    trace_to_agent_trace,
)
from enterprise_agent_lab.integrations.assistant_harness import (
    build_enterprise_read_assistant,
)
from enterprise_agent_lab.runner import run_case


def test_assistant_harness_wraps_scoped_semantic_search(tmp_path: Path):
    config = load_config(tmp_path)
    components = build_components(
        config,
        condition="semantic_catalog",
        account_scope="account-01",
        case_time="2026-05-31",
    )
    try:
        assistant = build_enterprise_read_assistant(
            components.retriever,
            account_scope="account-01",
        )
        response = assistant.respond("What does billable seats mean for the account?")
        assert response.status == ResponseStatus.GROUNDED
        assert response.selected_skill_id == "business_evidence"
        assert response.citations
        assert response.tool_calls[0].tool_name == "search_business_concepts"
    finally:
        components.store.close()


def test_agent_improvement_case_and_trace_adapters(tmp_path: Path):
    config = load_config(tmp_path)
    case = build_cases()[0]
    dataset = build_dataset([case])
    assert dataset.cases[0].case_id == case.case_id
    assert dataset.cases[0].tool_expectations[0].name == case.required_tools[0]

    trace = run_case(case.case_id, config=config)
    converted = trace_to_agent_trace(trace, candidate_id="candidate-test")
    assert converted.case_id == case.case_id
    assert converted.candidate_id == "candidate-test"
    assert converted.turns[0].tool_calls


def test_agent_improvement_runner_scores_one_enterprise_case(tmp_path: Path):
    result = run_replay_evaluation(
        case_ids=["case-02"],
        config=load_config(tmp_path),
    )
    assert len(result.traces) == 1
    assert len(result.report.case_results) == 1
    assert result.report.runtime_failures == ()
    assert result.report.case_results[0].passed is True

