from pathlib import Path

import pytest

from enterprise_agent_lab.config import ConfigurationError, load_config
from enterprise_agent_lab.cases import build_cases
from enterprise_agent_lab.runner import run_case


def test_replay_featured_case_writes_complete_trace(tmp_path: Path):
    trace = run_case("case-01", config=load_config(tmp_path))
    assert trace.mode == "replay"
    assert trace.final_decision.status == "needs_human_review"
    assert trace.final_decision.draft_action.requires_approval is True
    assert trace.tool_calls
    assert (tmp_path / "runs" / f"{trace.run_id}.json").exists()


def test_all_cases_have_replay_results(tmp_path: Path):
    config = load_config(tmp_path)
    for case in build_cases():
        trace = run_case(case.case_id, config=config)
        assert trace.final_decision is not None
        assert trace.error is None


def test_live_mode_fails_before_setup_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = load_config(tmp_path)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        run_case("case-01", mode="live", config=config)
    assert not (tmp_path / "runs").exists()

