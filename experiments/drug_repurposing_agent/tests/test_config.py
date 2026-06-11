from pathlib import Path

import pytest

from experiments.drug_repurposing_agent.config import Config


def test_config_loads_gemini_key_from_explicit_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")

    assert Config.load(env_path).GEMINI_API_KEY == "test-key"


def test_config_rejects_missing_gemini_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Config.load(tmp_path / "missing.env")
