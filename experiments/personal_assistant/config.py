from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = APP_ROOT / ".env"
DEFAULT_AGENT_MODEL_VALUE = "gemini-3.1-flash-lite"


@dataclass(frozen=True, slots=True)
class Config:

    GEMINI_API_KEY: str
    DEFAULT_AGENT_MODEL: str = DEFAULT_AGENT_MODEL_VALUE
    PERSONAL_ASSISTANT_AGENT_MODEL: str = ""
    CONTEXT_SYNTHESIZER_AGENT_MODEL: str = ""
    PLANNER_AGENT_MODEL: str = ""
    VERIFIER_AGENT_MODEL: str = ""
    INTENT_CLASSIFIER_AGENT_MODEL: str = ""

    @classmethod
    def load(cls, env_path: Path = DEFAULT_ENV_FILE) -> Config:
        """
        Load and validate configuration from the given .env file.

        Args:
            env_path: Path to the .env file. Falls back to system env if not found.

        Returns:
            A validated Config instance.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        load_dotenv(env_path)
        default_agent_model = _env_or("DEFAULT_AGENT_MODEL", DEFAULT_AGENT_MODEL_VALUE)

        config = cls(
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
            DEFAULT_AGENT_MODEL=default_agent_model,
            PERSONAL_ASSISTANT_AGENT_MODEL=_env_or("PERSONAL_ASSISTANT_AGENT_MODEL", default_agent_model),
            CONTEXT_SYNTHESIZER_AGENT_MODEL=_env_or(
                "CONTEXT_SYNTHESIZER_AGENT_MODEL",
                _env_or("MEMORY_SYNTHESIZER_AGENT_MODEL", default_agent_model),
            ),
            PLANNER_AGENT_MODEL=_env_or("PLANNER_AGENT_MODEL", default_agent_model),
            VERIFIER_AGENT_MODEL=_env_or("VERIFIER_AGENT_MODEL", default_agent_model),
            INTENT_CLASSIFIER_AGENT_MODEL=_env_or("INTENT_CLASSIFIER_AGENT_MODEL", default_agent_model),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        missing = [
            name
            for name, value in [
                ("GEMINI_API_KEY", self.GEMINI_API_KEY),
                ("DEFAULT_AGENT_MODEL", self.DEFAULT_AGENT_MODEL),
                ("PERSONAL_ASSISTANT_AGENT_MODEL", self.PERSONAL_ASSISTANT_AGENT_MODEL),
                ("CONTEXT_SYNTHESIZER_AGENT_MODEL", self.CONTEXT_SYNTHESIZER_AGENT_MODEL),
                ("PLANNER_AGENT_MODEL", self.PLANNER_AGENT_MODEL),
                ("VERIFIER_AGENT_MODEL", self.VERIFIER_AGENT_MODEL),
                ("INTENT_CLASSIFIER_AGENT_MODEL", self.INTENT_CLASSIFIER_AGENT_MODEL),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


def _env_or(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


app_config = Config.load()
