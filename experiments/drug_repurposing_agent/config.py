"""Environment-backed runtime configuration for the drug repurposing experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"


@dataclass(frozen=True, slots=True)
class Config:
    """Validated secrets and environment overrides used at runtime."""

    GEMINI_API_KEY: str

    @classmethod
    def load(cls, env_path: Path = DEFAULT_ENV_FILE) -> Config:
        """Load the repository `.env`, then validate required runtime values."""

        load_dotenv(env_path)
        config = cls(GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "").strip())
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.GEMINI_API_KEY:
            raise ValueError("Missing required environment variable: GEMINI_API_KEY")
