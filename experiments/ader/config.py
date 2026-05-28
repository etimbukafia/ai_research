from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = APP_ROOT / ".env"


@dataclass(frozen=True, slots=True)
class Config:

    GEMINI_API_KEY: str

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

        config = cls(
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        missing = [
            name
            for name, value in [
                ("GEMINI_API_KEY", self.GEMINI_API_KEY),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

# Create a singleton instance to be imported across the project
# You can import it with: from experiments.sport_news_agent.src.config import app_config
app_config = Config.load()
