"""Runtime configuration for local replay and Gemini live runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when a requested run cannot start with its current settings."""


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LabConfig:
    package_root: Path = PACKAGE_ROOT
    model_suffix: str = "gemini-3.5-flash-lite"
    api_key: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_minilm: bool = False
    source_db: Path = PACKAGE_ROOT / "runs" / "aster_cloud.sqlite3"
    run_dir: Path = PACKAGE_ROOT / "runs"
    review_dir: Path = PACKAGE_ROOT / "reviews"

    @property
    def model_name(self) -> str:
        return f"google:{self.model_suffix}"

    def require_live(self) -> None:
        if not self.api_key:
            raise ConfigurationError(
                "Live mode needs GOOGLE_API_KEY. Set GOOGLE_API_KEY and run again."
            )
        if "latest" in self.model_suffix.lower():
            raise ConfigurationError(
                "Live mode does not accept a moving latest model alias. "
                "Set GEMINI_MODEL to a pinned model name."
            )


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment settings while keeping all files inside the lab."""

    root = (base_dir or PACKAGE_ROOT).resolve()
    run_dir = root / "runs"
    return LabConfig(
        package_root=root,
        model_suffix=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=os.getenv("GOOGLE_API_KEY") or None,
        embedding_model=os.getenv(
            "EAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        use_minilm=_as_bool(os.getenv("EAL_USE_MINILM")),
        source_db=run_dir / "aster_cloud.sqlite3",
        run_dir=run_dir,
        review_dir=root / "reviews",
    )
