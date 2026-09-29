"""Trace creation and local JSON persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AgentDecision, PolicyCheck, RunTrace, ToolCallRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class TraceWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def write(self, trace: RunTrace) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"{trace.run_id}.json"
        path.write_text(
            json.dumps(trace.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def tool_records(raw_records: list[dict[str, Any]]) -> list[ToolCallRecord]:
        return [ToolCallRecord.model_validate(record) for record in raw_records]

    @staticmethod
    def policy_checks(decision: AgentDecision | None) -> list[PolicyCheck]:
        return decision.policy_checks if decision else []

