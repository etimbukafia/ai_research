"""Structured, deterministic boundary around the baseline evidence assessor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from experiments.drug_repurposing_agent.src.models import (
    Assessment,
    DiseaseDrugPair,
    MemoryCondition,
)
from experiments.drug_repurposing_agent.src.retrieval import RetrievalResult

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "assessor.md"


class AssessorError(ValueError):
    """Raised when model output violates the frozen assessor contract."""


class StructuredModel(Protocol):
    """Minimal model interface used by the runner and mocked by tests."""

    def complete(self, prompt: str) -> str | dict[str, Any] | Assessment:
        """Return one structured assessment response."""


class BaselineAssessor:
    """Build assessor input, call a model, and reject invalid output."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self.model = model
        self.instructions = prompt_path.read_text(encoding="utf-8").strip()

    def assess(
        self,
        *,
        pair: DiseaseDrugPair,
        retrieval: RetrievalResult,
        condition: MemoryCondition = "no_memory",
        memories: Sequence[dict[str, Any]] = (),
    ) -> Assessment:
        """Assess one pair using only supplied evidence and condition memories."""

        if retrieval.pair_id != pair.pair_id:
            raise AssessorError("retrieval pair_id does not match the requested pair")
        if condition == "no_memory" and memories:
            raise AssessorError("no_memory assessments cannot receive memories")

        prompt = self._render_prompt(pair, retrieval, condition, memories)
        raw_output = self.model.complete(prompt)
        for attempt in range(4):
            try:
                assessment = _parse_assessment(raw_output)
                _post_validate_assessment(pair, retrieval, assessment)
                return assessment
            except AssessorError as exc:
                if attempt == 3:
                    raise
                raw_output = self.model.complete(
                    f"{prompt}\n\n## Structured Repair\n\n"
                    "The previous JSON response violated the Assessment contract:\n"
                    f"{exc}\n\nReturn a corrected Assessment JSON object only. "
                    "Set explanation to the exact character-for-character value of "
                    "one citations[].claim entry. Every relationship claim must also "
                    "appear character-for-character as a citations[].claim entry.\n"
                )
        raise AssessorError("model failed to return a valid Assessment")

    def _render_prompt(
        self,
        pair: DiseaseDrugPair,
        retrieval: RetrievalResult,
        condition: MemoryCondition,
        memories: Sequence[dict[str, Any]],
    ) -> str:
        payload = {
            "assessment_json_schema": Assessment.model_json_schema(),
            "pair": pair.model_dump(mode="json"),
            "retrieval": retrieval.model_dump(mode="json"),
            "condition": condition,
            "memories": list(memories),
        }
        return (
            f"{self.instructions}\n\n"
            "## Runtime Input\n\n"
            f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n"
        )


def _parse_assessment(raw_output: str | dict[str, Any] | Assessment) -> Assessment:
    try:
        if isinstance(raw_output, Assessment):
            return raw_output
        if isinstance(raw_output, str):
            return Assessment.model_validate_json(raw_output)
        if isinstance(raw_output, dict):
            return Assessment.model_validate(raw_output)
    except (ValidationError, ValueError, TypeError) as exc:
        raise AssessorError(f"model returned an invalid Assessment: {exc}") from exc
    raise AssessorError("model must return an Assessment, JSON object, or JSON string")


def _post_validate_assessment(
    pair: DiseaseDrugPair,
    retrieval: RetrievalResult,
    assessment: Assessment,
) -> None:
    if assessment.pair_id != pair.pair_id:
        raise AssessorError("assessment pair_id does not match the requested pair")

    supplied = [item.model_dump(mode="json") for item in retrieval.evidence_items]
    returned = [item.model_dump(mode="json") for item in assessment.evidence_items]
    if returned != supplied:
        raise AssessorError(
            "assessment evidence_items must exactly match supplied retrieved evidence"
        )

    required_label = retrieval.required_abstention_label
    if not retrieval.evidence_items:
        required_label = "insufficient_evidence"
    if required_label is not None and assessment.label != required_label:
        raise AssessorError(
            f"available evidence requires the {required_label} abstention label"
        )
