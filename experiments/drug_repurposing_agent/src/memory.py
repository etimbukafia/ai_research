"""Condition-isolated memory policies with a lazy Mem0 integration adapter."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import Field

from experiments.drug_repurposing_agent.src.models import (
    EvaluatorFeedback,
    Lesson,
    LessonApprovalRecord,
    MemoryCondition,
    RunRecord,
    StrictModel,
)

MemoryKind = Literal["raw_trajectory", "validated_lesson"]


class MemoryPolicyError(ValueError):
    """Raised when a memory operation violates the experiment contract."""


class MemoryEntry(StrictModel):
    """One condition- and run-isolated record returned to the assessor."""

    memory_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    condition: MemoryCondition
    run_id: str = Field(min_length=1)
    kind: MemoryKind
    source_task: str = Field(pattern=r"^disease-drug-pair-\d{3}$")
    content: str = Field(min_length=1)
    applicable_evidence_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryBackend(Protocol):
    """Storage operations required from Mem0 or a deterministic test backend."""

    def add(self, *, content: str, metadata: dict[str, Any], namespace: str) -> str:
        """Store one immutable memory and return its ID."""

    def search(
        self,
        *,
        query: str,
        namespace: str,
        filters: dict[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return candidate records from exactly one namespace."""


class ExperimentMemory:
    """Apply frozen memory-condition, phase, validation, and isolation policies."""

    def __init__(
        self,
        *,
        condition: MemoryCondition,
        run_id: str,
        backend: MemoryBackend,
    ) -> None:
        self.condition = condition
        self.run_id = run_id
        self.backend = backend
        self.namespace = f"drug-repurposing-agent:{condition}:{run_id}"

    def retrieve(
        self,
        *,
        query: str,
        evidence_types: Sequence[str],
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Retrieve only memories allowed by this condition and run."""

        if self.condition == "no_memory":
            return []
        filters: dict[str, Any] = {
            "condition": self.condition,
            "run_id": self.run_id,
        }
        if self.condition == "validated_lessons":
            filters.update({"kind": "validated_lesson", "validated": True})
        candidates = self.backend.search(
            query=query,
            namespace=self.namespace,
            filters=filters,
            limit=limit,
        )
        entries = [self._validate_backend_entry(candidate) for candidate in candidates]
        if self.condition == "validated_lessons":
            requested = set(evidence_types)
            entries = [
                entry
                for entry in entries
                if requested & set(entry.applicable_evidence_types)
                and entry.metadata.get("validated") is True
            ]
        return entries[:limit]

    def store_raw_run(self, record: RunRecord) -> str:
        """Store a learning-stream trajectory and evaluator feedback."""

        self._require_condition("raw_memory")
        if record.condition != "raw_memory" or (
            record.run_id != self.run_id
            and record.metadata.get("session_id") != self.run_id
        ):
            raise MemoryPolicyError("raw run must match the memory condition and run")
        feedback = record.evaluator_feedback
        if (
            record.phase != "learning_stream"
            or feedback is None
            or not feedback.learning_feedback_allowed
        ):
            raise MemoryPolicyError(
                "raw memory accepts only learning-stream runs with eligible feedback"
            )
        payload = {
            "pair": record.pair.model_dump(mode="json"),
            "assessment": record.assessment.model_dump(mode="json"),
            "evaluator_feedback": feedback.model_dump(mode="json"),
        }
        return self._add(
            kind="raw_trajectory",
            source_task=record.pair.pair_id,
            content=json.dumps(payload, sort_keys=True),
            applicable_evidence_types=sorted(
                {item.evidence_type for item in record.assessment.evidence_items}
            ),
            metadata={"phase": record.phase},
        )

    def store_validated_lesson(
        self,
        lesson: Lesson,
        *,
        evaluator_feedback: EvaluatorFeedback,
        approval: LessonApprovalRecord,
    ) -> str:
        """Store one approved procedural lesson from eligible learning feedback."""

        self._require_condition("validated_lessons")
        if (
            evaluator_feedback.pair_id != lesson.source_task
            or evaluator_feedback.phase != "learning_stream"
            or not evaluator_feedback.learning_feedback_allowed
        ):
            raise MemoryPolicyError(
                "validated lessons require matching eligible learning-stream feedback"
            )
        if (
            approval.candidate_id != lesson.lesson_id
            or approval.approval_record_id != lesson.provenance.approval_record_id
            or approval.evaluator_feedback_id != lesson.provenance.evaluator_feedback_id
        ):
            raise MemoryPolicyError(
                "validated lesson requires its matching explicit approval record"
            )
        lesson_payload = lesson.model_dump(mode="json")
        return self._add(
            kind="validated_lesson",
            source_task=lesson.source_task,
            content=lesson.lesson,
            applicable_evidence_types=lesson.applicable_evidence_types,
            metadata={
                "validated": lesson.validated,
                "lesson": lesson_payload,
                "confidence": lesson.confidence,
                "provenance": lesson.provenance.model_dump(mode="json"),
                "supersession": lesson.supersession.model_dump(mode="json"),
            },
            memory_id=lesson.lesson_id,
        )

    def _add(
        self,
        *,
        kind: MemoryKind,
        source_task: str,
        content: str,
        applicable_evidence_types: Sequence[str],
        metadata: dict[str, Any],
        memory_id: str | None = None,
    ) -> str:
        record_metadata = {
            **metadata,
            "memory_id": memory_id or str(uuid4()),
            "namespace": self.namespace,
            "condition": self.condition,
            "run_id": self.run_id,
            "kind": kind,
            "source_task": source_task,
            "applicable_evidence_types": list(applicable_evidence_types),
        }
        return self.backend.add(
            content=content,
            metadata=record_metadata,
            namespace=self.namespace,
        )

    def _validate_backend_entry(self, candidate: Mapping[str, Any]) -> MemoryEntry:
        metadata = dict(candidate.get("metadata") or {})
        content = str(candidate.get("content") or candidate.get("memory") or "")
        entry = MemoryEntry(
            memory_id=str(candidate.get("id") or metadata.get("memory_id") or ""),
            namespace=str(metadata.get("namespace") or ""),
            condition=metadata.get("condition"),
            run_id=str(metadata.get("run_id") or ""),
            kind=metadata.get("kind"),
            source_task=str(metadata.get("source_task") or ""),
            content=content,
            applicable_evidence_types=list(
                metadata.get("applicable_evidence_types") or []
            ),
            metadata=metadata,
        )
        if (
            entry.namespace != self.namespace
            or entry.condition != self.condition
            or entry.run_id != self.run_id
        ):
            raise MemoryPolicyError("backend returned a memory outside its namespace")
        return entry

    def _require_condition(self, required: MemoryCondition) -> None:
        if self.condition != required:
            raise MemoryPolicyError(
                f"{self.condition} memory cannot perform a {required} write"
            )


class InMemoryBackend:
    """Deterministic backend used by tests and local policy development."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.calls: list[str] = []

    def add(self, *, content: str, metadata: dict[str, Any], namespace: str) -> str:
        self.calls.append("add")
        memory_id = str(metadata["memory_id"])
        self.records.append(
            {
                "id": memory_id,
                "content": content,
                "metadata": dict(metadata),
                "namespace": namespace,
            }
        )
        return memory_id

    def search(
        self,
        *,
        query: str,
        namespace: str,
        filters: dict[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        self.calls.append("search")
        query_tokens = _tokens(query)
        matches = [
            record
            for record in self.records
            if record["namespace"] == namespace
            and all(record["metadata"].get(key) == value for key, value in filters.items())
        ]
        matches.sort(
            key=lambda record: len(query_tokens & _tokens(record["content"])),
            reverse=True,
        )
        return matches[:limit]


class JsonFileBackend(InMemoryBackend):
    """Persistent deterministic backend for resumable experiment runs."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.records = list(payload)

    def add(self, *, content: str, metadata: dict[str, Any], namespace: str) -> str:
        memory_id = str(metadata["memory_id"])
        if any(record["id"] == memory_id for record in self.records):
            return memory_id
        result = super().add(content=content, metadata=metadata, namespace=namespace)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary.replace(self.path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))
        return result


class Mem0Backend:
    """Thin adapter over `mem0ai`; import and client setup remain explicit."""

    def __init__(self, memory: Any | None = None) -> None:
        if memory is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise RuntimeError(
                    "mem0ai is required for Mem0Backend; install pinned requirements"
                ) from exc
            memory = Memory()
        self.memory = memory

    def add(self, *, content: str, metadata: dict[str, Any], namespace: str) -> str:
        result = self.memory.add(content, user_id=namespace, metadata=metadata)
        if isinstance(result, Mapping):
            results = result.get("results") or []
            if results:
                return str(results[0].get("id") or metadata["memory_id"])
        return str(metadata["memory_id"])

    def search(
        self,
        *,
        query: str,
        namespace: str,
        filters: dict[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        result = self.memory.search(
            query=query,
            user_id=namespace,
            filters=filters,
            limit=limit,
        )
        if isinstance(result, Mapping):
            return list(result.get("results") or [])
        return list(result or [])


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))
