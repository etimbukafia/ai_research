"""Adapters from the enterprise lab to ``agent-improvement-lab``.

The enterprise lab owns domain cases and run traces. Agent Improvement Lab
owns generic trace contracts, evaluators, and experiment execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

from agent_improvement_lab import (
    AgentCandidate,
    AgentTrace,
    CaseProvenance,
    CandidateStatus,
    DatasetSplit,
    DatasetVersion,
    EvaluationCaseRef,
    PromptArtifact,
    PromptArtifactKind,
    RiskLevel,
    RunManifest,
    ToolCallExpectation,
)
from agent_improvement_lab.contracts.common import utc_now
from agent_improvement_lab.contracts.traces import (
    ObservedToolCall,
    ObservedTurn,
    TokenUsage,
    ToolCallOutcome,
)
from agent_improvement_lab.runner import EvaluationRunResult, PydanticEvalsRunner
from ..cases import build_cases
from ..config import LabConfig, load_config
from ..models import EvaluationCase, RunTrace
from ..runner import run_case


DATASET_ID = "enterprise_agent_semantic_retrieval"
DATASET_VERSION = "1.0.0"
RUNTIME_NAME = "enterprise-agent-lab"
RUNTIME_VERSION = "0.1.0"


def _split(case: EvaluationCase) -> DatasetSplit:
    if case.case_class == "concept_resolution":
        return DatasetSplit.SMOKE
    if case.case_class == "cross_system":
        return DatasetSplit.DEVELOPMENT
    if case.case_class == "policy_and_action":
        return DatasetSplit.SECURITY if case.approval_required else DatasetSplit.REGRESSION
    return DatasetSplit.REGRESSION


def _risk(case: EvaluationCase) -> RiskLevel:
    return RiskLevel.HIGH if case.approval_required else RiskLevel.MEDIUM


def _tool_expectations(case: EvaluationCase) -> tuple[ToolCallExpectation, ...]:
    expectations: list[ToolCallExpectation] = []
    for order, tool_name in enumerate(case.required_tools):
        exact_arguments = dict(case.expected_tool_arguments.get(tool_name, {}))
        protected_arguments = (
            ("account_id",) if "account_id" in exact_arguments else ()
        )
        expectations.append(
            ToolCallExpectation(
                name=tool_name,
                order=order,
                required_arguments=tuple(exact_arguments),
                exact_arguments=exact_arguments,
                protected_arguments=protected_arguments,
            )
        )
    return tuple(expectations)


def case_to_ref(case: EvaluationCase) -> EvaluationCaseRef:
    """Convert one enterprise case to the Lab's versioned case contract."""

    return EvaluationCaseRef(
        case_id=case.case_id,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        split=_split(case),
        risk=_risk(case),
        tags=tuple(
            [case.case_class]
            + (["approval_required"] if case.approval_required else [])
        ),
        input={
            "request": case.request,
            "account_id": case.account_id,
            "case_time": case.case_time,
        },
        expected={
            "status": case.expected_status,
            "required_concepts": list(case.required_concepts),
            "required_evidence_ids": list(case.required_evidence_ids),
            "claims": list(case.expected_claims),
            "action": case.expected_action,
            "approval_required": case.approval_required,
        },
        tool_expectations=_tool_expectations(case),
        provenance=CaseProvenance(
            source="enterprise_agent_lab.cases.build_cases",
            source_ref=case.case_id,
            notes="Synthetic Aster Cloud evaluation case.",
        ),
        metadata={
            "workflow": case.case_class,
            "authorized_tool_names": list(case.required_tools),
            "required_evidence_ids": list(case.required_evidence_ids),
        },
    )


def build_dataset(cases: Sequence[EvaluationCase] | None = None) -> DatasetVersion:
    """Build a versioned Lab dataset from the canonical enterprise cases."""

    selected = tuple(cases or build_cases())
    return DatasetVersion(
        dataset_id=DATASET_ID,
        version=DATASET_VERSION,
        description="Synthetic enterprise cases for semantic retrieval evaluation.",
        cases=tuple(case_to_ref(case) for case in selected),
        provenance=CaseProvenance(
            source="enterprise_agent_lab.cases.build_cases",
            notes="The enterprise lab remains the source of case content.",
        ),
        created_at=utc_now(),
        metadata={"case_count": str(len(selected))},
    )


def _parse_time(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _decision_text(trace: RunTrace) -> str:
    decision = trace.final_decision
    if decision is None:
        return f"The run failed: {trace.error or 'no decision was returned'}."
    sections = [
        f"status: {decision.status}",
        f"concepts: {', '.join(decision.resolved_concepts) or 'none'}",
        f"claims: {', '.join(decision.claims) or 'none'}",
        f"evidence: {', '.join(decision.evidence_ids) or 'none'}",
        f"explanation: {decision.explanation or 'none'}",
    ]
    if decision.draft_action is not None:
        sections.append(f"action: {decision.draft_action.action_type}")
    return "\n".join(sections)


def trace_to_agent_trace(trace: RunTrace, *, candidate_id: str) -> AgentTrace:
    """Convert one enterprise trace to the Lab's generic trace contract."""

    fallback = utc_now()
    started_at = _parse_time(trace.started_at, fallback)
    ended_at = _parse_time(trace.finished_at, started_at)
    if trace.finished_at is None and trace.latency_ms is not None:
        ended_at = started_at + timedelta(milliseconds=max(0.0, trace.latency_ms))

    usage = trace.usage or {}
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens")) or input_tokens + output_tokens
    tool_calls: list[ObservedToolCall] = []
    cursor = started_at
    for sequence, call in enumerate(trace.tool_calls):
        latency = 0.0
        call_started = cursor
        call_ended = call_started + timedelta(milliseconds=latency)
        tool_calls.append(
            ObservedToolCall(
                call_id=call.call_id,
                sequence=sequence,
                name=call.tool_name,
                arguments=call.arguments,
                outcome=ToolCallOutcome.SUCCESS,
                result_summary=call.result_summary,
                started_at=call_started,
                ended_at=call_ended,
                latency_ms=round(latency),
            )
        )
        cursor = call_ended

    turn = ObservedTurn(
        turn_id=f"turn-{trace.run_id}",
        sequence=0,
        input_text=trace.final_decision.request if trace.final_decision else trace.case_id or "enterprise request",
        output_text=_decision_text(trace),
        tool_calls=tuple(tool_calls),
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=round(trace.latency_ms or 0.0),
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
    )
    decision = trace.final_decision
    metadata = {
        "condition": trace.condition,
        "mode": trace.mode,
        "model_name": trace.model_name,
        "error": trace.error,
        "retrieved_context_ids": list(trace.retrieved_context_ids),
        "status": decision.status if decision else "error",
        "evidence_ids": list(decision.evidence_ids) if decision else [],
        "policy_check_ids": [check.policy_id for check in trace.policy_checks],
        "source_ids": sorted(
            {
                source_id
                for call in trace.tool_calls
                for source_id in call.source_ids
            }
        ),
    }
    return AgentTrace(
        trace_id=f"{trace.run_id}:{trace.condition}",
        case_id=trace.case_id or "unknown",
        candidate_id=candidate_id,
        session_id=f"session-{trace.case_id or 'unknown'}",
        started_at=started_at,
        ended_at=ended_at,
        turns=(turn,),
        metadata=metadata,
    )


@dataclass
class EnterpriseReplayRuntime:
    """Expose deterministic enterprise replay through the Lab runtime protocol."""

    condition: str = "semantic_catalog"
    config: LabConfig | None = None
    name: str = RUNTIME_NAME
    version: str = RUNTIME_VERSION

    async def execute(self, case: EvaluationCaseRef, candidate: AgentCandidate) -> AgentTrace:
        trace = await asyncio.to_thread(
            run_case,
            case.case_id,
            mode="replay",
            condition=self.condition,
            config=self.config or load_config(),
        )
        return trace_to_agent_trace(trace, candidate_id=candidate.candidate_id)


def build_candidate(*, condition: str = "semantic_catalog") -> AgentCandidate:
    """Build the immutable candidate record used by an evaluation run."""

    created_at = utc_now()
    artifact = PromptArtifact(
        artifact_id=f"enterprise-agent-prompt-{condition}",
        name="enterprise-agent-system-prompt",
        version="1.0.0",
        kind=PromptArtifactKind.SYSTEM_PROMPT,
        content=(
            "Resolve business concepts before tools. Use typed arguments and "
            "apply policy before draft actions."
        ),
        created_at=created_at,
        metadata={"condition": condition},
    )
    return AgentCandidate(
        candidate_id=f"enterprise-agent-{condition}",
        name=f"Enterprise agent ({condition})",
        version="1.0.0",
        status=CandidateStatus.DRAFT,
        prompt_artifact_ids=(artifact.artifact_id,),
        rationale="Canonical enterprise agent candidate for deterministic replay.",
        created_at=created_at,
        metadata={
            "condition": condition,
            "artifact_sha256": artifact.content_sha256,
        },
    )


def build_manifest(
    dataset: DatasetVersion,
    candidate: AgentCandidate,
    runtime: EnterpriseReplayRuntime,
    *,
    condition: str,
    model: str = "replay",
) -> RunManifest:
    """Build the reproducibility manifest for one evaluation run."""

    return RunManifest(
        run_id=f"{runtime.name}-{condition}-{dataset.version}",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        candidate_id=candidate.candidate_id,
        prompt_artifact_ids=candidate.prompt_artifact_ids,
        toolset=tuple(
            sorted(
                {
                    tool.name
                    for case in dataset.cases
                    for tool in case.tool_expectations
                }
            )
        ),
        runtime_name=runtime.name,
        runtime_version=runtime.version,
        provider="enterprise_agent_lab.replay",
        model=model,
        created_at=utc_now(),
        metadata={"condition": condition},
    )


def run_replay_evaluation(
    *,
    condition: str = "semantic_catalog",
    case_ids: Sequence[str] | None = None,
    config: LabConfig | None = None,
    repeat: int = 1,
) -> EvaluationRunResult:
    """Run the enterprise replay through Agent Improvement Lab."""

    cases = build_cases()
    if case_ids is not None:
        requested = set(case_ids)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    dataset = build_dataset(cases)
    candidate = build_candidate(condition=condition)
    runtime = EnterpriseReplayRuntime(condition=condition, config=config)
    manifest = build_manifest(dataset, candidate, runtime, condition=condition)
    runner = PydanticEvalsRunner(runtime, max_concurrency=1)
    return runner.run_sync(dataset, candidate, manifest, repeat=repeat)
