"""Replay and live case runner."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .agent import AgentDependencies, create_live_agent
from .cases import get_case
from .config import LabConfig, load_config
from .dependencies import build_components
from .models import AgentDecision, RunTrace
from .prompts import user_prompt
from .replay import replay_case
from .trace import TraceWriter, prompt_hash, utc_now


def _usage(result: Any) -> dict[str, Any]:
    usage_method = getattr(result, "usage", None)
    if not callable(usage_method):
        return {}
    try:
        usage = usage_method()
    except Exception:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return {"value": str(usage)}


def run_case(
    case_id: str,
    *,
    mode: str = "replay",
    condition: str = "semantic_catalog",
    config: LabConfig | None = None,
) -> RunTrace:
    """Run one case and persist one complete local trace."""

    if mode not in {"replay", "live"}:
        raise ValueError("mode must be replay or live")
    if condition not in {"raw_schema", "prose_rag", "semantic_catalog"}:
        raise ValueError(f"Unknown condition: {condition}")
    case = get_case(case_id)
    config = config or load_config()
    if mode == "live":
        # Check before SQLite setup, review files, or tool execution.
        config.require_live()
    components = build_components(config, condition, case.account_id, case.case_time)
    start = time.perf_counter()
    started_at = utc_now()
    contexts = components.retriever.search_business_context(
        case.request, account_scope=case.account_id, top_k=8
    )
    context_text = "\n".join(context.text for context in contexts)
    prompt = user_prompt(case.request, case.case_time, context_text)
    run_id = (
        f"replay-{case_id}-{condition}"
        if mode == "replay"
        else f"live-{case_id}-{uuid.uuid4().hex[:12]}"
    )
    result_usage: dict[str, Any] = {}
    try:
        if mode == "replay":
            decision = replay_case(case, condition, components.registry)
        else:
            agent = create_live_agent(config)
            result = asyncio.run(
                agent.run(
                    prompt,
                    deps=AgentDependencies(
                        registry=components.registry,
                        retriever=components.retriever,
                        account_scope=case.account_id,
                        case_time=case.case_time,
                    ),
                )
            )
            decision = result.output
            result_usage = _usage(result)
        if not isinstance(decision, AgentDecision):
            decision = AgentDecision.model_validate(decision)
        error = None
    except Exception as exc:
        error = str(exc)
        decision = None
        result_usage = {"errors": 1}
    finished_at = utc_now()
    trace = RunTrace(
        run_id=run_id,
        case_id=case_id,
        condition=condition,
        mode=mode,
        model_name=config.model_name,
        prompt_hash=prompt_hash(prompt),
        retrieved_context_ids=[context.context_id for context in contexts],
        tool_calls=TraceWriter.tool_records(components.registry.call_records),
        policy_checks=decision.policy_checks if decision else [],
        final_decision=decision,
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        usage={
            **result_usage,
            "tool_calls": len(components.registry.call_records),
            "retrieved_records": len(contexts),
            "policy_checks": len(decision.policy_checks) if decision else 0,
        },
        error=error,
    )
    TraceWriter(config.run_dir).write(trace)
    components.store.close()
    if error:
        raise RuntimeError(error)
    return trace
