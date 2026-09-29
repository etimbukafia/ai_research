"""Offline replay adapter with deterministic recorded decisions."""

from __future__ import annotations

from typing import Any

from .models import AgentDecision, EvaluationCase, PolicyCheck
from .tools import ToolRegistry


def _arguments(case: EvaluationCase, tool_name: str) -> dict[str, Any]:
    if tool_name in case.expected_tool_arguments:
        return dict(case.expected_tool_arguments[tool_name])
    account_id = case.account_id
    if tool_name == "find_account":
        return {"account_id": account_id}
    if tool_name == "get_active_contract":
        return {"account_id": account_id, "at": case.case_time}
    if tool_name == "get_subscription":
        return {"account_id": account_id, "at": case.case_time}
    if tool_name == "get_invoice":
        return {
            "account_id": account_id,
            "period_start": "2026-05-01",
            "period_end": "2026-05-31",
        }
    if tool_name == "get_usage_record":
        return {
            "account_id": account_id,
            "period_start": "2026-05-01",
            "period_end": "2026-05-31",
        }
    if tool_name == "get_support_tickets":
        return {"account_id": account_id, "status": "open"}
    if tool_name == "search_business_concepts":
        return {"query": case.request, "account_scope": account_id, "top_k": 8}
    if tool_name == "check_policy":
        return {
            "action": case.expected_action or "service_credit",
            "account_id": account_id,
            "amount": 750.0 if case.expected_action == "service_credit" else None,
            "evidence_ids": case.required_evidence_ids,
            "at": case.case_time,
            "reason": case.request,
        }
    if tool_name == "draft_service_credit":
        return {
            "account_id": account_id,
            "amount": 750.0,
            "reason": case.request,
            "evidence_ids": [item for item in case.required_evidence_ids if not item.startswith("policy-")],
            "at": case.case_time,
        }
    if tool_name == "draft_plan_change":
        return {
            "account_id": account_id,
            "target_plan": "Scale",
            "target_seats": 200,
            "reason": case.request,
            "evidence_ids": case.required_evidence_ids,
            "at": case.case_time,
        }
    if tool_name == "draft_support_ticket":
        return {
            "account_id": account_id,
            "subject": "Aster Cloud account request",
            "reason": case.request,
            "priority": "high" if case.approval_required else "normal",
            "evidence_ids": case.required_evidence_ids,
        }
    return {}


def replay_case(
    case: EvaluationCase,
    condition: str,
    registry: ToolRegistry,
) -> AgentDecision:
    """Replay the recorded response shape for one condition."""

    if condition not in {"raw_schema", "prose_rag", "semantic_catalog"}:
        raise ValueError(f"Unknown replay condition: {condition}")
    if condition == "semantic_catalog":
        resolved = list(case.required_concepts)
        evidence = list(case.required_evidence_ids)
        selected = list(case.required_tools)
    elif condition == "prose_rag":
        resolved = list(case.required_concepts[: max(1, len(case.required_concepts) // 2)])
        evidence = list(case.required_evidence_ids[: max(1, len(case.required_evidence_ids) // 2)])
        selected = [tool for tool in case.required_tools if tool.startswith("get_")][:2]
    else:
        resolved = []
        evidence = list(case.required_evidence_ids[:1])
        selected = []

    draft_action = None
    policy_checks: list[PolicyCheck] = []
    for tool_name in selected:
        if tool_name == "request_human_approval":
            continue
        arguments = _arguments(case, tool_name)
        if not arguments.get("account_id") and tool_name != "search_business_concepts":
            continue
        try:
            result = registry.call(tool_name, arguments)
            if hasattr(result, "action"):
                draft_action = result.action
            if hasattr(result, "checks"):
                policy_checks = list(result.checks)
        except Exception:
            # The recorded model response remains available even if its replay
            # call would fail a current policy. The trace records successful calls.
            continue

    # Policy checks are read from the policy engine for the final decision.
    if case.expected_action and condition == "semantic_catalog":
        checks, _ = registry.policy_engine.check_action(
            case.expected_action,
            case.account_id or "",
            amount=750.0 if case.expected_action == "service_credit" else None,
            evidence=case.required_evidence_ids,
            at=case.case_time,
            reason=case.request,
            priority="high" if case.approval_required else "normal",
        )
        policy_checks = checks
    status = case.expected_status
    if condition != "semantic_catalog":
        status = "insufficient_evidence" if condition == "raw_schema" else "needs_clarification"
    return AgentDecision(
        status=status,
        request=case.request,
        resolved_concepts=resolved,
        selected_tools=selected,
        typed_tool_arguments={
            name: _arguments(case, name) for name in selected if _arguments(case, name)
        },
        claims=case.expected_claims if condition == "semantic_catalog" else [],
        evidence_ids=evidence,
        policy_checks=policy_checks,
        draft_action=draft_action,
        explanation=(
            "Replay used the typed semantic catalog."
            if condition == "semantic_catalog"
            else f"Replay used the {condition} context and retained incomplete evidence."
        ),
    )
