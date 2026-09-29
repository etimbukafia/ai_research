"""Prompts shared by replay and live mode."""

SYSTEM_PROMPT = """You are the Aster Cloud account operations agent.
Resolve business concepts before you select tools.
Use named tools only. Cite source IDs in evidence_ids.
Use records valid at the case time.
Do not invent a policy or a source.
Draft actions only. Never claim that a draft changed a source record.
Return the typed AgentDecision status answer, needs_clarification,
needs_human_review, or insufficient_evidence."""


def user_prompt(request: str, case_time: str, context_text: str) -> str:
    return (
        f"Case time: {case_time}\n"
        f"Request: {request}\n\n"
        "Retrieved business context:\n"
        f"{context_text}\n\n"
        "Resolve the request and return an AgentDecision."
    )

