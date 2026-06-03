from __future__ import annotations


def format_recent_conversation(
    context_history: list[dict],
    assistant_output: str,
    *,
    max_chars: int = 24_000,
) -> str:
    lines: list[str] = []
    for msg in context_history:
        role = str(msg.get("role", "message")).strip() or "message"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    if assistant_output.strip():
        lines.append(f"assistant: {assistant_output.strip()}")
    text = "\n".join(lines).strip()
    if not text:
        return "No recent conversation available."
    if len(text) <= max_chars:
        return text
    return "[Earlier conversation omitted due to length]\n" + text[-max_chars:]
