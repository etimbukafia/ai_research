from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


TurnState = Literal[
    "direct_response",
    "plan",
    "resume_continuation",
    "cancel_continuation",
    "memory_review",
    "ddc_review_action",
]

TurnIntentName = Literal[
    "direct_chat",
    "task",
    "continuation_answer",
    "cancel_continuation",
    "memory_review",
    "ddc_review_action",
]

IntentConfidence = Literal["low", "medium", "high"]


class TurnIntent(BaseModel):
    intent: TurnIntentName
    confidence: IntentConfidence = "medium"
    needs_planning: bool = False
    needs_verification: bool = False
    needs_memory_synthesis: bool = True
    needs_ddc_analysis: bool = True
    reason: str


class TurnRoute(BaseModel):
    state: TurnState
    run_planner: bool = False
    run_verifier: bool = False
    run_memory_synthesis: bool = True
    run_ddc_analysis: bool = True
    reason: str


class VerificationResult(BaseModel):
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    revised_response: Optional[str] = None


class AssistantFSM:
    """Small deterministic router for the internal multi-agent pipeline."""

    _DIRECT_MESSAGES = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "got it",
    }
    _CANCEL_MESSAGES = {"cancel", "cancel this", "stop"}
    def deterministic_route(self, user_input: str, *, has_pending_continuation: bool) -> Optional[TurnRoute]:
        normalized = " ".join(user_input.strip().lower().split())

        if has_pending_continuation:
            if normalized in self._CANCEL_MESSAGES:
                return TurnRoute(
                    state="cancel_continuation",
                    run_memory_synthesis=False,
                    run_ddc_analysis=False,
                    reason="User cancelled an outstanding blocked task.",
                )
            return TurnRoute(
                state="resume_continuation",
                run_verifier=True,
                reason="User answered a blocking planner question.",
            )

        if not normalized or normalized in self._DIRECT_MESSAGES:
            return TurnRoute(
                state="direct_response",
                run_memory_synthesis=False,
                run_ddc_analysis=False,
                reason="Simple conversational message does not need planning or verification.",
            )

        return None

    def route(self, user_input: str, *, has_pending_continuation: bool) -> TurnRoute:
        deterministic = self.deterministic_route(user_input, has_pending_continuation=has_pending_continuation)
        if deterministic is not None:
            return deterministic
        return self.route_from_intent(
            TurnIntent(
                intent="task",
                confidence="low",
                needs_planning=True,
                needs_verification=True,
                reason="No classifier supplied; low confidence defaults to planning.",
            )
        )

    def route_from_intent(self, intent: TurnIntent) -> TurnRoute:
        if intent.confidence == "low":
            return TurnRoute(
                state="plan",
                run_planner=True,
                run_verifier=True,
                reason=f"Low-confidence intent defaults to planner: {intent.reason}",
            )

        if intent.intent == "direct_chat":
            return TurnRoute(
                state="direct_response",
                run_memory_synthesis=intent.needs_memory_synthesis,
                run_ddc_analysis=intent.needs_ddc_analysis,
                reason=intent.reason,
            )

        if intent.intent == "task":
            return TurnRoute(
                state="plan",
                run_planner=True,
                run_verifier=True,
                reason=intent.reason,
            )

        if intent.intent == "continuation_answer":
            return TurnRoute(
                state="resume_continuation",
                run_verifier=True,
                reason=intent.reason,
            )

        if intent.intent == "cancel_continuation":
            return TurnRoute(
                state="cancel_continuation",
                run_memory_synthesis=False,
                run_ddc_analysis=False,
                reason=intent.reason,
            )

        if intent.intent == "memory_review":
            return TurnRoute(
                state="memory_review",
                run_memory_synthesis=False,
                run_ddc_analysis=False,
                reason=intent.reason,
            )

        if intent.intent == "ddc_review_action":
            return TurnRoute(
                state="ddc_review_action",
                run_memory_synthesis=False,
                run_ddc_analysis=False,
                reason=intent.reason,
            )

        return TurnRoute(
            state="direct_response",
            run_memory_synthesis=intent.needs_memory_synthesis,
            run_ddc_analysis=intent.needs_ddc_analysis,
            reason=intent.reason,
        )
