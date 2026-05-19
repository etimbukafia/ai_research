"""
hooks.py — Reusable, flexible hook system for the agent harness.

Provides:
  - HookContext   : immutable snapshot of agent state at a hook point
  - HookDecision  : mutable decision object a hook returns to the harness
  - HookPhase     : enum of well-known hook phases
  - HookPriority  : priority constants for ordering hooks
  - Hook          : base class / protocol for individual hooks
  - HookRegistry  : manages registration and ordered dispatch
  - built-in hooks: LoggingHook, RateLimitHook, TimeoutHook, FilterHook
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum, auto
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HookPhase(str, Enum):
    """Well-known lifecycle phases where hooks may fire."""

    PRE_RUN = "pre_run"           # before the agent loop starts
    PRE_STEP = "pre_step"         # before each reasoning step
    POST_STEP = "post_step"       # after each reasoning step
    PRE_TOOL = "pre_tool"         # before a tool call is executed
    POST_TOOL = "post_tool"       # after a tool call returns
    PRE_LLM = "pre_llm"           # before the LLM is called
    POST_LLM = "post_llm"         # after the LLM responds
    ON_ERROR = "on_error"         # when an unhandled exception occurs
    POST_RUN = "post_run"         # after the agent loop finishes
    CUSTOM = "custom"             # user-defined phases


class HookAction(str, Enum):
    """What the harness should do after processing all hooks at a phase."""

    CONTINUE = "continue"         # proceed normally
    SKIP = "skip"                 # skip the current operation (return early)
    ABORT = "abort"               # abort the entire run with an error
    RETRY = "retry"               # retry the current operation
    REPLACE = "replace"           # replace the payload with HookDecision.replacement


# ---------------------------------------------------------------------------
# HookContext
# ---------------------------------------------------------------------------


class HookContext(BaseModel, Generic[InputT]):
    """
    Immutable snapshot of agent state delivered to each hook.

    The model is frozen so hooks cannot accidentally mutate shared state.
    To modify execution, hooks return a HookDecision instead.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # --- Identity & tracing ---
    hook_id: UUID = Field(default_factory=uuid4, description="Unique ID for this hook invocation")
    run_id: UUID = Field(description="ID of the current agent run")
    step: int = Field(default=0, ge=0, description="Current reasoning step index (0-based)")
    phase: HookPhase = Field(description="Lifecycle phase that triggered this hook")

    # --- Timing ---
    timestamp: float = Field(
        default_factory=time.monotonic,
        description="Monotonic timestamp when the context was created",
    )

    # --- Payload ---
    payload: InputT | None = Field(
        default=None,
        description="Phase-specific data: prompt, tool call, LLM response, error, etc.",
    )

    # --- Agent state snapshot ---
    agent_name: str = Field(description="Name / identifier of the agent")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached by the harness",
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool being called (set during PRE_TOOL / POST_TOOL)",
    )
    error: BaseException | None = Field(
        default=None,
        description="Exception that triggered ON_ERROR phase",
    )
    custom_phase_name: str | None = Field(
        default=None,
        description="Caller-defined label when phase == HookPhase.CUSTOM",
    )

    # --- Convenience helpers (not persisted) ---

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this context was created."""
        return (time.monotonic() - self.timestamp) * 1_000

    def with_metadata(self, **kwargs: Any) -> "HookContext[InputT]":
        """Return a new context with extra metadata merged in."""
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})

    @model_validator(mode="after")
    def _validate_custom_phase(self) -> "HookContext[InputT]":
        if self.phase == HookPhase.CUSTOM and not self.custom_phase_name:
            raise ValueError("custom_phase_name is required when phase == HookPhase.CUSTOM")
        return self


# ---------------------------------------------------------------------------
# HookDecision
# ---------------------------------------------------------------------------


class HookDecision(BaseModel, Generic[OutputT]):
    """
    Mutable decision returned by a hook to instruct the harness.

    Multiple HookDecisions from a chain of hooks are merged in priority order;
    the harness calls HookDecision.merge() to produce a single resolved decision.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: HookAction = Field(
        default=HookAction.CONTINUE,
        description="Primary instruction to the harness",
    )
    replacement: OutputT | None = Field(
        default=None,
        description="Replacement payload used when action == HookAction.REPLACE",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation for this decision",
    )
    retry_delay_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds to wait before retrying (action == HookAction.RETRY)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts before the harness gives up",
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary data hooks may attach for downstream consumers",
    )
    hook_name: str = Field(
        default="",
        description="Name of the hook that produced this decision",
    )

    @field_validator("replacement", mode="before")
    @classmethod
    def _replacement_requires_replace_action(cls, v: Any, info: Any) -> Any:
        # Pydantic v2: info.data may not yet have 'action' during field validation,
        # so this is a soft check; the model_validator below does the authoritative check.
        return v

    @model_validator(mode="after")
    def _check_replace_has_payload(self) -> "HookDecision[OutputT]":
        if self.action == HookAction.REPLACE and self.replacement is None:
            raise ValueError("replacement must be set when action == HookAction.REPLACE")
        return self

    # --- Factory helpers ---

    @classmethod
    def continue_(cls, *, reason: str = "", **annotations: Any) -> "HookDecision[Any]":
        return cls(action=HookAction.CONTINUE, reason=reason, annotations=annotations)

    @classmethod
    def skip(cls, *, reason: str = "", **annotations: Any) -> "HookDecision[Any]":
        return cls(action=HookAction.SKIP, reason=reason, annotations=annotations)

    @classmethod
    def abort(cls, *, reason: str = "", **annotations: Any) -> "HookDecision[Any]":
        return cls(action=HookAction.ABORT, reason=reason, annotations=annotations)

    @classmethod
    def retry(
        cls,
        *,
        delay_s: float = 1.0,
        max_retries: int = 3,
        reason: str = "",
        **annotations: Any,
    ) -> "HookDecision[Any]":
        return cls(
            action=HookAction.RETRY,
            retry_delay_s=delay_s,
            max_retries=max_retries,
            reason=reason,
            annotations=annotations,
        )

    @classmethod
    def replace(
        cls,
        payload: OutputT,
        *,
        reason: str = "",
        **annotations: Any,
    ) -> "HookDecision[OutputT]":
        return cls(
            action=HookAction.REPLACE,
            replacement=payload,
            reason=reason,
            annotations=annotations,
        )

    # --- Merge logic ---

    @staticmethod
    def merge(decisions: list["HookDecision[Any]"]) -> "HookDecision[Any]":
        """
        Fold multiple decisions into one.

        Priority (highest first):
          ABORT > RETRY > SKIP > REPLACE > CONTINUE

        The first (highest-priority) non-CONTINUE decision wins for `action`.
        Annotations from all decisions are merged (later keys override earlier ones).
        """
        if not decisions:
            return HookDecision.continue_()

        priority: dict[HookAction, int] = {
            HookAction.ABORT: 5,
            HookAction.RETRY: 4,
            HookAction.SKIP: 3,
            HookAction.REPLACE: 2,
            HookAction.CONTINUE: 1,
        }

        winning = max(decisions, key=lambda d: priority[d.action])
        merged_annotations: dict[str, Any] = {}
        for d in decisions:
            merged_annotations.update(d.annotations)

        return winning.model_copy(update={"annotations": merged_annotations})


# ---------------------------------------------------------------------------
# Hook base class / protocol
# ---------------------------------------------------------------------------


class Hook(BaseModel, Generic[InputT, OutputT]):
    """
    Abstract base for all hooks.

    Subclass and override `__call__` (sync or async).
    The harness always awaits hooks, wrapping sync ones automatically.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Unique hook name used in logs and decisions")
    phases: set[HookPhase] = Field(
        default_factory=lambda: set(HookPhase),
        description="Phases this hook is active for (defaults to all)",
    )
    priority: int = Field(
        default=0,
        description="Execution order within a phase; higher priority runs first",
    )
    enabled: bool = Field(default=True, description="Toggle without removing from registry")

    def applies_to(self, phase: HookPhase) -> bool:
        return self.enabled and phase in self.phases

    async def __call__(self, ctx: HookContext[InputT]) -> HookDecision[OutputT]:
        raise NotImplementedError  # pragma: no cover

    # Allow sync subclasses to override `run` instead
    def run(self, ctx: HookContext[InputT]) -> HookDecision[OutputT]:
        raise NotImplementedError  # pragma: no cover


# Convenience type alias for plain callables used as hooks
HookCallable = Callable[[HookContext[Any]], Awaitable[HookDecision[Any]] | HookDecision[Any]]


class FunctionHook(Hook[Any, Any]):
    """Wraps a plain function (sync or async) as a Hook."""

    fn: HookCallable = Field(description="Callable to invoke")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        result = self.fn(ctx)
        if asyncio.iscoroutine(result):
            return await result
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


class HookRegistry:
    """
    Manages hook registration and ordered dispatch.

    Usage::

        registry = HookRegistry()
        registry.register(my_hook)
        decisions = await registry.dispatch(ctx)
        decision  = HookDecision.merge(decisions)
    """

    def __init__(self) -> None:
        self._hooks: list[Hook[Any, Any]] = []

    # --- Registration ---

    def register(self, hook: Hook[Any, Any]) -> None:
        """Add a hook. Hooks with the same name replace the previous entry."""
        self._hooks = [h for h in self._hooks if h.name != hook.name]
        self._hooks.append(hook)
        self._hooks.sort(key=lambda h: -h.priority)  # descending: higher = first

    def register_fn(
        self,
        fn: HookCallable,
        *,
        name: str | None = None,
        phases: set[HookPhase] | None = None,
        priority: int = 0,
    ) -> FunctionHook:
        """Convenience wrapper: register a plain function as a hook."""
        hook = FunctionHook(
            name=name or fn.__name__,
            phases=phases or set(HookPhase),
            priority=priority,
            fn=fn,
        )
        self.register(hook)
        return hook

    def unregister(self, name: str) -> None:
        self._hooks = [h for h in self._hooks if h.name != name]

    def get(self, name: str) -> Hook[Any, Any] | None:
        return next((h for h in self._hooks if h.name == name), None)

    @property
    def hooks(self) -> list[Hook[Any, Any]]:
        return list(self._hooks)

    # --- Dispatch ---

    async def dispatch(
        self,
        ctx: HookContext[Any],
        *,
        fail_fast: bool = False,
    ) -> list[HookDecision[Any]]:
        """
        Call every hook that applies to `ctx.phase`, in priority order.

        Parameters
        ----------
        ctx:
            The context to deliver to each hook.
        fail_fast:
            If True, stop dispatching after the first non-CONTINUE decision.

        Returns
        -------
        List of decisions in dispatch order.  Pass to `HookDecision.merge()`.
        """
        decisions: list[HookDecision[Any]] = []
        for hook in self._hooks:
            if not hook.applies_to(ctx.phase):
                continue
            try:
                decision = await hook(ctx)
            except Exception as exc:
                logger.exception("Hook %r raised an exception: %s", hook.name, exc)
                decision = HookDecision.abort(reason=f"Hook {hook.name!r} raised: {exc}")

            decision = decision.model_copy(update={"hook_name": hook.name})
            decisions.append(decision)

            if fail_fast and decision.action != HookAction.CONTINUE:
                logger.debug("fail_fast: stopping dispatch after %r returned %s", hook.name, decision.action)
                break

        return decisions

    async def dispatch_and_merge(
        self,
        ctx: HookContext[Any],
        *,
        fail_fast: bool = False,
    ) -> HookDecision[Any]:
        """Dispatch and return a single merged decision."""
        decisions = await self.dispatch(ctx, fail_fast=fail_fast)
        return HookDecision.merge(decisions)


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


class LoggingHook(Hook[Any, Any]):
    """Logs every hook context at DEBUG level."""

    name: str = "logging"
    phases: set[HookPhase] = Field(default_factory=lambda: set(HookPhase))
    priority: int = -100  # run last so it sees final state

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        logger.debug(
            "[%s] run=%s step=%d phase=%s tool=%s payload_type=%s",
            ctx.agent_name,
            ctx.run_id,
            ctx.step,
            ctx.phase.value,
            ctx.tool_name or "-",
            type(ctx.payload).__name__ if ctx.payload is not None else "None",
        )
        return HookDecision.continue_()


class RateLimitHook(Hook[Any, Any]):
    """
    Token-bucket rate limiter.  Aborts or retries when the bucket is empty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "rate_limit"
    phases: set[HookPhase] = Field(
        default_factory=lambda: {HookPhase.PRE_LLM, HookPhase.PRE_TOOL}
    )
    priority: int = 50

    max_tokens: float = Field(default=10.0, gt=0)
    refill_rate: float = Field(default=1.0, gt=0, description="Tokens added per second")
    retry_on_empty: bool = Field(default=True, description="RETRY instead of ABORT when bucket is empty")

    _tokens: float = 0.0
    _last_refill: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        self._tokens = self.max_tokens
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return HookDecision.continue_()

        wait = (1.0 - self._tokens) / self.refill_rate
        reason = f"Rate limit exceeded; bucket empty (refill in ~{wait:.1f}s)"
        if self.retry_on_empty:
            return HookDecision.retry(delay_s=wait, reason=reason)
        return HookDecision.abort(reason=reason)


class TimeoutHook(Hook[Any, Any]):
    """Aborts if a step has been running longer than `max_duration_s`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "timeout"
    phases: set[HookPhase] = Field(
        default_factory=lambda: {HookPhase.PRE_STEP, HookPhase.PRE_TOOL, HookPhase.PRE_LLM}
    )
    priority: int = 80

    max_duration_s: float = Field(default=30.0, gt=0)
    _run_start: dict[UUID, float] = {}

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        if ctx.phase == HookPhase.PRE_STEP and ctx.step == 0:
            self._run_start[ctx.run_id] = time.monotonic()

        start = self._run_start.get(ctx.run_id)
        if start is not None:
            elapsed = time.monotonic() - start
            if elapsed > self.max_duration_s:
                return HookDecision.abort(
                    reason=f"Run exceeded timeout of {self.max_duration_s}s (elapsed {elapsed:.1f}s)"
                )
        return HookDecision.continue_()


class FilterHook(Hook[Any, Any]):
    """
    General-purpose filter hook.

    Calls a user-supplied predicate; if it returns False the hook
    returns the configured `action_on_reject` (default: SKIP).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "filter"
    priority: int = 60
    predicate: Callable[[HookContext[Any]], bool] = Field(
        description="Return True to allow, False to reject"
    )
    action_on_reject: HookAction = Field(default=HookAction.SKIP)
    reject_reason: str = Field(default="Filtered by FilterHook")

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        allowed = self.predicate(ctx)
        if allowed:
            return HookDecision.continue_()
        return HookDecision(action=self.action_on_reject, reason=self.reject_reason)


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------


def hook(
    *phases: HookPhase,
    name: str | None = None,
    priority: int = 0,
    registry: HookRegistry | None = None,
) -> Callable[[HookCallable], FunctionHook]:
    """
    Decorator that converts a function into a FunctionHook and optionally
    registers it in a registry.

    Example::

        @hook(HookPhase.PRE_TOOL, priority=10, registry=my_registry)
        async def block_shell(ctx: HookContext) -> HookDecision:
            if ctx.tool_name == "shell":
                return HookDecision.abort(reason="shell tool is disabled")
            return HookDecision.continue_()
    """

    def decorator(fn: HookCallable) -> FunctionHook:
        fn_hook = FunctionHook(
            name=name or fn.__name__,
            phases=set(phases) if phases else set(HookPhase),
            priority=priority,
            fn=fn,
        )
        if registry is not None:
            registry.register(fn_hook)
        return fn_hook

    return decorator