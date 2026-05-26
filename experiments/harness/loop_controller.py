"""
Reusable, flexible loop controller for agent harnesses.

Designed to power agents, tool runners, API retry loops, and any
system that needs controlled iteration with observability and safety.

Usage:
    from loop_controller import LoopController, LoopConfig, StopReason

    async def my_step(state):
        # do work, return updated state
        return state

    async def my_stop(state):
        return state.get("done", False)

    config = LoopConfig(max_iterations=10, timeout=30.0)
    controller = LoopController(config=config)
    result = await controller.run(step_fn=my_step, stop_fn=my_stop, initial_state={"done": False})
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Structures
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    """Why the loop terminated."""
    SUCCESS         = "success"          # stop_fn returned True
    MAX_ITERATIONS  = "max_iterations"   # hit iteration cap
    TIMEOUT         = "timeout"          # wall-clock timeout exceeded
    BUDGET_EXCEEDED = "budget_exceeded"  # cost/token budget exceeded
    ERROR           = "error"            # unrecoverable error
    EXTERNAL        = "external"         # stop() called externally


@dataclass
class LoopConfig:
    """
    Configuration for the loop controller.

    Attributes:
        max_iterations:  Hard cap on how many times step_fn can be called.
        timeout:         Wall-clock time limit in seconds (None = unlimited).
        budget:          Abstract cost/token budget (None = unlimited).
                         Each step can consume budget via the cost_fn hook.
        retry_limit:     How many consecutive errors to tolerate before aborting.
        retry_delay:     Seconds to wait between retries (exponential if backoff=True).
        backoff:         Whether to use exponential backoff on retries.
        raise_on_error:  If True, re-raise the final error after retries exhausted.
                         If False, return a result with StopReason.ERROR.
    """
    max_iterations: int             = 100
    timeout: Optional[float]        = None
    budget: Optional[float]         = None
    retry_limit: int                = 3
    retry_delay: float              = 1.0
    backoff: bool                   = True
    raise_on_error: bool            = False


@dataclass
class LoopResult:
    """
    Returned by LoopController.run() after the loop ends.

    Attributes:
        state:          Final state after the last successful step.
        stop_reason:    Why the loop stopped.
        iterations:     Number of step_fn calls that completed successfully.
        elapsed:        Total wall-clock time in seconds.
        budget_used:    Total budget consumed across all steps.
        error:          Last exception caught (if any).
    """
    state: Any
    stop_reason: StopReason
    iterations: int
    elapsed: float
    budget_used: float
    error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Hook Types (for type-checking / documentation)
# ---------------------------------------------------------------------------

# step_fn(state) -> new_state   (sync or async)
StepFn   = Callable[[Any], Any | Awaitable[Any]] # type alias for the function that does the actual work each iteration

# stop_fn(state) -> bool        (sync or async)
StopFn   = Callable[[Any], bool | Awaitable[bool]]

# cost_fn(state) -> float       (sync or async) — how much did this step cost?
CostFn   = Callable[[Any], float | Awaitable[float]]

# Hook signatures: (iteration, state, **kwargs) -> None  (sync or async)
HookFn   = Callable[..., None | Awaitable[None]] # type alias for any lifecycle callback (on_before, on_after, etc.) — receives iteration + state, returns nothing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _maybe_await(value: Any) -> Any:
    """Await a value if it is a coroutine, otherwise return as-is."""
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _call(fn: Callable, *args, **kwargs) -> Any:
    """
    Call fn (sync or async) with args and return the result.

    Utility used everywhere to invoke any function  (step, hook, cost, stop) 
    without caring whether it's sync or async.
    """
    result = fn(*args, **kwargs)
    return await _maybe_await(result)


# ---------------------------------------------------------------------------
# Core Loop Controller
# ---------------------------------------------------------------------------

class LoopController:
    """
    A reusable, flexible loop controller.

    Responsibilities:
    - Drive a step function repeatedly until a stop condition is met.
    - Enforce safety limits: max iterations, timeout, budget.
    - Handle errors with configurable retry + backoff.
    - Fire lifecycle hooks for observability (before/after step, on error, on done).
    - Support external interruption via stop().

    This class is fully async-native but transparently wraps sync functions.

    Parameters:
        config:     LoopConfig controlling safety limits and retry behaviour.
        on_before:  Hook called before each step.  Signature: (iteration, state)
        on_after:   Hook called after each step.   Signature: (iteration, state, cost)
        on_error:   Hook called on step error.     Signature: (iteration, state, error)
        on_done:    Hook called when loop ends.    Signature: (result)
        cost_fn:    Optional function to measure per-step cost from the new state.
    """

    def __init__(
        self,
        config: LoopConfig = LoopConfig(),
        on_before: Optional[HookFn]  = None, # called before each step; e.g. for logging, tracing, or rate limiting
        on_after:  Optional[HookFn]  = None, # called after each successful step; e.g. for metrics, checkpointing, or progress reporting
        on_error:  Optional[HookFn]  = None, # called when a step throws; e.g. for alerting, logging the failure, or custom recovery logic
        on_done:   Optional[HookFn]  = None, # called once when the loop exits for any reason; e.g. for cleanup, final logging, or emitting a summary, or cleanup
        cost_fn:   Optional[CostFn]  = None, # called after each step to measure how much it cost; e.g. for budget tracking or cost estimation. feeds the budget safety limit.
    ):
        self.config     = config
        self._on_before = on_before
        self._on_after  = on_after
        self._on_error  = on_error
        self._on_done   = on_done
        self._cost_fn   = cost_fn

        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Request the loop to stop after the current step completes.
        Safe to call from another thread or coroutine.
        """
        self._stop_requested = True

    async def run(
        self,
        step_fn:       StepFn,
        stop_fn:       Optional[StopFn] = None,
        initial_state: Any              = None,
    ) -> LoopResult:
        """
        Run the loop.

        Args:
            step_fn:        Called each iteration.  Receives state, returns new state.
            stop_fn:        Optional predicate.  Loop stops when it returns True.
            initial_state:  Seed value passed to the first step_fn call.

        Returns:
            LoopResult with final state, stop reason, and diagnostics.
        """
        self._stop_requested = False

        state        = initial_state
        iterations   = 0
        budget_used  = 0.0
        start_time   = time.monotonic()
        last_error: Optional[Exception] = None

        while True:
            # ── Safety: max iterations ──────────────────────────────────
            if iterations >= self.config.max_iterations:
                logger.warning("Loop hit max_iterations=%d", self.config.max_iterations)
                return await self._finish(
                    state, StopReason.MAX_ITERATIONS,
                    iterations, start_time, budget_used
                )

            # ── Safety: timeout ─────────────────────────────────────────
            elapsed = time.monotonic() - start_time
            if self.config.timeout is not None and elapsed >= self.config.timeout:
                logger.warning("Loop timed out after %.2fs", elapsed)
                return await self._finish(
                    state, StopReason.TIMEOUT,
                    iterations, start_time, budget_used
                )

            # ── Safety: budget ──────────────────────────────────────────
            if self.config.budget is not None and budget_used >= self.config.budget:
                logger.warning("Loop exceeded budget (used=%.4f)", budget_used)
                return await self._finish(
                    state, StopReason.BUDGET_EXCEEDED,
                    iterations, start_time, budget_used
                )

            # ── External stop ───────────────────────────────────────────
            if self._stop_requested:
                logger.info("Loop stopped externally at iteration %d", iterations)
                return await self._finish(
                    state, StopReason.EXTERNAL,
                    iterations, start_time, budget_used
                )

            # ── Before hook ─────────────────────────────────────────────
            if self._on_before:
                await _call(self._on_before, iterations, state)

            # ── Step execution (with retry) ─────────────────────────────
            new_state, step_error = await self._run_step(step_fn, state, iterations)

            if step_error is not None:
                last_error = step_error
                if self._on_error:
                    await _call(self._on_error, iterations, state, step_error)
                if self.config.raise_on_error:
                    raise step_error
                return await self._finish(
                    state, StopReason.ERROR,
                    iterations, start_time, budget_used, error=step_error
                )

            state = new_state
            iterations += 1

            # ── Cost accounting ─────────────────────────────────────────
            step_cost = 0.0
            if self._cost_fn:
                step_cost = float(await _call(self._cost_fn, state))
                budget_used += step_cost

            # ── After hook ──────────────────────────────────────────────
            if self._on_after:
                await _call(self._on_after, iterations, state, step_cost)

            # ── Stop condition ──────────────────────────────────────────
            if stop_fn and await _call(stop_fn, state):
                logger.info("stop_fn triggered at iteration %d", iterations)
                return await self._finish(
                    state, StopReason.SUCCESS,
                    iterations, start_time, budget_used
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        step_fn:   StepFn,
        state:     Any,
        iteration: int,
    ) -> tuple[Any, Optional[Exception]]:
        """
        Execute step_fn with retry + backoff.

        Returns (new_state, None) on success, (state, error) on failure.
        """
        delay = self.config.retry_delay

        for attempt in range(self.config.retry_limit + 1):
            try:
                new_state = await _call(step_fn, state)
                if attempt > 0:
                    logger.info(
                        "Step recovered after %d attempt(s) at iteration %d",
                        attempt, iteration
                    )
                return new_state, None

            except Exception as exc:
                is_last = attempt >= self.config.retry_limit
                logger.warning(
                    "Step error at iteration %d, attempt %d/%d: %s",
                    iteration, attempt + 1, self.config.retry_limit + 1, exc
                )
                if is_last:
                    return state, exc

                await asyncio.sleep(delay)
                if self.config.backoff:
                    delay *= 2

        # Should not reach here, but satisfy type checker
        return state, RuntimeError("Unexpected retry loop exit")

    async def _finish(
        self,
        state:       Any,
        reason:      StopReason,
        iterations:  int,
        start_time:  float,
        budget_used: float,
        error:       Optional[Exception] = None,
    ) -> LoopResult:
        """
        Cleanup helper called at every exit point of the loop to ensure
        we always return a consistent LoopResult.   
        """
        elapsed = time.monotonic() - start_time
        result  = LoopResult(
            state       = state,
            stop_reason = reason,
            iterations  = iterations,
            elapsed     = elapsed,
            budget_used = budget_used,
            error       = error,
        )
        if self._on_done:
            await _call(self._on_done, result)
        return result