"""Small client-side limiter for Gemini requests."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field


@dataclass
class RequestRateLimiter:
    """Space requests below a project RPM limit.

    The default safety factor leaves room below the configured limit. Gemini
    quotas apply to the project, so callers should use one live worker.
    """

    rpm: float = 15.0
    safety_factor: float = 0.8
    _lock: asyncio.Lock = field(init=False, repr=False)
    _next_allowed: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rpm <= 0:
            raise ValueError("rpm must be greater than zero")
        if not 0 < self.safety_factor <= 1:
            raise ValueError("safety_factor must be in the range (0, 1]")
        self._lock = asyncio.Lock()

    @property
    def interval_seconds(self) -> float:
        return 60.0 / (self.rpm * self.safety_factor)

    @classmethod
    def from_env(cls) -> "RequestRateLimiter":
        return cls(
            rpm=float(os.getenv("GEMINI_RPM", "15")),
            safety_factor=float(os.getenv("GEMINI_RATE_SAFETY", "0.8")),
        )

    async def wait(self) -> None:
        """Wait until the next request is allowed."""

        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_allowed = time.monotonic() + self.interval_seconds
