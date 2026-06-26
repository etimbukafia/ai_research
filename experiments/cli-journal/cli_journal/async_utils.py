from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")


class AsyncRunner:
    """Run async SDK calls on one persistent loop instead of repeated asyncio.run."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="cli-journal-async", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, awaitable: Awaitable[T]) -> T:
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result()


_RUNNER: AsyncRunner | None = None
_LOCK = threading.Lock()


def run_async(awaitable: Awaitable[T]) -> T:
    global _RUNNER
    with _LOCK:
        if _RUNNER is None:
            _RUNNER = AsyncRunner()
    return _RUNNER.run(awaitable)
