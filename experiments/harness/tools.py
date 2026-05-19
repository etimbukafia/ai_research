"""Tool registry for agent harness."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BeforeToolCallEvent:
    """Event emitted immediately before a tool is invoked.

    Attributes:
        tool_name: Name of the tool about to be called.
        args:      Positional arguments that will be forwarded to the tool.
        kwargs:    Keyword arguments that will be forwarded to the tool.
        timestamp: UTC time at which the event was created.
    """

    tool_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AfterToolCallEvent:
    """Event emitted immediately after a tool has been invoked.

    Attributes:
        tool_name:   Name of the tool that was called.
        args:        Positional arguments that were forwarded to the tool.
        kwargs:      Keyword arguments that were forwarded to the tool.
        result:      Return value from the tool (``None`` when *error* is set).
        error:       Exception raised by the tool, or ``None`` on success.
        duration_ms: Wall-clock execution time in milliseconds.
        timestamp:   UTC time at which the event was created.
    """

    tool_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    result: Any
    error: Optional[Exception]
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Tool:
    """A registered agent tool.

    Attributes:
        name:        Unique identifier used to look up and invoke the tool.
        description: Human-readable description surfaced to the LLM.
        fn:          The callable that implements the tool's logic.
        permission:  Permission level required to execute (e.g. "read", "write", "admin").
        schema:      Optional JSON schema dict describing the tool's parameters.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    permission: str
    schema: Optional[Dict[str, Any]] = field(default=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the tool directly."""
        return self.fn(*args, **kwargs)


class ToolRegistry:
    """Registry that maps tool names to ``Tool`` instances.

    Supports registration, lookup, execution, and descriptor generation
    for LLM function-calling payloads.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._before_hooks: List[Callable[[BeforeToolCallEvent], None]] = []
        self._after_hooks: List[Callable[[AfterToolCallEvent], None]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        permission: str,
        description: str = "",
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new tool.

        Args:
            name:        Unique tool name.
            fn:          Callable implementing the tool.
            permission:  Permission level required to run this tool.
            description: Optional description shown to the LLM.
            schema:      Optional JSON schema for the tool's parameters.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name!r}")
        self._tools[name] = Tool(
            name=name,
            description=description,
            fn=fn,
            permission=permission,
            schema=schema,
        )

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove.

        Raises:
            KeyError: If no tool with that name exists.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name: {name!r}")
        del self._tools[name]

    # ------------------------------------------------------------------
    # Lookup & execution
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tool]:
        """Return the ``Tool`` for *name*, or ``None`` if not found."""
        return self._tools.get(name)

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Look up and invoke a tool by name, firing lifecycle hooks.

        Before the tool runs every registered *before_tool* hook receives a
        :class:`BeforeToolCallEvent`.  After the tool finishes (or raises)
        every registered *after_tool* hook receives an
        :class:`AfterToolCallEvent` that includes the wall-clock duration and
        any exception.  The exception is then re-raised so callers still see
        it.

        Args:
            name:     Registered tool name.
            *args:    Positional arguments forwarded to the tool.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            Whatever the tool's callable returns.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"No tool registered with name: {name!r}")

        before_event = BeforeToolCallEvent(
            tool_name=name, args=args, kwargs=kwargs
        )
        for hook in self._before_hooks:
            hook(before_event)

        result: Any = None
        error: Optional[Exception] = None
        start = time.perf_counter()
        try:
            result = tool(*args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1_000
            after_event = AfterToolCallEvent(
                tool_name=name,
                args=args,
                kwargs=kwargs,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )
            for hook in self._after_hooks:
                hook(after_event)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def descriptors(self) -> List[Dict[str, Any]]:
        """Return LLM-ready descriptor dicts for all registered tools."""
        result = []
        for tool in self._tools.values():
            desc: Dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "permission": tool.permission,
            }
            if tool.schema is not None:
                desc["parameters"] = tool.schema
            result.append(desc)
        return result

    def names(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = ", ".join(self._tools) or "<empty>"
        return f"ToolRegistry({names})"

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def before_tool(
        self, hook: Callable[[BeforeToolCallEvent], None]
    ) -> None:
        """Register *hook* to be called before every tool invocation.

        Args:
            hook: Callable that receives a :class:`BeforeToolCallEvent`.
                  It should not raise; exceptions will propagate and abort
                  the tool call.
        """
        self._before_hooks.append(hook)

    def after_tool(
        self, hook: Callable[[AfterToolCallEvent], None]
    ) -> None:
        """Register *hook* to be called after every tool invocation.

        The hook fires regardless of whether the tool raised an exception.
        Inspect ``event.error`` to distinguish success from failure.

        Args:
            hook: Callable that receives an :class:`AfterToolCallEvent`.
        """
        self._after_hooks.append(hook)