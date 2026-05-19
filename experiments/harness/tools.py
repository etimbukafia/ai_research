"""Tool registry for agent harness."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from experiments.harness.hooks import HookAction, HookContext, HookPhase, HookRegistry


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

    def __init__(self, hook_registry: Optional[HookRegistry] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        self.hook_registry = hook_registry or HookRegistry()

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

    async def execute(
        self,
        name: str,
        run_id: UUID,
        step: int,
        agent_name: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Look up and invoke a tool by name, firing lifecycle hooks.

        Before the tool runs, PRE_TOOL hooks are fired.
        After the tool finishes (or raises), POST_TOOL hooks are fired.

        Args:
            name:       Registered tool name.
            run_id:     ID of the current agent run.
            step:       Current reasoning step index.
            agent_name: Name of the agent.
            *args:      Positional arguments forwarded to the tool.
            **kwargs:   Keyword arguments forwarded to the tool.

        Returns:
            Whatever the tool's callable returns, or a mocked replacement
            if a hook returned HookAction.SKIP.

        Raises:
            KeyError:     If no tool with that name is registered.
            RuntimeError: If a hook aborts the execution.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"No tool registered with name: {name!r}")

        pre_ctx = HookContext(
            run_id=run_id,
            step=step,
            agent_name=agent_name,
            phase=HookPhase.PRE_TOOL,
            tool_name=name,
            payload={"args": args, "kwargs": kwargs}
        )
        
        pre_decision = await self.hook_registry.dispatch_and_merge(pre_ctx)
        
        if pre_decision.action == HookAction.ABORT:
            raise RuntimeError(f"Tool execution aborted: {pre_decision.reason}")
        elif pre_decision.action == HookAction.SKIP:
            return pre_decision.replacement

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
            post_ctx = HookContext(
                run_id=run_id,
                step=step,
                agent_name=agent_name,
                phase=HookPhase.POST_TOOL,
                tool_name=name,
                payload={
                    "result": result,
                    "args": args,
                    "kwargs": kwargs,
                    "duration_ms": duration_ms
                },
                error=error
            )
            await self.hook_registry.dispatch(post_ctx)

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
