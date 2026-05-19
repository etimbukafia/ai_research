"""Sub-agent specification, base interface, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Agent Specification
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentSpec:
    """Metadata contract for every sub-agent."""

    name: str
    description: str
    capabilities: list[str]

    tags: set[str] = field(default_factory=set)
    priority: int = 1
    max_retries: int = 2
    timeout_seconds: int = 60
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Sub-Agent Interface
# ---------------------------------------------------------------------------

class BaseSubAgent(ABC):
    """Base interface all sub-agents must implement."""

    @property
    @abstractmethod
    def spec(self) -> AgentSpec:
        """Return this agent's specification."""
        ...

    @abstractmethod
    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the agent task and return a result dict."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SubAgentRegistry:
    """Registry for storing and discovering sub-agents by name, capability, or tag."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseSubAgent] = {}
        self._capability_index: dict[str, list[BaseSubAgent]] = defaultdict(list)
        self._tag_index: dict[str, list[BaseSubAgent]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseSubAgent, overwrite: bool = False) -> None:
        """Register *agent*.

        Raises:
            ValueError: If the name is already taken and *overwrite* is False.
        """
        name = agent.spec.name
        if name in self._agents:
            if not overwrite:
                raise ValueError(f"Agent already registered: {name!r}")
            self.unregister(name)

        self._agents[name] = agent
        for cap in agent.spec.capabilities:
            self._capability_index[cap].append(agent)
        for tag in agent.spec.tags:
            self._tag_index[tag].append(agent)

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry.

        Raises:
            KeyError: If no agent with that name is registered.
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered: {name!r}")

        agent = self._agents.pop(name)

        for cap in agent.spec.capabilities:
            self._capability_index[cap] = [
                a for a in self._capability_index[cap] if a.spec.name != name
            ]
            if not self._capability_index[cap]:
                del self._capability_index[cap]

        for tag in agent.spec.tags:
            self._tag_index[tag] = [
                a for a in self._tag_index[tag] if a.spec.name != name
            ]
            if not self._tag_index[tag]:
                del self._tag_index[tag]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseSubAgent:
        """Return the agent for *name*.

        Raises:
            KeyError: If no agent with that name is registered.
        """
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(f"No agent registered: {name!r}") from None

    def exists(self, name: str) -> bool:
        return name in self._agents

    def all(self, enabled_only: bool = False) -> list[BaseSubAgent]:
        """Return all registered agents, optionally filtering to enabled ones."""
        agents = list(self._agents.values())
        return [a for a in agents if a.spec.enabled] if enabled_only else agents

    def names(self) -> list[str]:
        return list(self._agents)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_by_capability(
        self, capability: str, enabled_only: bool = True
    ) -> list[BaseSubAgent]:
        """Return agents that declare *capability*, sorted by priority (desc)."""
        agents = self._capability_index.get(capability, [])
        if enabled_only:
            agents = [a for a in agents if a.spec.enabled]
        return sorted(agents, key=lambda a: a.spec.priority, reverse=True)

    def find_by_tag(
        self, tag: str, enabled_only: bool = True
    ) -> list[BaseSubAgent]:
        """Return agents tagged with *tag*, sorted by priority (desc)."""
        agents = self._tag_index.get(tag, [])
        if enabled_only:
            agents = [a for a in agents if a.spec.enabled]
        return sorted(agents, key=lambda a: a.spec.priority, reverse=True)

    def discover(self, query: str, enabled_only: bool = True) -> list[BaseSubAgent]:
        """Substring search across name, description, capabilities, and tags."""
        q = query.lower()
        results = []
        for agent in self._agents.values():
            if enabled_only and not agent.spec.enabled:
                continue
            haystack = " ".join([
                agent.spec.name,
                agent.spec.description,
                *agent.spec.capabilities,
                *agent.spec.tags,
            ]).lower()
            if q in haystack:
                results.append(agent)
        return sorted(results, key=lambda a: a.spec.priority, reverse=True)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        self.get(name).spec.enabled = True

    def disable(self, name: str) -> None:
        self.get(name).spec.enabled = False

    def clear(self) -> None:
        self._agents.clear()
        self._capability_index.clear()
        self._tag_index.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self):
        return iter(self._agents.values())

    def __repr__(self) -> str:
        return f"SubAgentRegistry(agents={len(self._agents)})"