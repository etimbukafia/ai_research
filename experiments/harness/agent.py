"""
Base agent abstractions for the harness.

Provides a flexible BaseAgent that integrates the harness components
(ContextManager, ToolRegistry, HookRegistry) and defines a standard lifecycle.
Developers can build custom agents by subclassing BaseAgent, or wrap existing
frameworks (like pydantic-ai) to plug them into the harness observability tools.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from experiments.harness.context.manager import ContextManager
from experiments.harness.hooks import HookContext, HookPhase, HookRegistry, HookAction
from experiments.harness.tools import ToolRegistry
from experiments.harness.session import SessionManager, JSONLSessionManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Standard configuration for agents."""
    name: str
    description: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    max_steps: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for agents in the harness.
    
    Provides standard lifecycle hooks (PRE_RUN, POST_RUN, ON_ERROR) and
    holds references to the core harness components.
    """
    def __init__(
        self,
        config: AgentConfig,
        context_manager: Optional[ContextManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        hook_registry: Optional[HookRegistry] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        self.config = config
        self.name = config.name
        
        # Initialize or assign registries
        self.hook_registry = hook_registry or HookRegistry()
        # Ensure ToolRegistry shares the same HookRegistry for integrated tracing
        self.tool_registry = tool_registry or ToolRegistry(hook_registry=self.hook_registry)
        self.context_manager = context_manager or ContextManager()
        self.session_manager = session_manager
        
        if self.config.system_prompt:
            self.context_manager.add_message({
                "role": "system", 
                "content": self.config.system_prompt
            })

    async def run(self, user_input: str, *args: Any, **kwargs: Any) -> Any:
        """
        Main entry point for running the agent.
        
        Wraps the internal execution with PRE_RUN and POST_RUN/ON_ERROR hooks.
        """
        run_id = uuid4()
        session_id = str(uuid4())

        # Fire PRE_RUN hooks
        pre_ctx = HookContext(
            run_id=run_id,
            session_id=session_id,
            step=0,
            agent_name=self.name,
            phase=HookPhase.PRE_RUN,
            payload={"user_input": user_input, "args": args, "kwargs": kwargs}
        )
        pre_decision = await self.hook_registry.dispatch_and_merge(pre_ctx)
        
        if pre_decision.action == HookAction.ABORT:
            raise RuntimeError(f"Agent run aborted by hook: {pre_decision.reason}")
        elif pre_decision.action in (HookAction.SKIP, HookAction.REPLACE):
            return pre_decision.replacement

        result: Any = None
        try:
            # Delegate to subclass implementation
            result = await self._execute_run(run_id, session_id, user_input, *args, **kwargs)
            
            # Fire POST_RUN hooks
            post_ctx = HookContext(
                run_id=run_id,
                session_id=session_id,
                step=0,
                agent_name=self.name,
                phase=HookPhase.POST_RUN,
                payload={"result": result}
            )
            await self.hook_registry.dispatch(post_ctx)
            return result
            
        except Exception as e:
            # Fire ON_ERROR hooks
            err_ctx = HookContext(
                run_id=run_id,
                session_id=session_id,
                step=0,
                agent_name=self.name,
                phase=HookPhase.ON_ERROR,
                error=e
            )
            await self.hook_registry.dispatch(err_ctx)
            raise

    async def dispatch_hook(
        self,
        phase: HookPhase,
        run_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        step: Optional[int] = 0,
        payload: Any = None,
        error: Optional[Exception] = None,
        custom_phase_name: Optional[str] = None
    ) -> Any:
        """
        Helper to quickly dispatch a hook from within subclass execution logic.
        Handles ABORT and SKIP actions automatically.
        
        Returns:
            The replacement payload if SKIP/REPLACE is chosen by a hook, otherwise None.
        """
        ctx = HookContext(
            run_id=run_id,
            step=step,
            agent_name=self.name,
            phase=phase,
            session_id=session_id,
            payload=payload,
            error=error,
            custom_phase_name=custom_phase_name
        )
        decision = await self.hook_registry.dispatch_and_merge(ctx)
        
        if decision.action == HookAction.ABORT:
            reason = decision.reason or "No reason provided"
            raise RuntimeError(f"Hook aborted execution at phase {phase.value}: {reason}")
            
        if decision.action in (HookAction.SKIP, HookAction.REPLACE):
            return decision.replacement
            
        return None

    @abstractmethod
    async def _execute_run(self, run_id: UUID, session_id: UUID, user_input: str, *args: Any, **kwargs: Any) -> Any:
        """
        Internal execution logic to be implemented by subclasses.
        
        Subclasses should:
        1. Add the user_input to the context_manager.
        2. Enter the reasoning/tool loop (e.g. a ReAct loop or framework loop).
        3. Dispatch PRE_STEP, POST_STEP, PRE_LLM, POST_LLM hooks as appropriate 
           using `await self.dispatch_hook(...)`.
        4. Return the final output.
        """
        pass


class Agent(BaseAgent):
    """
    A specialized base class for wrapping third-party frameworks (e.g. pydantic-ai, langchain) or building custom agents..
    
    This class can intercept the framework's internal events and route them to 
    the harness HookRegistry, or sync the framework's message history with the 
    ContextManager.
    """
    def __init__(self, agent: Any, config: AgentConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.agent = agent
        
    async def _execute_run(self, run_id: UUID, session_id: UUID, user_input: str, *args: Any, **kwargs: Any) -> Any:
        """
        Implementation depends heavily on the framework being wrapped.
        
        Typically you would:
        1. Sync the harness ContextManager with the framework's state.
        2. Bind harness tools (from self.tool_registry) to the framework.
        3. Execute the framework's native run method.
        4. Return the result.
        """
        raise NotImplementedError("Subclasses must implement the framework execution logic.")
