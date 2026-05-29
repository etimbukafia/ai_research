import asyncio
from uuid import UUID
from pathlib import Path
from typing import Any, Optional, List, Dict
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse,
    UserPromptPart, TextPart, SystemPromptPart
)

from experiments.harness.hooks import HookPhase
from experiments.harness.session import JSONLSessionManager
from experiments.harness.agent import Agent as HarnessAgent, AgentConfig

from .memory import AderMemory, AderMemoryStateUpdate
from ..config import app_config
from .db import MockDatabase
from .prompt_builder import build_ader_prompt


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------
def load_prompts(prompt_folder: str = "prompts") -> tuple[str, str]:
    logger.debug(f"Loading prompts from folder: {prompt_folder}")
    base_path = Path(__file__).parent / prompt_folder
    memory_path = base_path / "memory_synthesizer_agent_prompt.md"
    ader_path   = base_path / "ader_agent_prompt.md"

    if not memory_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {memory_path}")
    if not ader_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {ader_path}")

    return memory_path.read_text(), ader_path.read_text()


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class AderSessionManager(JSONLSessionManager):
    def _messages_path(self, session_id: str) -> Path:
        d = self.base_path / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d / "messages.jsonl"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Ader(HarnessAgent):
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        sessions_folder: str = "sessions",
        db: Optional[MockDatabase] = None,
        user_id: Optional[UUID] = None,
        **kwargs
    ):
        memory_synthesizer_agent_prompt, ader_agent_prompt = load_prompts()
        logger.debug("Prompts loaded successfully")

        config = config or AgentConfig(
            name="Ader",
            description="A cognitive companion for neurodivergents.",
            model="gemini-3.1-flash-lite"
        )
        logger.debug(f"Agent config initialized: {config}")

        self.ader_agent = PydanticAgent(
            model=config.model,
            output_type=str,
            system_prompt=ader_agent_prompt
        )
        logger.debug("Ader agent initialized")

        self.memory_synthesizer_agent = PydanticAgent(
            model=config.model,
            output_type=AderMemoryStateUpdate,
            system_prompt=memory_synthesizer_agent_prompt
        )
        logger.debug("Memory synthesizer agent initialized")

        self.base_sessions_path = Path(__file__).parent / sessions_folder
        session_manager = AderSessionManager(self.base_sessions_path)
        logger.debug(f"AderSessionManager initialized at {self.base_sessions_path}")

        self.db = db
        self.user_id = user_id
        self._dirty_users = set()
        self._flush_timers = {}
        self._memory_cache: Dict[str, AderMemory] = {}

        # Fetch user name from DB if available
        self.user_name = "User"  # default
        if self.db and self.user_id:
            user_row = next((u for u in self.db.users if u.user_id == self.user_id), None)
            if user_row:
                self.user_name = user_row.name
                logger.debug(f"Loaded user name: {self.user_name}")

        super().__init__(agent=self.ader_agent, config=config, session_manager=session_manager, **kwargs)
        logger.debug("Ader base initialization complete")

    async def _synthesize_message_task(
        self,
        session_id: str,
        conversation_history: List[ModelMessage],
        memory: AderMemory,
        base_path: Path
    ):
        try:
            logger.debug(f"Starting synthesis task for session {session_id}")
            prompt = (
                f"Current Memory State:\n{memory.render()}\n\n"
                "Please synthesize any necessary memory updates based on the recent conversation."
            )
            logger.debug(f"Synthesis prompt constructed: {prompt}")
            result = await self.memory_synthesizer_agent.run(
                prompt, message_history=conversation_history
            )
            logger.debug("Synthesis agent returned result")

            update_payload: AderMemoryStateUpdate = result.output 
            memory.apply_update(update_payload, confidence_threshold=0.7)
            logger.debug("Memory update applied")

            if self.db is not None and self.user_id is not None:
                cache_key = str(self.user_id)
                self._dirty_users.add(cache_key)

                if cache_key in self._flush_timers:
                    self._flush_timers[cache_key].cancel()
                    logger.debug(f"Cancelled existing flush timer for user {cache_key}")

                loop = asyncio.get_running_loop()
                uid = self.user_id  # capture value now, not at lambda fire time
                self._flush_timers[cache_key] = loop.call_later(
                    180.0,
                    lambda: asyncio.create_task(self.flush_user(uid))
                )
                logger.debug(f"Scheduled new flush timer for user {cache_key} in 180 seconds")
            else:
                session_dir = base_path / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    (session_dir / "memory.json").write_text, memory.to_json()
                )
                logger.debug(f"Memory persisted to {session_dir / 'memory.json'}")
        except Exception as e:
            logger.error(f"Error in _synthesize_message_task: {e}")

    async def _execute_run(
        self, run_id: UUID, session_id: UUID, user_input: str, *args: Any, **kwargs: Any
    ) -> Any:
        logger.debug(f"_execute_run called with run_id={run_id}, session_id={session_id}, user_input={user_input}")
        session_id_str = str(session_id)

        if self.db is not None and self.user_id is not None:
            cache_key = str(self.user_id)
            if cache_key in self._memory_cache:
                mem = self._memory_cache[cache_key]
                logger.debug(f"Memory cache hit for user {cache_key}")
            else:
                mem = await asyncio.to_thread(self.db.get_ader_memory, self.user_id)
                self._memory_cache[cache_key] = mem
                logger.debug(f"Memory cache miss; loaded memory from DB for user {cache_key}")
        else:
            cache_key = session_id_str
            if cache_key in self._memory_cache:
                mem = self._memory_cache[cache_key]
                logger.debug(f"Memory cache hit for session {cache_key}")
            else:
                session_dir  = self.base_sessions_path / cache_key
                memory_path  = session_dir / "memory.json"
                mem = (
                    AderMemory.from_json(memory_path.read_text())
                    if memory_path.exists()
                    else AderMemory(name=cache_key)
                )
                self._memory_cache[cache_key] = mem
                logger.debug(f"Memory cache miss; loaded memory for session {cache_key}")

        mem.working.last_session_id = session_id

        logger.debug("Dispatching PRE_LLM hook")
        await self.dispatch_hook(
            phase=HookPhase.PRE_LLM,
            run_id=run_id,
            session_id=session_id,
            payload={"framework": "pydantic-ai", "input": user_input}
        )

        self.context_manager.add_message({"role": "user", "content": user_input})
        asyncio.create_task(
            self.session_manager.append_async(session_id_str, {"role": "user", "content": user_input})
        )

        context_history = self.context_manager.get_context()
        system_prompt = build_ader_prompt(mem, self.user_name)
        ader_history: List[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
        ]

        for msg in context_history:
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                ader_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                ader_history.append(ModelResponse(parts=[TextPart(content=content)]))
                logger.debug(f"Appended assistant message to history: {content}")

        result = await self.agent.run(user_input, message_history=ader_history[:-1])
        logger.debug(f"LLM agent returned output: {result.output}")

        self.context_manager.add_message({"role": "assistant", "content": result.output})
        asyncio.create_task(
            self.session_manager.append_async(session_id_str, {"role": "assistant", "content": result.output})
        )

        logger.debug("Dispatching POST_LLM hook")
        await self.dispatch_hook(
            phase=HookPhase.POST_LLM,
            run_id=run_id,
            step=1,
            payload={"response": result.output}
        )

        full_synthesis_history = (
            ader_history[:-1]
            + [ModelRequest(parts=[UserPromptPart(content=user_input)])]
            + [ModelResponse(parts=[TextPart(content=result.output)])]
        )
        logger.debug("Scheduling synthesis background task")
        asyncio.create_task(
            self._synthesize_message_task(
                session_id_str, full_synthesis_history, mem, self.base_sessions_path
            )
        )

        return result.output

    async def flush_user(self, user_id: UUID):
        cache_key = str(user_id)
        # Removing the timer handle first so it doesn't interfere regardless of outcome
        timer = self._flush_timers.pop(cache_key, None)
        if timer:
            timer.cancel()

        if cache_key not in self._dirty_users:
            return

        mem = self._memory_cache.get(cache_key)
        if mem and self.db is not None:
            try:
                logger.debug(f"Flushing memory to DB for user {cache_key}...")
                await asyncio.to_thread(self.db.save_ader_memory, user_id, mem)
                self._dirty_users.discard(cache_key)
                logger.debug(f"Successfully flushed memory to DB for user {cache_key}")
            except Exception as e:
                logger.error(
                    f"Flush failed for user {cache_key}: {e}. "
                    "State remains dirty and will be retried on next update or close."
                )

    async def close(self):
        logger.debug("Closing Ader agent; performing final dirty memory flush...")
        for cache_key in list(self._dirty_users):
            user_id = UUID(cache_key)
            await self.flush_user(user_id)
        logger.debug("Ader agent closed successfully.")