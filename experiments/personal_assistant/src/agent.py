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
from experiments.harness.session import SQLiteSessionManager
from experiments.harness.agent import Agent as HarnessAgent, AgentConfig

from .memory import PersonalAssistantMemory, PersonalAssistantMemoryStateUpdate
from .db import DEFAULT_DB_PATH, DEFAULT_PROFILE_ID, PersonalAssistantDatabase
from .prompt_builder import build_personal_assistant_prompt, load_prompts

from ..config import app_config


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PersonalAssistant(HarnessAgent):
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        db_path: str | Path = DEFAULT_DB_PATH,
        db: Optional[PersonalAssistantDatabase] = None,
        profile_id: UUID = DEFAULT_PROFILE_ID,
        **kwargs
    ):
        memory_synthesizer_agent_prompt, personal_assistant_agent_prompt = load_prompts()
        logger.debug("Prompts loaded successfully")

        config = config or AgentConfig(
            name="Personal Assistant",
            description="A supportive personal companion.",
            model="gemini-3.1-flash-lite"
        )
        logger.debug(f"Agent config initialized: {config}")

        self.personal_assistant_agent = PydanticAgent(
            model=config.model,
            output_type=str,
            system_prompt=personal_assistant_agent_prompt
        )
        logger.debug("Personal Assistant agent initialized")

        self.memory_synthesizer_agent = PydanticAgent(
            model=config.model,
            output_type=PersonalAssistantMemoryStateUpdate,
            system_prompt=memory_synthesizer_agent_prompt
        )
        logger.debug("Memory synthesizer agent initialized")

        self.db = db or PersonalAssistantDatabase(db_path)
        self.profile_id = profile_id
        self.db_path = Path(self.db.db_path)
        session_manager = SQLiteSessionManager(self.db_path)
        logger.debug(f"SQLiteSessionManager initialized at {self.db_path}")

        self._dirty_profiles = set()
        self._flush_timers = {}
        self._memory_cache: Dict[str, PersonalAssistantMemory] = {}
        # Holds the most recently scheduled synthesis task so callers (e.g. the
        # evaluation harness) can await it explicitly between turns rather than
        # relying on arbitrary sleeps.
        self._last_synthesis_task: Optional[asyncio.Task] = None

        self.user_name = self.db.get_profile_name(self.profile_id)
        logger.debug(f"Loaded profile name: {self.user_name}")

        super().__init__(agent=self.personal_assistant_agent, config=config, session_manager=session_manager, **kwargs)
        logger.debug("Personal Assistant base initialization complete")

    async def _synthesize_message_task(
        self,
        session_id: str,
        conversation_history: List[ModelMessage],
        memory: PersonalAssistantMemory,
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

            update_payload: PersonalAssistantMemoryStateUpdate = result.output 
            memory.apply_update(update_payload, confidence_threshold=0.7)
            logger.debug("Memory update applied")

            cache_key = str(self.profile_id)
            self._dirty_profiles.add(cache_key)

            if cache_key in self._flush_timers:
                self._flush_timers[cache_key].cancel()
                logger.debug(f"Cancelled existing flush timer for profile {cache_key}")

            loop = asyncio.get_running_loop()
            profile_id = self.profile_id
            self._flush_timers[cache_key] = loop.call_later(
                180.0,
                lambda: asyncio.create_task(self.flush_user(profile_id))
            )
            logger.debug(f"Scheduled new flush timer for profile {cache_key} in 180 seconds")

            await asyncio.to_thread(self.session_manager.save_memory, session_id, memory.to_json())
        except Exception as e:
            logger.error(f"Error in _synthesize_message_task: {e}")

    async def _execute_run(
        self, run_id: UUID, session_id: UUID, user_input: str, *args: Any, **kwargs: Any
    ) -> Any:
        logger.debug(f"_execute_run called with run_id={run_id}, session_id={session_id}, user_input={user_input}")
        session_id_str = str(session_id)

        cache_key = str(self.profile_id)
        if cache_key in self._memory_cache:
            mem = self._memory_cache[cache_key]
            logger.debug(f"Memory cache hit for profile {cache_key}")
        else:
            mem = await asyncio.to_thread(self.db.get_personal_assistant_memory, self.profile_id)
            self._memory_cache[cache_key] = mem
            logger.debug(f"Memory cache miss; loaded memory from DB for profile {cache_key}")

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
        system_prompt = build_personal_assistant_prompt(mem, self.user_name)
        personal_assistant_history: List[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
        ]

        for msg in context_history:
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                personal_assistant_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                personal_assistant_history.append(ModelResponse(parts=[TextPart(content=content)]))
                logger.debug(f"Appended assistant message to history: {content}")

        result = await self.agent.run(user_input, message_history=personal_assistant_history[:-1])
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
            personal_assistant_history[:-1]
            + [ModelRequest(parts=[UserPromptPart(content=user_input)])]
            + [ModelResponse(parts=[TextPart(content=result.output)])]
        )
        logger.debug("Scheduling synthesis background task")
        self._last_synthesis_task = asyncio.create_task(
            self._synthesize_message_task(
                session_id_str, full_synthesis_history, mem
            )
        )

        return result.output

    async def flush_user(self, user_id: UUID):
        cache_key = str(user_id)
        # Removing the timer handle first so it doesn't interfere regardless of outcome
        timer = self._flush_timers.pop(cache_key, None)
        if timer:
            timer.cancel()

        if cache_key not in self._dirty_profiles:
            return

        mem = self._memory_cache.get(cache_key)
        if mem:
            try:
                logger.debug(f"Flushing memory to DB for profile {cache_key}...")
                await asyncio.to_thread(self.db.save_personal_assistant_memory, user_id, mem)
                self._dirty_profiles.discard(cache_key)
                logger.debug(f"Successfully flushed memory to DB for profile {cache_key}")
            except Exception as e:
                logger.error(
                    f"Flush failed for user {cache_key}: {e}. "
                    "State remains dirty and will be retried on next update or close."
                )

    async def close(self):
        logger.debug("Closing Personal Assistant agent; performing final dirty memory flush...")
        for cache_key in list(self._dirty_profiles):
            user_id = UUID(cache_key)
            await self.flush_user(user_id)
        self.session_manager.close()
        self.db.close()
        logger.debug("Personal Assistant agent closed successfully.")
