import asyncio
from uuid import UUID
from pathlib import Path
from typing import Any, Optional, List, Dict
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse,
    UserPromptPart, TextPart, SystemPromptPart
)

from experiments.harness.hooks import HookPhase
from experiments.harness.session import SQLiteSessionManager
from experiments.harness.agent import Agent as HarnessAgent, AgentConfig

from .context_renderer import PersonalContextRenderer
from .context_synthesis import ContextSynthesisResult
from .conversation_format import format_recent_conversation
from .ddc import DDCReviewService
from .entities import ContextEntityService
from .execution import AssistantExecutionResult
from .memory import PersonalAssistantMemory
from .orchestration import AssistantFSM, TurnIntent, TurnRoute, VerificationResult
from .planner_ddc import PlannerDDCBridge, continuation_answer_context, continuation_source_summary
from .planning import MissingInfoItem, PlannerRuntimeService, TaskPlan
from .db import DEFAULT_DB_PATH, DEFAULT_PROFILE_ID, PersonalAssistantDatabase
from .prompt_builder import (
    build_personal_assistant_prompt,
    load_context_synthesizer_prompt,
    load_intent_classifier_prompt,
    load_planner_prompt,
    load_prompts,
    load_verifier_prompt,
)

from ..config import app_config


def _model_for(config: AgentConfig, metadata_key: str, configured_default: str) -> str:
    value = config.metadata.get(metadata_key)
    if isinstance(value, str) and value.strip():
        return value
    return configured_default or config.model


def _original_task_for_continuation(execution_input: str, fallback: str) -> str:
    if not execution_input.startswith("Original task:\n"):
        return fallback
    remainder = execution_input.removeprefix("Original task:\n")
    return remainder.split("\n\n", 1)[0].strip() or fallback


def _format_checklist_item(item: MissingInfoItem) -> str:
    label = item.label.strip() or item.category
    answer = (item.answer or "").strip() or "unresolved"
    required = item.required_confidence if item.required_confidence is not None else 0.0
    lines = [
        f"- {label}: {item.status or 'unknown'}",
        f"  Question: {item.question}",
        f"  Risk: {item.risk_level}",
        f"  Confidence: {item.confidence:.2f} / required {required:.2f}",
        f"  Source: {item.answer_source}",
        f"  Answer: {answer}",
    ]
    if item.why_needed.strip():
        lines.append(f"  Why needed: {item.why_needed.strip()}")
    return "\n".join(lines)


def _format_plan_checklist(plan: TaskPlan) -> str:
    if not plan.gaps:
        return "Checklist: none"
    return "Checklist:\n" + "\n".join(_format_checklist_item(item) for item in plan.gaps)


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
        _, personal_assistant_agent_prompt = load_prompts()
        context_synthesizer_agent_prompt = load_context_synthesizer_prompt()
        planner_agent_prompt = load_planner_prompt()
        verifier_agent_prompt = load_verifier_prompt()
        intent_classifier_agent_prompt = load_intent_classifier_prompt()

        config = config or AgentConfig(
            name="Personal Assistant",
            description="A supportive personal companion.",
            model=app_config.PERSONAL_ASSISTANT_AGENT_MODEL,
        )
        personal_assistant_model = _model_for(
            config,
            "personal_assistant_model",
            app_config.PERSONAL_ASSISTANT_AGENT_MODEL,
        )
        context_synthesizer_model = _model_for(
            config,
            "context_synthesizer_model",
            app_config.CONTEXT_SYNTHESIZER_AGENT_MODEL,
        )
        planner_model = _model_for(
            config,
            "planner_model",
            app_config.PLANNER_AGENT_MODEL,
        )
        verifier_model = _model_for(
            config,
            "verifier_model",
            app_config.VERIFIER_AGENT_MODEL,
        )
        intent_classifier_model = _model_for(
            config,
            "intent_classifier_model",
            app_config.INTENT_CLASSIFIER_AGENT_MODEL,
        )

        self.personal_assistant_agent = PydanticAgent(
            model=personal_assistant_model,
            output_type=AssistantExecutionResult,
            system_prompt=personal_assistant_agent_prompt
        )   

        self.context_synthesizer_agent = PydanticAgent(
            model=context_synthesizer_model,
            output_type=ContextSynthesisResult,
            system_prompt=context_synthesizer_agent_prompt
        )

        self.planner_agent = PydanticAgent(
            model=planner_model,
            output_type=TaskPlan,
            system_prompt=planner_agent_prompt
        )

        self.verifier_agent = PydanticAgent(
            model=verifier_model,
            output_type=VerificationResult,
            system_prompt=verifier_agent_prompt
        )

        self.intent_classifier_agent = PydanticAgent(
            model=intent_classifier_model,
            output_type=TurnIntent,
            system_prompt=intent_classifier_agent_prompt
        )

        self.db = db or PersonalAssistantDatabase(db_path)
        self.ddc_review_service = DDCReviewService(self.db)
        self.entity_service = ContextEntityService(self.db)
        self.planner_runtime_service = PlannerRuntimeService(self.db)
        self.planner_ddc_bridge = PlannerDDCBridge(self.ddc_review_service)
        self.fsm = AssistantFSM()
        self.context_renderer = PersonalContextRenderer(self.db)
        self.profile_id = profile_id
        self.db_path = Path(self.db.db_path)
        session_manager = SQLiteSessionManager(self.db_path)

        self._dirty_profiles = set()
        self._flush_timers = {}
        self._memory_cache: Dict[str, PersonalAssistantMemory] = {}
        self._memory_cache_revisions: Dict[str, int] = {}
        # Holds the most recently scheduled synthesis task so callers can await it explicitly between turns rather than
        # relying on arbitrary sleeps.
        self._last_context_synthesis_task: Optional[asyncio.Task] = None

        self.user_name = self.db.get_profile_name(self.profile_id)

        super().__init__(agent=self.personal_assistant_agent, config=config, session_manager=session_manager, **kwargs)

    async def _synthesize_context_task(
        self,
        *,
        session_id: UUID,
        session_id_str: str,
        user_input: str,
        assistant_output: str,
        memory: PersonalAssistantMemory,
        runtime_context: str,
        source_task: str,
        recent_conversation: str,
        create_ddc_reviews: bool = True,
    ) -> None:
        try:
            approved_entities = await asyncio.to_thread(
                self.entity_service.approved,
                self.profile_id,
                limit=100,
            )
            pending_entities = await asyncio.to_thread(
                self.entity_service.pending,
                self.profile_id,
            )
            approved_text = "\n".join(
                f"- {entity.name} ({entity.entity_type}); aliases={entity.aliases}; context={entity.description}"
                for entity in approved_entities
            ) or "None"
            pending_text = "\n".join(
                f"- {item.name} ({item.entity_type}); aliases={item.aliases}; context={item.description}"
                for item in pending_entities
            ) or "None"
            prompt = (
                f"Current Memory State:\n{memory.render()}\n\n"
                f"Approved Runtime Context:\n{runtime_context}\n\n"
                f"Approved Entities:\n{approved_text}\n\n"
                f"Pending Entity Proposals:\n{pending_text}\n\n"
                f"Recent Conversation:\n{recent_conversation or 'No recent conversation available.'}\n\n"
                f"User Task:\n{user_input}\n\n"
                f"Assistant Response:\n{assistant_output}\n\n"
                "Synthesize memory updates, DDC review proposals, and entity review proposals."
            )
            result = await self.context_synthesizer_agent.run(prompt)
            synthesis: ContextSynthesisResult = result.output

            cache_key = str(self.profile_id)
            if synthesis.memory_update is not None:
                memory.apply_update(synthesis.memory_update, confidence_threshold=0.7)
                self._dirty_profiles.add(cache_key)
                if cache_key in self._flush_timers:
                    self._flush_timers[cache_key].cancel()
                loop = asyncio.get_running_loop()
                profile_id = self.profile_id
                self._flush_timers[cache_key] = loop.call_later(
                    180.0,
                    lambda: asyncio.create_task(self.flush_user(profile_id)),
                )
                await asyncio.to_thread(self.session_manager.save_memory, session_id_str, memory.to_json())
                revision = await asyncio.to_thread(self.db.increment_context_revision, self.profile_id)
                self._memory_cache_revisions[cache_key] = revision

            if create_ddc_reviews and synthesis.ddc_review_items:
                created = await asyncio.to_thread(
                    self.ddc_review_service.create_review_items,
                    profile_id=self.profile_id,
                    source_task=source_task,
                    proposals=synthesis.ddc_review_items,
                    session_id=session_id,
                )
                logger.debug("Created %s DDC review item(s)", len(created))

            if synthesis.entity_review_items:
                created = await asyncio.to_thread(
                    self.entity_service.create_review_items,
                    profile_id=self.profile_id,
                    source_task=source_task,
                    proposals=synthesis.entity_review_items,
                    session_id=session_id,
                )
                logger.debug("Created %s entity review item(s)", len(created))
        except Exception as e:
            logger.error(f"Error in _synthesize_context_task: {e}")

    async def _plan_task(
        self,
        user_input: str,
        memory: PersonalAssistantMemory,
        runtime_context: str,
    ) -> TaskPlan:
        prompt = (
            f"Current Memory State:\n{memory.render()}\n\n"
            f"Approved Runtime Context:\n{runtime_context}\n\n"
            f"User Task:\n{user_input}\n\n"
            "Return the runtime plan."
        )
        result = await self.planner_agent.run(prompt)
        return result.output

    async def _create_review_items_for_continuation_answer(
        self,
        *,
        session_id: UUID,
        continuation,
        user_answer: str,
        memory: PersonalAssistantMemory,
        runtime_context: str,
    ):
        context = continuation_answer_context(continuation, user_answer)
        source_task = continuation_source_summary(continuation)
        prompt = (
            f"Current Memory State:\n{memory.render()}\n\n"
            f"Approved Runtime Context:\n{runtime_context}\n\n"
            f"Recent Conversation:\n{context}\n\n"
            f"User Task:\n{source_task}\n\n"
            "The user answered a paused task's red checklist. Create DDC review proposals only for "
            "answers that are useful as reviewable durable context. Return no entity proposals unless "
            "the answer defines a stable named concept, person, tool, system, document, or project."
        )
        try:
            result = await self.context_synthesizer_agent.run(prompt)
            synthesis: ContextSynthesisResult = result.output
            if not synthesis.ddc_review_items:
                return []
            return await asyncio.to_thread(
                self.ddc_review_service.create_review_items,
                profile_id=self.profile_id,
                source_task=source_task,
                proposals=synthesis.ddc_review_items,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error creating DDC review items for continuation answer: %s", e)
            return await asyncio.to_thread(
                self.planner_ddc_bridge.create_review_items_for_continuation_answer,
                profile_id=self.profile_id,
                continuation=continuation,
                user_answer=user_answer,
                session_id=session_id,
            )

    async def _verify_response(
        self,
        *,
        user_input: str,
        assistant_output: str,
        runtime_context: str,
        planner_context: str,
    ) -> str:
        prompt = (
            f"User Task:\n{user_input}\n\n"
            f"Approved Runtime Context:\n{runtime_context}\n\n"
            f"Planner/FSM Context:\n{planner_context}\n\n"
            f"Proposed Assistant Response:\n{assistant_output}\n\n"
            "Verify this response."
        )
        result = await self.verifier_agent.run(prompt)
        verification: VerificationResult = result.output
        if not verification.passed and verification.revised_response:
            logger.debug("Verifier revised response due to issues: %s", verification.issues)
            return verification.revised_response
        return assistant_output

    async def _route_turn(self, user_input: str, *, has_pending_continuation: bool) -> TurnRoute:
        deterministic = self.fsm.deterministic_route(
            user_input,
            has_pending_continuation=has_pending_continuation,
        )
        if deterministic is not None:
            return deterministic

        prompt = (
            f"Pending continuation: {has_pending_continuation}\n\n"
            f"User message:\n{user_input}\n\n"
            "Classify this turn for routing."
        )
        result = await self.intent_classifier_agent.run(prompt)
        intent: TurnIntent = result.output
        return self.fsm.route_from_intent(intent)

    async def _execute_run(
        self, run_id: UUID, session_id: UUID, user_input: str, *args: Any, **kwargs: Any
    ) -> Any:
        session_id_str = str(session_id)
        session_uuid = session_id if isinstance(session_id, UUID) else UUID(session_id_str)

        cache_key = str(self.profile_id)
        current_revision = await asyncio.to_thread(self.db.get_context_revision, self.profile_id)
        cached_revision = self._memory_cache_revisions.get(cache_key)
        if cache_key in self._memory_cache and cached_revision == current_revision:
            mem = self._memory_cache[cache_key]
        else:
            mem = await asyncio.to_thread(self.db.get_personal_assistant_memory, self.profile_id)
            self._memory_cache[cache_key] = mem
            self._memory_cache_revisions[cache_key] = current_revision

        mem.working.last_session_id = session_uuid

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
        runtime_context = self.context_renderer.render_if_stale(self.profile_id)
        ddc_review_created_this_turn = False
        analysis_user_input = user_input
        review_source_task = user_input

        pending_continuation = await asyncio.to_thread(
            self.db.get_pending_planner_continuation,
            self.profile_id,
        )
        route = await self._route_turn(user_input, has_pending_continuation=pending_continuation is not None)
        logger.debug("FSM route: %s", route)

        if route.state == "cancel_continuation" and pending_continuation is not None:
            await asyncio.to_thread(
                self.db.set_planner_continuation_status,
                self.profile_id,
                pending_continuation.continuation_id,
                "cancelled",
            )
            result_output = "Cancelled the paused task."
            self.context_manager.add_message({"role": "assistant", "content": result_output})
            asyncio.create_task(
                self.session_manager.append_async(session_id_str, {"role": "assistant", "content": result_output})
            )
            await self.dispatch_hook(
                phase=HookPhase.POST_LLM,
                run_id=run_id,
                step=1,
                payload={"response": result_output, "cancelled_planner_continuation": True}
            )
            return result_output

        if route.state == "resume_continuation" and pending_continuation is not None:
            answered_questions = continuation_answer_context(pending_continuation, user_input)
            review_source_task = continuation_source_summary(pending_continuation)
            analysis_user_input = (
                f"{review_source_task}\n\n"
                "Resolved checklist answers:\n"
                f"{answered_questions}"
            )
            continuation_review_items = await self._create_review_items_for_continuation_answer(
                session_id=session_uuid,
                continuation=pending_continuation,
                user_answer=user_input,
                memory=mem,
                runtime_context=runtime_context,
            )
            ddc_review_created_this_turn = bool(continuation_review_items)
            await asyncio.to_thread(
                self.db.set_planner_continuation_status,
                self.profile_id,
                pending_continuation.continuation_id,
                "completed",
            )
            planner_context = (
                "## Current Task Plan\n"
                f"Original task: {pending_continuation.original_user_task}\n"
                f"Planner output: {pending_continuation.planner_output_json}\n"
                "Resolved red checklist items for this task only:\n"
                f"{answered_questions}\n"
            )
            execution_input = (
                f"Original task:\n{pending_continuation.original_user_task}\n\n"
                "The user resolved all red checklist item(s):\n"
                f"{answered_questions}\n\n"
                "Use these answers as supplied context for this task only unless they are later approved "
                "as durable context. Do not ask the same blocking question again when the answer is usable. "
                "Now execute the original task."
            )
        elif route.run_planner:
            task_plan = await self._plan_task(user_input, mem, runtime_context)
            if self.planner_runtime_service.is_blocked(task_plan):
                continuation = await asyncio.to_thread(
                    self.planner_runtime_service.create_continuation,
                    profile_id=self.profile_id,
                    original_user_task=user_input,
                    plan=task_plan,
                    session_id=session_uuid,
                )
                logger.debug("Created planner continuation: %s", continuation.continuation_id)
                result_output = self.planner_runtime_service.blocking_message(task_plan)
                self.context_manager.add_message({"role": "assistant", "content": result_output})
                asyncio.create_task(
                    self.session_manager.append_async(session_id_str, {"role": "assistant", "content": result_output})
                )
                await self.dispatch_hook(
                    phase=HookPhase.POST_LLM,
                    run_id=run_id,
                    step=1,
                    payload={"response": result_output, "blocked_by_planner": True}
                )
                return result_output

            assumption_review_items = await asyncio.to_thread(
                self.planner_ddc_bridge.create_review_items_for_assumptions,
                profile_id=self.profile_id,
                source_task=user_input,
                assumptions=task_plan.assumptions,
                session_id=session_uuid,
            )
            ddc_review_created_this_turn = bool(assumption_review_items)
            planner_context = (
                "## Current Task Plan\n"
                f"Objective: {task_plan.objective}\n"
                f"{_format_plan_checklist(task_plan)}\n"
                "Steps:\n"
                + "\n".join(f"- {step}" for step in task_plan.steps)
                + "\nAssumptions:\n"
                + "\n".join(f"- {assumption.text}" for assumption in task_plan.assumptions)
            )
            execution_input = user_input
        else:
            planner_context = (
                "## Current Turn State\n"
                f"State: {route.state}\n"
                f"Reason: {route.reason}\n"
                "Planner skipped for this turn."
            )
            execution_input = user_input

        system_prompt = build_personal_assistant_prompt(mem, self.user_name, runtime_context=runtime_context)
        system_prompt = f"{system_prompt}\n\n---\n\n{planner_context}"
        personal_assistant_history: List[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
        ]

        for msg in context_history:
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                personal_assistant_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                personal_assistant_history.append(ModelResponse(parts=[TextPart(content=content)]))

        result = await self.agent.run(execution_input, message_history=personal_assistant_history[:-1])
        execution_result: AssistantExecutionResult = result.output
        if execution_result.kind == "context_gap" and execution_result.context_gap is not None:
            continuation = await asyncio.to_thread(
                self.planner_runtime_service.create_continuation_for_context_gap,
                profile_id=self.profile_id,
                original_user_task=_original_task_for_continuation(execution_input, user_input),
                summary=execution_result.context_gap.summary,
                gaps=execution_result.context_gap.blocking_items,
                session_id=session_uuid,
            )
            logger.debug("Created execution-gap continuation: %s", continuation.continuation_id)
            result_output = execution_result.user_message()
            self.context_manager.add_message({"role": "assistant", "content": result_output})
            asyncio.create_task(
                self.session_manager.append_async(session_id_str, {"role": "assistant", "content": result_output})
            )
            await self.dispatch_hook(
                phase=HookPhase.POST_LLM,
                run_id=run_id,
                step=1,
                payload={"response": result_output, "blocked_by_execution_gap": True}
            )
            return result_output

        result_output = execution_result.user_message()
        if route.run_verifier:
            result_output = await self._verify_response(
                user_input=execution_input,
                assistant_output=result_output,
                runtime_context=runtime_context,
                planner_context=planner_context,
            )
        recent_conversation = format_recent_conversation(context_history, result_output)

        self.context_manager.add_message({"role": "assistant", "content": result_output})
        asyncio.create_task(
            self.session_manager.append_async(session_id_str, {"role": "assistant", "content": result_output})
        )

        await self.dispatch_hook(
            phase=HookPhase.POST_LLM,
            run_id=run_id,
            step=1,
            payload={"response": result_output, "fsm_route": route.state}
        )

        if route.run_memory_synthesis:
            self._last_context_synthesis_task = asyncio.create_task(
                self._synthesize_context_task(
                    session_id=session_uuid,
                    session_id_str=session_id_str,
                    user_input=analysis_user_input,
                    assistant_output=result_output,
                    memory=mem,
                    runtime_context=runtime_context,
                    source_task=review_source_task,
                    recent_conversation=recent_conversation,
                    create_ddc_reviews=route.run_ddc_analysis and not ddc_review_created_this_turn,
                )
            )

        return result_output

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
                await asyncio.to_thread(self.db.save_personal_assistant_memory, user_id, mem)
                self._dirty_profiles.discard(cache_key)
            except Exception as e:
                logger.error(
                    f"Flush failed for user {cache_key}: {e}. "
                    "State remains dirty and will be retried on next update or close."
                )

    async def wait_for_context_review_updates(self) -> None:
        """Wait for the latest post-turn context synthesis, if one is running."""
        task = self._last_context_synthesis_task
        if task and not task.done():
            try:
                await task
            except Exception as e:
                logger.error(f"Context synthesis task failed: {e}")

    async def invalidate_memory_cache(self, user_id: UUID | None = None) -> None:
        """Flush dirty memory, then force the next turn to reload memory from SQLite."""
        user_id = user_id or self.profile_id
        cache_key = str(user_id)
        if cache_key in self._dirty_profiles:
            await self.flush_user(user_id)
        self._memory_cache.pop(cache_key, None)
        self._memory_cache_revisions.pop(cache_key, None)

    async def close(self):
        for task in (self._last_context_synthesis_task,):
            if task and not task.done():
                try:
                    await task
                except Exception as e:
                    logger.error(f"Background task failed during close: {e}")
        for cache_key in list(self._dirty_profiles):
            user_id = UUID(cache_key)
            await self.flush_user(user_id)
        self.session_manager.close()
        self.db.close()
        logger.debug("Personal Assistant agent closed successfully.")
