# Personal Assistant memory
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any
from datetime import date
from uuid import UUID

from experiments.harness.context.memory import MemoryBlock, MemoryType


# ---------------------------------------------------------------------------
# Memory Models
# ---------------------------------------------------------------------------

class PersonalAssistantWorkingMemory(BaseModel):
    """
    Short-lived active conversational state.
    Resets or updates every turn.
    """
    current_focus: Optional[str] = None
    active_goals: List[str] = Field(default_factory=list)
    active_tasks: List[str] = Field(default_factory=list)
    active_goals_completed: Dict[str, bool] = Field(default_factory=dict)
    open_loops: List[str] = Field(default_factory=list)
    open_loops_completed: Dict[str, bool] = Field(default_factory=dict)
    pending_decisions: List[str] = Field(default_factory=list)
    waiting_on: Dict[str, str] = Field(default_factory=dict)
    last_session_id: Optional[UUID] = None


class PersonalAssistantEpisodicMemory(BaseModel):
    """
    A remembered moment, decision, commitment, or follow-up from the user's life.
    """
    title: Optional[str] = None
    summary: Optional[str] = None
    category: Literal[
        "conversation",
        "commitment",
        "decision",
        "preference_signal",
        "life_event",
        "task_progress",
        "follow_up",
        "reflection",
        "goal",
    ] = "conversation"
    people: List[str] = Field(default_factory=list)
    related_goals: List[str] = Field(default_factory=list)
    commitments: List[str] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    occurred_on: date = Field(default_factory=date.today)


class PersonalAssistantSemanticMemory(BaseModel):
    """
    Stable, slowly-changing facts and preferences about the user.
    """
    preferences:                   List[str]                               = Field(default_factory=list)
    triggers:                      List[str]                               = Field(default_factory=list)
    prefers_direct_language:       bool                                    = True
    dislikes_open_ended_questions: bool                                    = True
    best_focus_time:               Literal["morning", "afternoon", "night"] = "night"
    sensitive_to_noise:            bool                                    = True


class PersonalAssistantProceduralMemory(BaseModel):
    """
    Accumulated knowledge of what works for this user.
    Grows over sessions; never cleared.
    """
    successful_interventions:       List[str] = Field(default_factory=list)
    routines_that_worked:           List[str] = Field(default_factory=list)
    effective_grouping_strategies:  List[str] = Field(default_factory=list)
    preferred_planning_structures:  List[str] = Field(default_factory=list)


class PersonalAssistantScratchpadMemory(BaseModel):
    """
    Temporary in-turn reasoning space.
    Cleared at the start of each new turn.
    """
    possible_causes:   List[str]        = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    next_steps:        str              = ""

    @field_validator("confidence_scores", mode="before")
    @classmethod
    def clamp_confidences(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {k: _clamp(float(val)) for k, val in v.items()}


class AffectiveState(BaseModel):
    """
    Continuously updated emotional and cognitive energy model.
    All values are floats in [0.0, 1.0].
    Lower = depleted / stressed; Higher = resourced / regulated.
    """
    stress_level:         float = Field(default=0.5, ge=0.0, le=1.0)
    energy_level:         float = Field(default=0.5, ge=0.0, le=1.0)
    cognitive_load:       float = Field(default=0.5, ge=0.0, le=1.0)
    social_energy:        float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_regulation: float = Field(default=0.5, ge=0.0, le=1.0)
    executive_function:   float = Field(default=0.5, ge=0.0, le=1.0)

    def is_overloaded(self) -> bool:
        """True when the user is likely cognitively or emotionally overwhelmed."""
        return (
            self.cognitive_load > 0.8
            or self.stress_level > 0.8
            or self.executive_function < 0.2
        )

    def summary(self) -> str:
        """Returns a compact, LLM-ready string of the current affective state."""
        status = "overloaded" if self.is_overloaded() else "regulated"
        return (
            f"[Affective: {status} | "
            f"energy={self.energy_level:.1f} "
            f"stress={self.stress_level:.1f} "
            f"cog_load={self.cognitive_load:.1f} "
            f"exec_fn={self.executive_function:.1f}]"
        )


# ---------------------------------------------------------------------------
# Memory State Update Schema (For Background Worker Synthesis)
# ---------------------------------------------------------------------------

class PersonalAssistantMemoryStateUpdate(BaseModel):
    """
    Schema for the LLM to output when synthesizing new memory updates.
    Fields with a confidence score < 0.7 will be ignored by apply_update.
    """
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Confidence scores (0.0 to 1.0) for each field updated. "
            "Keys must match the exact field names below."
        ),
    )

    # Working memory
    current_focus: Optional[str] = Field(None, description="What the user is currently focused on.")
    active_goals: List[str] = Field(default_factory=list, description="New active short-term goals.")
    active_tasks: List[str] = Field(default_factory=list, description="Immediate concrete tasks the user is working on.")
    open_loops:   List[str] = Field(
        default_factory=list,
        description="Thoughts, anxieties, or unresolved tasks occupying the user's working memory.",
    )
    pending_decisions: List[str] = Field(default_factory=list, description="Decisions the user has not made yet.")
    waiting_on: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of what the user is waiting on to who or what is blocking it.",
    )
    waiting_on_resolved: List[str] = Field(
        default_factory=list,
        description="Waiting-on item keys that are now resolved and should be removed.",
    )

    # Semantic memory
    preferences: List[str] = Field(default_factory=list, description="New long-term user preferences discovered.")
    triggers:    List[str] = Field(default_factory=list, description="New triggers or sensitivities discovered.")
    best_focus_time: Optional[Literal["morning", "afternoon", "night"]] = Field(
        None, description="User's best focus time."
    )
    sensitive_to_noise:            Optional[bool] = Field(None, description="Whether the user is sensitive to noise.")
    prefers_direct_language:       Optional[bool] = Field(None, description="Whether the user prefers direct language.")
    dislikes_open_ended_questions: Optional[bool] = Field(None, description="Whether the user dislikes open-ended questions.")

    # Procedural memory
    successful_interventions:      List[str] = Field(default_factory=list, description="New successful support interventions.")
    routines_that_worked:          List[str] = Field(default_factory=list, description="New routines that worked well.")
    effective_grouping_strategies: List[str] = Field(default_factory=list, description="New effective task grouping strategies.")
    preferred_planning_structures: List[str] = Field(default_factory=list, description="New preferred planning structures.")

    # Affective state updates
    stress_level:         Optional[float] = Field(None, description="User's stress level [0.0, 1.0]")
    energy_level:         Optional[float] = Field(None, description="User's energy level [0.0, 1.0]")
    cognitive_load:       Optional[float] = Field(None, description="User's cognitive load [0.0, 1.0]")
    social_energy:        Optional[float] = Field(None, description="User's social energy [0.0, 1.0]")
    emotional_regulation: Optional[float] = Field(None, description="User's emotional regulation capacity [0.0, 1.0]")
    executive_function:   Optional[float] = Field(None, description="User's executive function capacity [0.0, 1.0]")

    # Episodic
    episode_title: Optional[str] = Field(None, description="Short label for a remembered moment, decision, goal, commitment, or follow-up.")
    episode_summary: Optional[str] = Field(None, description="Brief summary of what happened and why it matters.")
    episode_category: Optional[Literal[
        "conversation",
        "commitment",
        "decision",
        "preference_signal",
        "life_event",
        "task_progress",
        "follow_up",
        "reflection",
        "goal",
    ]] = Field(None, description="Type of episode to save.")
    episode_people: List[str] = Field(default_factory=list, description="People involved or mentioned.")
    episode_related_goals: List[str] = Field(default_factory=list, description="Goals this episode connects to.")
    episode_commitments: List[str] = Field(default_factory=list, description="Commitments made by the user or assistant.")
    episode_follow_ups: List[str] = Field(default_factory=list, description="Future follow-ups the assistant should remember.")
    episode_risks: List[str] = Field(default_factory=list, description="Risks, blockers, or failure modes connected to this episode.")
    episode_salience: Optional[float] = Field(None, description="Importance for future recall [0.0, 1.0].")


# ---------------------------------------------------------------------------
# PersonalAssistantMemory - top-level memory block
# ---------------------------------------------------------------------------


class PersonalAssistantMemory(MemoryBlock):
    """
    Top-level memory for the Personal Assistant companion.

    Composes all five memory tiers and an affective state model on top of
    MemoryBlock.  The `content` field holds the rendered LLM-ready snapshot
    produced by `render()`; call it before passing this block to the model.

    Usage
    -----
        mem = PersonalAssistantMemory(name="personal_assistant_session")
        mem.working.active_goals.append("finish the report")
        mem.affective.stress_level = 0.75
        mem.episodic.append(PersonalAssistantEpisodicMemory(title="Report plan", category="goal"))
        mem.update(mem.render())   # sync content before LLM call
    """

    memory_type: MemoryType = MemoryType.WORKING  # top-level type

    # Memory tiers
    working:    PersonalAssistantWorkingMemory        = Field(default_factory=PersonalAssistantWorkingMemory)
    semantic:   PersonalAssistantSemanticMemory       = Field(default_factory=PersonalAssistantSemanticMemory)
    procedural: PersonalAssistantProceduralMemory     = Field(default_factory=PersonalAssistantProceduralMemory)
    scratchpad: PersonalAssistantScratchpadMemory     = Field(default_factory=PersonalAssistantScratchpadMemory)
    episodic:   List[PersonalAssistantEpisodicMemory] = Field(default_factory=list)

    # Continuously updated affective model
    affective: AffectiveState = Field(default_factory=AffectiveState)

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def add_episode(self, **kwargs: Any) -> PersonalAssistantEpisodicMemory:
        """Create and record a new episodic memory."""
        episode = PersonalAssistantEpisodicMemory(**kwargs)
        self.episodic.append(episode)
        self._touch()
        return episode

    def recent_episodes(self, n: int = 5) -> List[PersonalAssistantEpisodicMemory]:
        """Return the n most recent episodes."""
        return self.episodic[-n:]

    # ------------------------------------------------------------------
    # Scratchpad helpers
    # ------------------------------------------------------------------

    def reset_scratchpad(self) -> None:
        """Clear the scratchpad at the start of each new turn."""
        self.scratchpad = PersonalAssistantScratchpadMemory()
        self._touch()

    # ------------------------------------------------------------------
    # Synthesis Update Applier
    # ------------------------------------------------------------------

    def apply_update(self, update: PersonalAssistantMemoryStateUpdate, confidence_threshold: float = 0.7) -> None:
        """Applies updates from the LLM synthesis, ignoring low-confidence values."""

        def check_conf(field_name: str) -> bool:
            return update.confidence_scores.get(field_name, 1.0) >= confidence_threshold

        self._update_affective(update, check_conf)
        self._update_working(update, check_conf)
        self._update_semantic(update, check_conf)
        self._update_procedural(update, check_conf)
        self._update_episodic(update, check_conf)

        self._touch()

    def _update_affective(self, update: PersonalAssistantMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["stress_level", "energy_level", "cognitive_load", "social_energy", "emotional_regulation", "executive_function"]:
            if getattr(update, f) is not None and check_conf(f):
                setattr(self.affective, f, getattr(update, f))

    def _update_working(self, update: PersonalAssistantMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["current_focus"]:
            val = getattr(update, f)
            if val is not None and check_conf(f):
                setattr(self.working, f, val)

        for f in ["active_goals", "active_tasks", "open_loops", "pending_decisions"]:
            val = getattr(update, f)
            if val and check_conf(f):
                setattr(self.working, f, val)

        if update.waiting_on and check_conf("waiting_on"):
            self.working.waiting_on.update(update.waiting_on)

        if update.waiting_on_resolved and check_conf("waiting_on_resolved"):
            for item in update.waiting_on_resolved:
                self.working.waiting_on.pop(item, None)

    def _update_semantic(self, update: PersonalAssistantMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["preferences", "triggers"]:
            val = getattr(update, f)
            if val and check_conf(f):
                getattr(self.semantic, f).extend(val)

        for f in ["best_focus_time", "sensitive_to_noise", "prefers_direct_language", "dislikes_open_ended_questions"]:
            val = getattr(update, f)
            if val is not None and check_conf(f):
                setattr(self.semantic, f, val)

    def _update_procedural(self, update: PersonalAssistantMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["successful_interventions", "routines_that_worked", "effective_grouping_strategies", "preferred_planning_structures"]:
            val = getattr(update, f)
            if val and check_conf(f):
                getattr(self.procedural, f).extend(val)

    def _update_episodic(self, update: PersonalAssistantMemoryStateUpdate, check_conf: callable) -> None:
        if update.episode_title and check_conf("episode_title"):
            self.add_episode(
                title=update.episode_title,
                summary=update.episode_summary,
                category=update.episode_category or "conversation",
                people=update.episode_people,
                related_goals=update.episode_related_goals,
                commitments=update.episode_commitments,
                follow_ups=update.episode_follow_ups,
                risks=update.episode_risks,
                salience=update.episode_salience if update.episode_salience is not None else 0.5,
            )

    # ------------------------------------------------------------------
    # Rendering - serializes structured state into LLM-ready content
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Serialize all memory tiers into a single structured string
        suitable for injection as a system message.

        Call `mem.update(mem.render())` before each LLM turn to sync
        the `content` field.
        """
        w  = self.working
        s  = self.semantic
        p  = self.procedural
        sc = self.scratchpad

        lines: List[str] = [
            "=== Personal Assistant MEMORY SNAPSHOT ===",
            "",
            f"[Affective] {self.affective.summary()}",
            "",
            "## Working Memory",
            f"Current Focus:     {w.current_focus or 'none'}",
            f"Active Goals:      {', '.join(w.active_goals) or 'none'}",
            f"Active Tasks:      {', '.join(w.active_tasks) or 'none'}",
            f"Open Loops:        {', '.join(w.open_loops) or 'none'}",
            f"Pending Decisions: {', '.join(w.pending_decisions) or 'none'}",
            f"Waiting On:        {_format_waiting_on(w.waiting_on)}",
            "",
            "## Semantic Memory",
            f"Best Focus Time:   {s.best_focus_time}",
            f"Direct Language:   {s.prefers_direct_language}",
            f"Noise Sensitive:   {s.sensitive_to_noise}",
            f"Preferences:       {', '.join(s.preferences) or 'none'}",
            f"Known Triggers:    {', '.join(s.triggers) or 'none'}",
            "",
            "## Procedural Memory",
            f"Interventions:     {', '.join(p.successful_interventions) or 'none'}",
            f"Working Routines:  {', '.join(p.routines_that_worked) or 'none'}",
            f"Planning Style:    {', '.join(p.preferred_planning_structures) or 'none'}",
        ]

        recent = self.recent_episodes(5)
        if recent:
            lines += ["", "## Recent Episodes"]
            for ep in recent:
                ts = ep.occurred_on.isoformat() if ep.occurred_on else "unknown"
                lines.append(
                    f"  [{ts}] {ep.category}: {ep.title or '-'} | "
                    f"{ep.summary or '-'} | follow-ups: {', '.join(ep.follow_ups) or 'none'} "
                    f"| risks: {', '.join(ep.risks) or 'none'}"
                )

        if sc.possible_causes or sc.next_steps:
            lines += ["", "## Scratchpad"]
            if sc.possible_causes:
                lines.append(f"Possible Causes: {', '.join(sc.possible_causes)}")
            if sc.next_steps:
                lines.append(f"Next Steps:      {sc.next_steps}")

        lines.append("")
        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize full memory state to JSON (for session persistence)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "PersonalAssistantMemory":
        """Restore memory state from a persisted JSON string."""
        return cls.model_validate_json(data)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _clamp(val: float) -> float:
    return max(0.0, min(1.0, val))


def _format_waiting_on(waiting_on: Dict[str, str]) -> str:
    if not waiting_on:
        return "none"
    return "; ".join(f"{item} -> {blocker}" for item, blocker in waiting_on.items())


PersonalAssistantMemory.model_rebuild()
