# Ader memory
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any
from datetime import date
from uuid import UUID

from experiments.harness.context.memory import MemoryBlock, MemoryType


# ---------------------------------------------------------------------------
# Memory Models
# ---------------------------------------------------------------------------

class AderWorkingMemory(BaseModel):
    """
    Short-lived active conversational state.
    Resets or updates every turn.
    """
    mode: Literal["calm", "organized", "planning", "reflective", "low_stimulation"] = "calm"
    user_emotional_state: str = "neutral"
    active_goals: List[str] = Field(default_factory=list)
    active_goals_completed: Dict[str, bool] = Field(default_factory=dict)
    open_loops: List[str] = Field(default_factory=list)
    open_loops_completed: Dict[str, bool] = Field(default_factory=dict)
    last_session_id: Optional[UUID] = None


class AderEpisodicMemory(BaseModel):
    """
    A single narrative experience tied to a moment in time.
    Append episodes to a list; never overwrite.
    """
    event:     Optional[str]  = None
    trigger:   Optional[str]  = None
    response:  Optional[str]  = None
    outcome:   Optional[str]  = None
    timestamp: Optional[date] = Field(default_factory=date.today)


class AderSemanticMemory(BaseModel):
    """
    Stable, slowly-changing facts and preferences about the user.
    """
    preferences:                   List[str]                               = Field(default_factory=list)
    triggers:                      List[str]                               = Field(default_factory=list)
    prefers_direct_language:       bool                                    = True
    dislikes_open_ended_questions: bool                                    = True
    best_focus_time:               Literal["morning", "afternoon", "night"] = "night"
    sensitive_to_noise:            bool                                    = True


class AderProceduralMemory(BaseModel):
    """
    Accumulated knowledge of what works for this user.
    Grows over sessions; never cleared.
    """
    successful_interventions:       List[str] = Field(default_factory=list)
    routines_that_worked:           List[str] = Field(default_factory=list)
    effective_grouping_strategies:  List[str] = Field(default_factory=list)
    preferred_planning_structures:  List[str] = Field(default_factory=list)


class AderScratchpadMemory(BaseModel):
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

class AderMemoryStateUpdate(BaseModel):
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
    mode: Optional[Literal["calm", "organized", "planning", "reflective", "low_stimulation"]] = Field(
        None, description="Current operational mode of the companion."
    )
    user_emotional_state: Optional[str] = Field(
        None, description="The user's currently inferred emotional state in a short phrase."
    )
    active_goals: List[str] = Field(default_factory=list, description="New active short-term goals.")
    open_loops:   List[str] = Field(
        default_factory=list,
        description="Thoughts, anxieties, or unresolved tasks occupying the user's working memory.",
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
    event:    Optional[str] = Field(None, description="A significant event that just occurred, to be saved as an episodic memory.")
    trigger:  Optional[str] = Field(None, description="Trigger for the new episodic event.")
    response: Optional[str] = Field(None, description="How the user or agent responded to the event.")
    outcome:  Optional[str] = Field(None, description="Outcome of the episodic event.")


# ---------------------------------------------------------------------------
# AderMemory — top-level memory block
# ---------------------------------------------------------------------------


class AderMemory(MemoryBlock):
    """
    Top-level memory for the Ader cognitive companion.

    Composes all five memory tiers and an affective state model on top of
    MemoryBlock.  The `content` field holds the rendered LLM-ready snapshot
    produced by `render()` — call it before passing this block to the model.

    Usage
    -----
        mem = AderMemory(name="ader_session")
        mem.working.active_goals.append("finish the report")
        mem.affective.stress_level = 0.75
        mem.episodic.append(AderEpisodicMemory(event="panic spiral", trigger="deadline"))
        mem.update(mem.render())   # sync content before LLM call
    """

    memory_type: MemoryType = MemoryType.WORKING  # top-level type

    # Memory tiers
    working:    AderWorkingMemory        = Field(default_factory=AderWorkingMemory)
    semantic:   AderSemanticMemory       = Field(default_factory=AderSemanticMemory)
    procedural: AderProceduralMemory     = Field(default_factory=AderProceduralMemory)
    scratchpad: AderScratchpadMemory     = Field(default_factory=AderScratchpadMemory)
    episodic:   List[AderEpisodicMemory] = Field(default_factory=list)

    # Continuously updated affective model
    affective: AffectiveState = Field(default_factory=AffectiveState)

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def add_episode(self, **kwargs: Any) -> AderEpisodicMemory:
        """Create and record a new episodic memory."""
        episode = AderEpisodicMemory(**kwargs)
        self.episodic.append(episode)
        self._touch()
        return episode

    def recent_episodes(self, n: int = 5) -> List[AderEpisodicMemory]:
        """Return the n most recent episodes."""
        return self.episodic[-n:]

    # ------------------------------------------------------------------
    # Scratchpad helpers
    # ------------------------------------------------------------------

    def reset_scratchpad(self) -> None:
        """Clear the scratchpad at the start of each new turn."""
        self.scratchpad = AderScratchpadMemory()
        self._touch()

    # ------------------------------------------------------------------
    # Synthesis Update Applier
    # ------------------------------------------------------------------

    def apply_update(self, update: AderMemoryStateUpdate, confidence_threshold: float = 0.7) -> None:
        """Applies updates from the LLM synthesis, ignoring low-confidence values."""

        def check_conf(field_name: str) -> bool:
            return update.confidence_scores.get(field_name, 1.0) >= confidence_threshold

        self._update_affective(update, check_conf)
        self._update_working(update, check_conf)
        self._update_semantic(update, check_conf)
        self._update_procedural(update, check_conf)
        self._update_episodic(update, check_conf)

        self._touch()

    def _update_affective(self, update: AderMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["stress_level", "energy_level", "cognitive_load", "social_energy", "emotional_regulation", "executive_function"]:
            if getattr(update, f) is not None and check_conf(f):
                setattr(self.affective, f, getattr(update, f))

    def _update_working(self, update: AderMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["mode", "user_emotional_state"]:
            val = getattr(update, f)
            if val is not None and check_conf(f):
                setattr(self.working, f, val)

        for f in ["active_goals", "open_loops"]:
            val = getattr(update, f)
            if val and check_conf(f):
                setattr(self.working, f, val)

    def _update_semantic(self, update: AderMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["preferences", "triggers"]:
            val = getattr(update, f)
            if val and check_conf(f):
                getattr(self.semantic, f).extend(val)

        for f in ["best_focus_time", "sensitive_to_noise", "prefers_direct_language", "dislikes_open_ended_questions"]:
            val = getattr(update, f)
            if val is not None and check_conf(f):
                setattr(self.semantic, f, val)

    def _update_procedural(self, update: AderMemoryStateUpdate, check_conf: callable) -> None:
        for f in ["successful_interventions", "routines_that_worked", "effective_grouping_strategies", "preferred_planning_structures"]:
            val = getattr(update, f)
            if val and check_conf(f):
                getattr(self.procedural, f).extend(val)

    def _update_episodic(self, update: AderMemoryStateUpdate, check_conf: callable) -> None:
        if update.event and check_conf("event"):
            self.add_episode(
                event=update.event,
                trigger=update.trigger,
                response=update.response,
                outcome=update.outcome,
            )

    # ------------------------------------------------------------------
    # Rendering — serializes structured state into LLM-ready content
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
            "=== ADER MEMORY SNAPSHOT ===",
            "",
            f"[Affective] {self.affective.summary()}",
            "",
            "## Working Memory",
            f"Mode:              {w.mode}",
            f"Emotional State:   {w.user_emotional_state}",
            f"Active Goals:      {', '.join(w.active_goals) or 'none'}",
            f"Open Loops:        {', '.join(w.open_loops) or 'none'}",
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
                ts = ep.timestamp.isoformat() if ep.timestamp else "unknown"
                lines.append(
                    f"  [{ts}] {ep.event or '—'} | trigger: {ep.trigger or '—'} "
                    f"| outcome: {ep.outcome or '—'}"
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
    def from_json(cls, data: str) -> "AderMemory":
        """Restore memory state from a persisted JSON string."""
        return cls.model_validate_json(data)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _clamp(val: float) -> float:
    return max(0.0, min(1.0, val))


AderMemory.model_rebuild()
