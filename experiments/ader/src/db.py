"""
Create a pydantic mock db with these five tables to populate with data for testing Ader

## Tables

### User Table:

- UserID: UUID
- Name: string

### Working Memory Table:

- UserID Reference
- Last Ader Mode
- Last User Emotional State
- Active Goals: list of strings
- Active Goals Completed: dictionary mapping goal to boolean (true/false)
- Open Loops: list of strings
- Open Loops Completed: dictionary mapping open loop to boolean (true/false)

### Episodic Memory Table:

- UserID Reference
- User last episode (event): string (a description of what happened)
- Trigger of user last episode: string (what triggered the episode)
- User's response to last episode: string (how the user responded)
- Outcome of user last episode: string (what happened as a result)
- Timestamp: datetime (when the episode occurred)

### Semantic Memory Table:

- UserID Reference
- User preferences: list of strings (interests/likes/dislikes/exclusions)
- User triggers: list of strings (what the user is triggered by)
- User prefers_direct_language: boolean (whether the user prefers direct language)
- User dislikes_open_ended_questions: boolean (whether the user dislikes open ended questions)
- User best_focus_time: list of strings (morning/afternoon/night)
- User sensitive_to_noise: boolean (whether the user is sensitive to noise)

### Procedural Memory Table

- UserID Reference
- successful_interventions: Dict[reference to episode, description of successful intervention(string)]
- rountines_that_worked: List[string]
- effective_grouping_strategies: List[string]
- preferred_planning_structures: List[string]

### Affective State Table:

- UserID Reference
- user_last_stress_level: float (0.0 - 1.0) (0 being calm, 1 being panic)
- user_last_energy_level: float (0.0 - 1.0) (0 being exhausted, 1 being hyper)
- user_last_cognitive_load: float (0.0 - 1.0) (0 being not much information to hold, 1 being overwhelmed)
- user_last_social_energy: float (0.0 - 1.0) (0 being exhausted, 1 being hyper)
- user_last_emotional_regulation: float (0.0 - 1.0) (0 being meltdown, 1 being stable)
- user_last_executive_function: float (0.0 - 1.0) (0 being unable to start/switch tasks, 1 being able to start/switch tasks)

"""

from uuid import UUID, uuid4
from datetime import datetime, date, time
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

# Import Ader Memory models for the helper functions
from experiments.ader.src.memory import (
    AderMemory,
    AderWorkingMemory,
    AderEpisodicMemory,
    AderSemanticMemory,
    AderProceduralMemory,
    AffectiveState
)


# ---------------------------------------------------------------------------
# Database Table Row Schema Models
# ---------------------------------------------------------------------------

class UserRow(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    name: str
    last_session_id: Optional[UUID] = None


class WorkingMemoryRow(BaseModel):
    user_id: UUID
    mode: Literal["calm", "organized", "planning", "reflective", "low_stimulation"] = "calm"
    user_emotional_state: str = "neutral"
    active_goals: List[str] = Field(default_factory=list)
    active_goals_completed: Dict[str, bool] = Field(default_factory=dict)
    open_loops: List[str] = Field(default_factory=list)
    open_loops_completed: Dict[str, bool] = Field(default_factory=dict)
    last_session_id: Optional[UUID] = None


class EpisodicMemoryRow(BaseModel):
    episode_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    last_session_id: Optional[UUID] = None
    event: str
    trigger: str
    response: str
    outcome: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SemanticMemoryRow(BaseModel):
    user_id: UUID
    preferences: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    prefers_direct_language: bool = True
    dislikes_open_ended_questions: bool = True
    best_focus_time: List[str] = Field(default_factory=list)  # List of strings e.g. ["morning", "afternoon", "night"]
    sensitive_to_noise: bool = True
    last_session_id: Optional[UUID] = None


class ProceduralMemoryRow(BaseModel):
    user_id: UUID
    successful_interventions: Dict[UUID, str] = Field(default_factory=dict)  # Maps episode_id (UUID) -> description
    routines_that_worked: List[str] = Field(default_factory=list)
    effective_grouping_strategies: List[str] = Field(default_factory=list)
    preferred_planning_structures: List[str] = Field(default_factory=list)
    last_session_id: Optional[UUID] = None


class AffectiveStateRow(BaseModel):
    user_id: UUID
    user_last_stress_level: float = Field(default=0.5, ge=0.0, le=1.0)
    user_last_energy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    user_last_cognitive_load: float = Field(default=0.5, ge=0.0, le=1.0)
    user_last_social_energy: float = Field(default=0.5, ge=0.0, le=1.0)
    user_last_emotional_regulation: float = Field(default=0.5, ge=0.0, le=1.0)
    user_last_executive_function: float = Field(default=0.5, ge=0.0, le=1.0)
    last_session_id: Optional[UUID] = None


# ---------------------------------------------------------------------------
# Mock Database Container
# ---------------------------------------------------------------------------

class MockDatabase(BaseModel):
    users: List[UserRow] = Field(default_factory=list)
    working_memories: List[WorkingMemoryRow] = Field(default_factory=list)
    episodic_memories: List[EpisodicMemoryRow] = Field(default_factory=list)
    semantic_memories: List[SemanticMemoryRow] = Field(default_factory=list)
    procedural_memories: List[ProceduralMemoryRow] = Field(default_factory=list)
    affective_states: List[AffectiveStateRow] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Helper Methods: Stitching to/from AderMemory
    # -----------------------------------------------------------------------

    def get_ader_memory(self, user_id: UUID) -> AderMemory:
        """Stitch together the database table rows for a user into a unified AderMemory object."""
        # Verify user exists
        user_exists = any(u.user_id == user_id for u in self.users)
        if not user_exists:
            raise ValueError(f"User with ID {user_id} not found in database.")

        # Get or build Working Memory Row
        working_row = next((r for r in self.working_memories if r.user_id == user_id), None)
        if not working_row:
                working_row = WorkingMemoryRow(user_id=user_id)
                # default working last_session to user's last_session if present
                user_row = next((u for u in self.users if u.user_id == user_id), None)
                if user_row and user_row.last_session_id:
                    working_row.last_session_id = user_row.last_session_id

        # Get or build Semantic Memory Row
        semantic_row = next((r for r in self.semantic_memories if r.user_id == user_id), None)
        if not semantic_row:
            semantic_row = SemanticMemoryRow(user_id=user_id)
            user_row = next((u for u in self.users if u.user_id == user_id), None)
            if user_row and user_row.last_session_id:
                semantic_row.last_session_id = user_row.last_session_id

        # Get or build Procedural Memory Row
        procedural_row = next((r for r in self.procedural_memories if r.user_id == user_id), None)
        if not procedural_row:
            procedural_row = ProceduralMemoryRow(user_id=user_id)
            user_row = next((u for u in self.users if u.user_id == user_id), None)
            if user_row and user_row.last_session_id:
                procedural_row.last_session_id = user_row.last_session_id

        # Get or build Affective State Row
        affective_row = next((r for r in self.affective_states if r.user_id == user_id), None)
        if not affective_row:
            affective_row = AffectiveStateRow(user_id=user_id)
            user_row = next((u for u in self.users if u.user_id == user_id), None)
            if user_row and user_row.last_session_id:
                affective_row.last_session_id = user_row.last_session_id

        # Retrieve all episodic records for this user
        user_episodes = [r for r in self.episodic_memories if r.user_id == user_id]

        # 1. Map Working Memory
        working = AderWorkingMemory(
            mode=working_row.mode,
            user_emotional_state=working_row.user_emotional_state,
            active_goals=working_row.active_goals,
            active_goals_completed=working_row.active_goals_completed,
            open_loops=working_row.open_loops,
            open_loops_completed=working_row.open_loops_completed,
            last_session_id=working_row.last_session_id
        )

        # 2. Map Semantic Memory (Resolves best_focus_time list to Literal expected by AderSemanticMemory)
        best_focus = "night"
        if semantic_row.best_focus_time:
            for val in semantic_row.best_focus_time:
                if val in ("morning", "afternoon", "night"):
                    best_focus = val
                    break

        semantic = AderSemanticMemory(
            preferences=semantic_row.preferences,
            triggers=semantic_row.triggers,
            prefers_direct_language=semantic_row.prefers_direct_language,
            dislikes_open_ended_questions=semantic_row.dislikes_open_ended_questions,
            best_focus_time=best_focus,
            sensitive_to_noise=semantic_row.sensitive_to_noise
        )

        # 3. Map Procedural Memory (Reformats dictionary keys to descriptions containing ref ID)
        interventions = []
        for ep_id, desc in procedural_row.successful_interventions.items():
            interventions.append(f"{desc} (ref: {ep_id})")

        procedural = AderProceduralMemory(
            successful_interventions=interventions,
            routines_that_worked=procedural_row.routines_that_worked,
            effective_grouping_strategies=procedural_row.effective_grouping_strategies,
            preferred_planning_structures=procedural_row.preferred_planning_structures
        )

        # 4. Map Episodic Memory (datetime -> date conversion)
        episodic = []
        for ep_row in user_episodes:
            ep_date = ep_row.timestamp.date() if ep_row.timestamp else date.today()
            episodic.append(AderEpisodicMemory(
                event=ep_row.event,
                trigger=ep_row.trigger,
                response=ep_row.response,
                outcome=ep_row.outcome,
                timestamp=ep_date
            ))

        # 5. Map Affective State
        affective = AffectiveState(
            stress_level=affective_row.user_last_stress_level,
            energy_level=affective_row.user_last_energy_level,
            cognitive_load=affective_row.user_last_cognitive_load,
            social_energy=affective_row.user_last_social_energy,
            emotional_regulation=affective_row.user_last_emotional_regulation,
            executive_function=affective_row.user_last_executive_function
        )

        mem = AderMemory(
            name=str(user_id),
            working=working,
            semantic=semantic,
            procedural=procedural,
            episodic=episodic,
            affective=affective
        )
        mem.update(mem.render())
        return mem

    def save_ader_memory(self, user_id: UUID, memory: AderMemory) -> None:
        """Deconstruct AderMemory into database table rows and save them."""
        # Verify user exists
        user_exists = any(u.user_id == user_id for u in self.users)
        if not user_exists:
            raise ValueError(f"User with ID {user_id} not found in database.")

        # 1. Update Working Memory
        working_row = next((r for r in self.working_memories if r.user_id == user_id), None)
        if not working_row:
            working_row = WorkingMemoryRow(user_id=user_id)
            self.working_memories.append(working_row)
        working_row.mode = memory.working.mode
        working_row.user_emotional_state = memory.working.user_emotional_state
        working_row.active_goals = memory.working.active_goals
        working_row.active_goals_completed = memory.working.active_goals_completed
        working_row.open_loops = memory.working.open_loops
        working_row.open_loops_completed = memory.working.open_loops_completed
        working_row.last_session_id = memory.working.last_session_id

        # Also persist to the canonical user row
        user_row = next((u for u in self.users if u.user_id == user_id), None)
        if user_row:
            user_row.last_session_id = memory.working.last_session_id

        # 2. Update Semantic Memory
        semantic_row = next((r for r in self.semantic_memories if r.user_id == user_id), None)
        if not semantic_row:
            semantic_row = SemanticMemoryRow(user_id=user_id)
            self.semantic_memories.append(semantic_row)
        semantic_row.preferences = memory.semantic.preferences
        semantic_row.triggers = memory.semantic.triggers
        semantic_row.prefers_direct_language = memory.semantic.prefers_direct_language
        semantic_row.dislikes_open_ended_questions = memory.semantic.dislikes_open_ended_questions
        semantic_row.best_focus_time = [memory.semantic.best_focus_time] if memory.semantic.best_focus_time else ["night"]
        semantic_row.sensitive_to_noise = memory.semantic.sensitive_to_noise
        semantic_row.last_session_id = memory.working.last_session_id

        # 3. Update Affective State
        affective_row = next((r for r in self.affective_states if r.user_id == user_id), None)
        if not affective_row:
            affective_row = AffectiveStateRow(user_id=user_id)
            self.affective_states.append(affective_row)
        affective_row.user_last_stress_level = memory.affective.stress_level
        affective_row.user_last_energy_level = memory.affective.energy_level
        affective_row.user_last_cognitive_load = memory.affective.cognitive_load
        affective_row.user_last_social_energy = memory.affective.social_energy
        affective_row.user_last_emotional_regulation = memory.affective.emotional_regulation
        affective_row.user_last_executive_function = memory.affective.executive_function
        affective_row.last_session_id = memory.working.last_session_id

        # 4. Update Episodic Memory
        old_episodes = [r for r in self.episodic_memories if r.user_id == user_id]
        self.episodic_memories = [r for r in self.episodic_memories if r.user_id != user_id]
        
        saved_episodes: List[EpisodicMemoryRow] = []
        for ep in memory.episodic:
            dt = datetime.combine(ep.timestamp or date.today(), time.min)
            
            # Attempt to preserve the original episode_id and session metadata
            matched_id = uuid4()
            matched_session_id = memory.working.last_session_id
            for old_ep in old_episodes:
                if old_ep.event == ep.event and old_ep.trigger == ep.trigger:
                    matched_id = old_ep.episode_id
                    matched_session_id = old_ep.last_session_id
                    break

            ep_row = EpisodicMemoryRow(
                episode_id=matched_id,
                user_id=user_id,
                last_session_id=matched_session_id,
                event=ep.event or "",
                trigger=ep.trigger or "",
                response=ep.response or "",
                outcome=ep.outcome or "",
                timestamp=dt
            )
            self.episodic_memories.append(ep_row)
            saved_episodes.append(ep_row)

        # 5. Update Procedural Memory
        procedural_row = next((r for r in self.procedural_memories if r.user_id == user_id), None)
        if not procedural_row:
            procedural_row = ProceduralMemoryRow(user_id=user_id)
            self.procedural_memories.append(procedural_row)

        # Rebuild dictionary mapping episode reference to description
        existing_interventions = dict(procedural_row.successful_interventions)
        new_interventions = {}
        for intervention_desc in memory.procedural.successful_interventions:
            matched_uuid = None
            clean_desc = intervention_desc

            # Attempt to parse existing UUID from string formatted as "desc (ref: UUID)"
            if " (ref: " in intervention_desc:
                try:
                    parts = intervention_desc.rsplit(" (ref: ", 1)
                    clean_desc = parts[0]
                    ref_uuid = UUID(parts[1].rstrip(")"))
                    matched_uuid = ref_uuid
                except ValueError:
                    pass

            if not matched_uuid:
                # Fallback: check if description matches any of our existing keys
                for uid, desc in existing_interventions.items():
                    if desc == clean_desc:
                        matched_uuid = uid
                        break

            if not matched_uuid:
                # If it's a new description, map it to a newly generated UUID
                matched_uuid = uuid4()

            new_interventions[matched_uuid] = clean_desc

        procedural_row.successful_interventions = new_interventions
        procedural_row.routines_that_worked = memory.procedural.routines_that_worked
        procedural_row.effective_grouping_strategies = memory.procedural.effective_grouping_strategies
        procedural_row.preferred_planning_structures = memory.procedural.preferred_planning_structures
        procedural_row.last_session_id = memory.working.last_session_id


# ---------------------------------------------------------------------------
# Pre-populated Test Data Factory
# ---------------------------------------------------------------------------

def create_default_db() -> MockDatabase:
    """Creates a MockDatabase pre-populated with profiles for testing."""
    # Define User IDs
    alex_id = UUID("d3b07384-d113-4956-a5e2-4c5b3648a301")
    taylor_id = UUID("e6c98522-83b6-4bfe-bb4f-b3a1a6b0cfa0")

    # Example session IDs for auditing metadata
    alex_session_id = UUID("11111111-1111-1111-1111-111111111111")
    taylor_session_id = UUID("22222222-2222-2222-2222-222222222222")

    # 1. Users
    users = [
        UserRow(user_id=alex_id, name="Alex", last_session_id=alex_session_id),
        UserRow(user_id=taylor_id, name="Taylor", last_session_id=taylor_session_id)
    ]

    # 2. Working Memories
    working_memories = [
        WorkingMemoryRow(
            user_id=alex_id,
            mode="organized",
            user_emotional_state="focused",
            active_goals=["finish science project", "buy groceries"],
            active_goals_completed={"finish science project": False, "buy groceries": True},
            open_loops=["worried about tomorrow's weather change"],
            open_loops_completed={"worried about tomorrow's weather change": False},
            last_session_id=alex_session_id
        ),
        WorkingMemoryRow(
            user_id=taylor_id,
            mode="low_stimulation",
            user_emotional_state="overwhelmed",
            active_goals=["clean desk", "respond to emails"],
            active_goals_completed={"clean desk": False, "respond to emails": False},
            open_loops=["feeling behind on housework", "unread notifications causing anxiety"],
            open_loops_completed={"feeling behind on housework": False, "unread notifications causing anxiety": False},
            last_session_id=taylor_session_id
        )
    ]

    # 3. Episodic Memories
    alex_episode_id = UUID("fbc4b2da-36ee-4f99-92c2-84a1e9cbf69e")
    taylor_episode_id = UUID("a118df2c-4972-4686-b48c-9c765ef3dfc8")

    episodic_memories = [
        EpisodicMemoryRow(
            episode_id=alex_episode_id,
            user_id=alex_id,
            last_session_id=alex_session_id,
            event="missed morning group standup meeting",
            trigger="forgot to set the alarm clock",
            response="felt highly anxious, contacted coordinator directly for notes",
            outcome="submitted coordinator notes late but successfully obtained an extension",
            timestamp=datetime(2026, 5, 27, 8, 30, 0)
        ),
        EpisodicMemoryRow(
            episode_id=taylor_episode_id,
            user_id=taylor_id,
            last_session_id=taylor_session_id,
            event="social overload during team review",
            trigger="multiple team members talking simultaneously over noise",
            response="withdrew completely from speaking, turned video off",
            outcome="felt drained for hours afterwards, unable to switch back to coding",
            timestamp=datetime(2026, 5, 26, 14, 15, 0)
        )
    ]

    # 4. Semantic Memories
    semantic_memories = [
        SemanticMemoryRow(
            user_id=alex_id,
            preferences=["dark mode templates", "bullet points plan formats", "written recaps"],
            triggers=["loud sirens", "sudden schedule changes", "flashing alerts"],
            prefers_direct_language=True,
            dislikes_open_ended_questions=False,
            best_focus_time=["night", "afternoon"],
            sensitive_to_noise=True,
            last_session_id=alex_session_id
        ),
        SemanticMemoryRow(
            user_id=taylor_id,
            preferences=["silent visual cues", "step-by-step guidance", "plenty of white space"],
            triggers=["multiple sensory inputs", "consecutive back-to-back meetings"],
            prefers_direct_language=True,
            dislikes_open_ended_questions=True,
            best_focus_time=["morning"],
            sensitive_to_noise=True,
            last_session_id=taylor_session_id
        )
    ]

    # 5. Procedural Memories
    procedural_memories = [
        ProceduralMemoryRow(
            user_id=alex_id,
            successful_interventions={
                alex_episode_id: "Prompted to use visual schedule blocks with soft tone alerts."
            },
            routines_that_worked=["night time planning check-ins", "daily tasks grouping"],
            effective_grouping_strategies=["categorize by estimated effort level"],
            preferred_planning_structures=["hierarchical lists"],
            last_session_id=alex_session_id
        ),
        ProceduralMemoryRow(
            user_id=taylor_id,
            successful_interventions={
                taylor_episode_id: "Guided through a prompt-based quiet breathing routine and disabled chat sound."
            },
            routines_that_worked=["immediate sensory breaks", "morning single-focus block"],
            effective_grouping_strategies=["one high-priority task per day only"],
            preferred_planning_structures=["single progress bar step visualizer"],
            last_session_id=taylor_session_id
        )
    ]

    # 6. Affective States
    affective_states = [
        AffectiveStateRow(
            user_id=alex_id,
            user_last_stress_level=0.3,
            user_last_energy_level=0.75,
            user_last_cognitive_load=0.35,
            user_last_social_energy=0.8,
            user_last_emotional_regulation=0.9,
            user_last_executive_function=0.85,
            last_session_id=alex_session_id
        ),
        AffectiveStateRow(
            user_id=taylor_id,
            user_last_stress_level=0.85,
            user_last_energy_level=0.2,
            user_last_cognitive_load=0.9,
            user_last_social_energy=0.15,
            user_last_emotional_regulation=0.35,
            user_last_executive_function=0.25,
            last_session_id=taylor_session_id
        )
    ]

    return MockDatabase(
        users=users,
        working_memories=working_memories,
        episodic_memories=episodic_memories,
        semantic_memories=semantic_memories,
        procedural_memories=procedural_memories,
        affective_states=affective_states
    )
