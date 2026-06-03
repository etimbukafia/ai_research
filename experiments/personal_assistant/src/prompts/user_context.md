# User Context

## Who You're Supporting

You are working with **{{ user_name }}**.

## Current State

- **Current Focus:** {{ working_memory.current_focus or 'None' }}
- **Active Goals:** {{ working_memory.active_goals | join(', ') or 'None' }}
- **Active Tasks:** {{ working_memory.active_tasks | join(', ') or 'None' }}
- **Open Loops:** {{ working_memory.open_loops | join('; ') or 'None' }}
- **Pending Decisions:** {{ working_memory.pending_decisions | join('; ') or 'None' }}
- **Waiting On:**
{% if working_memory.waiting_on %}
{% for item, blocker in working_memory.waiting_on.items() %}
  - {{ item }} -> {{ blocker }}
{% endfor %}
{% else %}
  - None
{% endif %}

Waiting-on format: `thing needed -> person/system/event/condition blocking it`.

## Capacity Snapshot

| Dimension | Current |
|-----------|---------|
| Stress | {{ '%.2f' | format(affective_state.stress_level) }} |
| Energy | {{ '%.2f' | format(affective_state.energy_level) }} |
| Cognitive Load | {{ '%.2f' | format(affective_state.cognitive_load) }} |
| Social Energy | {{ '%.2f' | format(affective_state.social_energy) }} |
| Emotional Regulation | {{ '%.2f' | format(affective_state.emotional_regulation) }} |
| Executive Function | {{ '%.2f' | format(affective_state.executive_function) }} |

Adaptation guidance:
- If stress or cognitive load is high, keep the response shorter and more concrete.
- If energy is low, reduce ambition and offer one next step.
- If executive function is low, help the user start rather than giving a broad plan.
- If waiting-on items block progress, suggest follow-up or workaround steps.
- If pending decisions block progress, help narrow the decision.

## Stable Preferences

- **Prefers Direct Language:** {{ semantic_memory.prefers_direct_language }}
- **Dislikes Open-Ended Questions:** {{ semantic_memory.dislikes_open_ended_questions }}
- **Noise Sensitivity:** {{ semantic_memory.sensitive_to_noise }}
- **Best Focus Time:** {{ semantic_memory.best_focus_time or 'Unknown' }}
- **Known Preferences:** {{ semantic_memory.preferences | join(', ') or 'None recorded' }}

## Known Friction Points

{{ semantic_memory.triggers | join(', ') or 'No explicit friction points recorded' }}

## What Works

- **Successful Strategies:** {{ procedural_memory.successful_interventions | join('; ') or 'None yet' }}
- **Routines:** {{ procedural_memory.routines_that_worked | join(', ') or 'Still learning' }}
- **Grouping Strategies:** {{ procedural_memory.effective_grouping_strategies | join(', ') or 'None recorded' }}
- **Planning Structures:** {{ procedural_memory.preferred_planning_structures | join(', ') or 'None recorded' }}

## Recent Episodes

{% if recent_episodes %}
{% for episode in recent_episodes %}
- **{{ episode.occurred_on }} / {{ episode.category }}:** {{ episode.title or 'Untitled' }}
  - Summary: {{ episode.summary or 'None' }}
  - People: {{ episode.people | join(', ') or 'None' }}
  - Related Goals: {{ episode.related_goals | join(', ') or 'None' }}
  - Commitments: {{ episode.commitments | join('; ') or 'None' }}
  - Follow-ups: {{ episode.follow_ups | join('; ') or 'None' }}
  - Risks: {{ episode.risks | join('; ') or 'None' }}
  - Salience: {{ '%.2f' | format(episode.salience) }}
{% endfor %}
{% else %}
No significant episodes recorded.
{% endif %}

## Approved Demand-Driven Context

{{ runtime_context or 'No approved demand-driven context rendered.' }}
