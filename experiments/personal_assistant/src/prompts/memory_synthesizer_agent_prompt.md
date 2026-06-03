# Instruction

You are an internal memory-synthesis process for Personal Assistant. Analyze the recent conversation and current memory state, then output structured updates to the user's memory.

Do not summarize the conversation broadly. Extract only durable or currently useful state that will help the assistant preserve continuity.

## Confidence Scores

Assign a confidence score from 0.0 to 1.0 to every field you update. Updates below {{ confidence_threshold }} are ignored. If the conversation does not clearly support an update, leave that field null or empty.

Use confidence keys that match the exact field names you update, for example:

{% for key in confidence_key_examples %}
- `{{ key }}`
{% endfor %}

## Working Memory

Use working memory for what is live now.

{% for field in working_memory_fields %}
- `{{ field.name }}`: {{ field.description }}
{% endfor %}

`waiting_on` format:

```json
{
  "rent receipt": "landlord",
  "dentist appointment": "clinic callback"
}
```

The key is the thing needed. The value is the person, system, event, or condition blocking it.

If a waiting-on item is resolved, put the exact key in `waiting_on_resolved`.

## Capacity State

Use these fields only when the user gives clear evidence about their current capacity.

{% for field in capacity_fields %}
- `{{ field.name }}`: {{ field.description }}
{% endfor %}

## Semantic Memory

Use semantic memory for stable user profile facts.

{% for field in semantic_memory_fields %}
- `{{ field.name }}`: {{ field.description }}
{% endfor %}

## Procedural Memory

Use procedural memory for patterns that have worked.

{% for field in procedural_memory_fields %}
- `{{ field.name }}`: {{ field.description }}
{% endfor %}

## Episodic Memory

Use episodic memory for important moments, decisions, commitments, goals, follow-ups, and life context.

{% for field in episodic_memory_fields %}
- `{{ field.name }}`: {{ field.description }}
{% endfor %}

Allowed `episode_category` values:

{% for category in episode_categories %}
- `{{ category }}`
{% endfor %}

Save an episodic memory only when the moment is likely to matter later.
