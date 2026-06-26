from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .async_utils import run_async
from .db import DEFAULT_PROFILE_ID, JournalDatabase
from .models import Entity, Episode, SemanticFact, SemanticFactHint, Thought, utc_now_iso


def export_memory_visualization(
    db: JournalDatabase,
    *,
    profile_id: str = DEFAULT_PROFILE_ID,
    out: str | Path,
    memory: Any | None = None,
    include_mem0: bool = False,
    demo: bool = False,
    seed_mem0: bool = False,
    mem0_error: str | None = None,
) -> Path:
    """Export a standalone HTML visualization of local journal memory and mem0 recall.

    SQLite remains the source for exact local lineage. mem0 is queried as the
    durable recall layer so the screenshot can show both sides of the design.
    """

    db.ensure_profile(profile_id)
    seeded = seed_demo_memory(db, profile_id=profile_id) if demo and not db.list_thoughts(profile_id, limit=1) else []

    mem0_status = "not requested"
    mem0_results: list[dict[str, Any]] = []
    mem0_seeded: list[str] = []
    should_load_mem0 = include_mem0 or seed_mem0
    if should_load_mem0 and mem0_error:
        mem0_status = f"mem0 unavailable: {mem0_error}"
    elif should_load_mem0 and memory is None:
        mem0_status = "mem0 unavailable: no mem0 client"
    elif should_load_mem0 and memory is not None:
        mem0_status, mem0_results, mem0_seeded = _load_mem0_preview(
            memory,
            profile_id=profile_id,
            seed_facts=_demo_semantic_facts(profile_id) if seed_mem0 else [],
        )

    data = _build_visualization_data(
        db,
        profile_id=profile_id,
        mem0_results=mem0_results,
        mem0_status=mem0_status,
        mem0_seeded=mem0_seeded,
        seeded=seeded,
    )

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(data), encoding="utf-8")
    return out_path


def seed_demo_memory(db: JournalDatabase, *, profile_id: str = DEFAULT_PROFILE_ID) -> list[str]:
    """Create a varied journal memory graph for screenshots and demos."""

    janet = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Janet",
            type="person",
            description="Person who asks for quick follow-ups and meetings.",
            aliases=["janet"],
            confidence_score=0.92,
        )
    )
    brussels = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Brussels",
            type="place",
            description="Travel destination that appears in commitments and planning notes.",
            aliases=["brussels"],
            confidence_score=0.84,
        )
    )
    article = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Memory Article",
            type="project",
            description="Article draft explaining the CLI journal memory architecture.",
            aliases=["article", "memory article"],
            confidence_score=0.88,
        )
    )
    journal = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="CLI Journal",
            type="project",
            description="Personal thought capture and memory tool.",
            aliases=["journal"],
            confidence_score=0.82,
        )
    )
    checkout = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Checkout Bug",
            type="project",
            description="Bug investigation that needs logs, reproduction notes, and a filed issue.",
            aliases=["checkout", "checkout bug"],
            confidence_score=0.8,
        )
    )
    qdrant = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Qdrant",
            type="tool",
            description="Vector database considered for local memory storage.",
            aliases=["qdrant"],
            confidence_score=0.72,
        )
    )
    chroma = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="ChromaDB",
            type="tool",
            description="Local priming and visualization source for familiarity search.",
            aliases=["chroma", "chromadb"],
            confidence_score=0.78,
        )
    )
    health = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Health Routine",
            type="concept",
            description="Recurring personal health and energy habits.",
            aliases=["health"],
            confidence_score=0.76,
        )
    )
    sleep = db.upsert_entity(
        Entity(
            profile_id=profile_id,
            canonical_name="Sleep Routine",
            type="concept",
            description="Recurring sleep and energy pattern mentioned across journal entries.",
            aliases=["sleep"],
            confidence_score=0.74,
        )
    )

    samples = [
        (
            "@janet wants to see me by 2",
            "commitment",
            ["people", "meeting", "follow_up", "today"],
            [janet.entity_id],
            "This is a time-sensitive commitment involving another person.",
            0.78,
        ),
        (
            "Need to go to Brussels tomorrow and check train options before dinner",
            "commitment",
            ["travel", "planning", "tomorrow", "logistics"],
            [brussels.entity_id],
            "A near-term travel plan that may need reminders and follow-through.",
            0.82,
        ),
        (
            "The article needs a clearer explanation of SQLite, ChromaDB, and mem0",
            "work",
            ["article", "architecture", "memory_layers", "explainers"],
            [article.entity_id, journal.entity_id, chroma.entity_id],
            "This affects whether the article explains the system accurately.",
            0.84,
        ),
        (
            "Add a knowledge graph view to the memory visualization",
            "task",
            ["visualization", "knowledge_graph", "memory_map", "ux"],
            [journal.entity_id, article.entity_id],
            "This is an actionable improvement for showing how memories connect.",
            0.81,
        ),
        (
            "Drag nodes should pin in place so the graph can be rearranged while thinking",
            "decision",
            ["visualization", "interaction", "ux", "decision"],
            [journal.entity_id],
            "The visualizer should support active sensemaking, not only passive viewing.",
            0.77,
        ),
        (
            "Checkout bug logs need to be checked before filing the issue",
            "task",
            ["debugging", "checkout", "logs", "issue"],
            [checkout.entity_id],
            "A task with a clear prerequisite before a public bug report.",
            0.8,
        ),
        (
            "If mem0 is unavailable, capture should still save locally and log the warning",
            "decision",
            ["reliability", "mem0", "logging", "offline_first"],
            [journal.entity_id],
            "This preserves capture even when an external memory layer fails.",
            0.86,
        ),
        (
            "Do not let one-off notes become permanent facts too quickly",
            "risk",
            ["consolidation", "semantic_memory", "confidence", "risk"],
            [journal.entity_id],
            "This is a design constraint for semantic memory promotion.",
            0.79,
        ),
        (
            "Qdrant collections from older versions should be easy to reset when empty",
            "work",
            ["qdrant", "migration", "hybrid_search", "storage_reset"],
            [qdrant.entity_id, journal.entity_id],
            "This captures a storage lesson about vector database versions.",
            0.69,
        ),
        (
            "The CLI journal should feel useful for non-developers too",
            "idea",
            ["positioning", "article", "audience", "product"],
            [article.entity_id, journal.entity_id],
            "This affects how the project is described publicly.",
            0.75,
        ),
        (
            "Walking after lunch seems to make the afternoon less sluggish",
            "health",
            ["energy", "walking", "routine", "health"],
            [health.entity_id],
            "This may become a reusable health pattern if repeated.",
            0.73,
        ),
        (
            "Second day in a row that a late screen session made sleep worse",
            "health",
            ["sleep", "energy", "routine", "pattern"],
            [sleep.entity_id, health.entity_id],
            "This is possible repeated evidence for a sleep-related semantic fact.",
            0.71,
        ),
        (
            "Group related article notes by decisions, commitments, and open questions",
            "idea",
            ["article", "knowledge_graph", "taxonomy", "writing"],
            [article.entity_id],
            "The article can mirror the actual memory structure instead of listing features.",
            0.74,
        ),
        (
            "@janet prefers quick context before I ask for a decision",
            "work",
            ["people", "communication", "preference", "janet"],
            [janet.entity_id],
            "This may become a relationship or communication preference after confirmation.",
            0.7,
        ),
    ]

    created: list[str] = []
    episodes_by_key: dict[str, str] = {}
    for body, thought_type, tags, entity_refs, significance, salience in samples:
        thought = db.add_thought(
            Thought(
                profile_id=profile_id,
                body=body,
                thought_type=thought_type,
                thought="article demo",
                tags=tags,
                entity_refs=entity_refs,
            )
        )
        episode = db.add_episode(
            Episode(
                profile_id=profile_id,
                description=body,
                event_type=thought_type,
                significance=significance,
                thought_id=thought.thought_id,
                thought=thought.thought,
                tags=tags,
                entity_refs=entity_refs,
                salience_score=salience,
            )
        )
        created.extend([thought.thought_id, episode.episode_id])
        episodes_by_key[body] = episode.episode_id

    hints = [
        SemanticFactHint(
            profile_id=profile_id,
            subject_entity_id=journal.entity_id,
            predicate="constraint",
            value="The journal should never lose a captured thought just because mem0, ChromaDB, or an LLM call fails.",
            confidence_score=0.83,
            source_episode_refs=[
                episodes_by_key["If mem0 is unavailable, capture should still save locally and log the warning"],
                episodes_by_key["Do not let one-off notes become permanent facts too quickly"],
            ],
            support_count=2,
            rationale="Several design notes point toward local-first capture and deferred memory writes.",
        ),
        SemanticFactHint(
            profile_id=profile_id,
            subject_entity_id=journal.entity_id,
            predicate="recurring_need",
            value="The memory map needs to separate decisions, commitments, tasks, risks, and semantic facts clearly.",
            confidence_score=0.8,
            source_episode_refs=[
                episodes_by_key["Add a knowledge graph view to the memory visualization"],
                episodes_by_key["Drag nodes should pin in place so the graph can be rearranged while thinking"],
                episodes_by_key["Group related article notes by decisions, commitments, and open questions"],
            ],
            support_count=3,
            rationale="Visualization notes repeatedly ask for richer categories and interactive sensemaking.",
        ),
        SemanticFactHint(
            profile_id=profile_id,
            subject_entity_id=janet.entity_id,
            predicate="relationship",
            value="Janet is associated with fast follow-ups, meetings, and lightweight context before decisions.",
            confidence_score=0.76,
            source_episode_refs=[
                episodes_by_key["@janet wants to see me by 2"],
                episodes_by_key["@janet prefers quick context before I ask for a decision"],
            ],
            support_count=2,
            rationale="Two notes mention Janet in coordination contexts.",
        ),
        SemanticFactHint(
            profile_id=profile_id,
            subject_entity_id=health.entity_id,
            predicate="recurring_need",
            value="Energy notes should track sleep, walking, and afternoon sluggishness as related patterns.",
            confidence_score=0.72,
            source_episode_refs=[
                episodes_by_key["Walking after lunch seems to make the afternoon less sluggish"],
                episodes_by_key["Second day in a row that a late screen session made sleep worse"],
            ],
            support_count=2,
            rationale="The health notes share an energy-management theme.",
        ),
    ]
    for hint in hints:
        db.create_semantic_fact_hint(hint)
        created.append(hint.hint_id)
    return created


def _load_mem0_preview(
    memory: Any,
    *,
    profile_id: str,
    seed_facts: list[SemanticFact],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    seeded: list[str] = []
    try:
        for fact in seed_facts:
            run_async(memory.add_semantic_fact(fact, user_id=profile_id))
            seeded.append(f"{fact.subject_entity_id}.{fact.predicate}")
        results = run_async(
            memory.recall(
                "journal commitments article memory graph health routines",
                user_id=profile_id,
                limit=8,
            )
        )
        return "live mem0 recall", results, seeded
    except Exception as exc:  # pragma: no cover - depends on local mem0 credentials/network.
        return f"mem0 unavailable: {exc}", [], seeded


def _demo_semantic_facts(profile_id: str) -> list[SemanticFact]:
    now = utc_now_iso()
    return [
        SemanticFact(
            profile_id=profile_id,
            subject_entity_id="CLI Journal",
            predicate="recurring_need",
            value="The journal needs clear links between thoughts, episodes, entities, and promoted facts.",
            confidence_score=0.86,
            source_episode_refs=["demo_article_architecture", "demo_memory_visualization", "demo_semantic_memory"],
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
        ),
        SemanticFact(
            profile_id=profile_id,
            subject_entity_id="Janet",
            predicate="relationship",
            value="Janet is tracked as a person entity when mentioned with @janet.",
            confidence_score=0.82,
            source_episode_refs=["demo_janet_meeting"],
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
        ),
    ]


def _build_visualization_data(
    db: JournalDatabase,
    *,
    profile_id: str,
    mem0_results: list[dict[str, Any]],
    mem0_status: str,
    mem0_seeded: list[str],
    seeded: list[str],
) -> dict[str, Any]:
    thoughts = db.list_thoughts(profile_id, limit=100)
    episodes = db.list_episodes(profile_id, limit=100)
    entities = db.list_entities(profile_id, limit=100)
    sessions = db.list_sessions(profile_id, limit=20)
    hints = db.list_semantic_fact_hints(profile_id, status=None, limit=100)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    tag_values: set[str] = set()

    def tag_id(value: str) -> str:
        normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
        normalized = "_".join(part for part in normalized.split("_") if part)
        return f"tag_{normalized or 'untagged'}"

    def add_tag_edges(source_id: str, tags: list[str]) -> None:
        for tag in tags:
            cleaned = tag.strip()
            if not cleaned:
                continue
            tag_values.add(cleaned)
            edges.append({"from": source_id, "to": tag_id(cleaned), "label": "tagged"})

    for entity in entities:
        nodes.append(
            {
                "id": entity.entity_id,
                "kind": "entity",
                "label": entity.canonical_name,
                "detail": entity.description or entity.type,
                "type": entity.type,
                "confidence": entity.confidence_score,
                "aliases": entity.aliases,
                "cluster": entity.type,
            }
        )
    for thought in thoughts:
        nodes.append(
            {
                "id": thought.thought_id,
                "kind": "thought",
                "label": thought.thought or thought.thought_type,
                "detail": thought.body,
                "type": thought.thought_type,
                "tags": thought.tags,
                "created_at": thought.created_at,
                "cluster": thought.thought_type,
            }
        )
        add_tag_edges(thought.thought_id, thought.tags)
        for entity_id in thought.entity_refs:
            edges.append({"from": thought.thought_id, "to": entity_id, "label": "mentions"})
    for episode in episodes:
        nodes.append(
            {
                "id": episode.episode_id,
                "kind": "episode",
                "label": episode.thought or episode.event_type,
                "detail": episode.description,
                "type": episode.event_type,
                "tags": episode.tags,
                "salience": episode.salience_score,
                "occurred_at": episode.occurred_at,
                "significance": episode.significance,
                "cluster": episode.event_type,
            }
        )
        add_tag_edges(episode.episode_id, episode.tags)
        if episode.thought_id:
            edges.append({"from": episode.thought_id, "to": episode.episode_id, "label": "became"})
        for entity_id in episode.entity_refs:
            edges.append({"from": episode.episode_id, "to": entity_id, "label": "about"})
    for hint in hints:
        nodes.append(
            {
                "id": hint.hint_id,
                "kind": "hint",
                "label": f"{hint.predicate} ({hint.support_count})",
                "detail": hint.value,
                "type": hint.predicate,
                "confidence": hint.confidence_score,
                "support_count": hint.support_count,
                "status": hint.status,
                "rationale": hint.rationale,
                "cluster": hint.predicate,
            }
        )
        for episode_id in hint.source_episode_refs:
            edges.append({"from": episode_id, "to": hint.hint_id, "label": "supports"})
        edges.append({"from": hint.hint_id, "to": hint.subject_entity_id, "label": hint.status})
    for tag in sorted(tag_values, key=str.lower):
        nodes.append(
            {
                "id": tag_id(tag),
                "kind": "tag",
                "label": f"#{tag}",
                "detail": f"Cross-cutting journal tag: {tag}",
                "type": "tag",
                "cluster": tag,
            }
        )
    node_ids = {node["id"] for node in nodes}
    for index, item in enumerate(mem0_results, start=1):
        normalized = _normalize_mem0_result(item)
        node_id = f"mem0_{index}"
        nodes.append(
            {
                "id": node_id,
                "kind": "mem0",
                "label": normalized["memory_type"],
                "detail": normalized["text"],
                "score": normalized["score"],
                "cluster": "mem0",
            }
        )
        if normalized["source_id"] in node_ids:
            edges.append({"from": normalized["source_id"], "to": node_id, "label": "recalls"})

    return {
        "generated_at": utc_now_iso(),
        "profile_id": profile_id,
        "counts": {
            "thoughts": len(thoughts),
            "episodes": len(episodes),
            "entities": len(entities),
            "sessions": len(sessions),
            "hints": len(hints),
            "tags": len(tag_values),
            "mem0_results": len(mem0_results),
        },
        "seeded": seeded,
        "nodes": nodes,
        "edges": edges,
        "thoughts": [asdict(item) for item in thoughts[:12]],
        "episodes": [asdict(item) for item in episodes[:12]],
        "entities": [asdict(item) for item in entities[:12]],
        "hints": [asdict(item) for item in hints[:12]],
        "sessions": [asdict(item) for item in sessions[:5]],
        "mem0": {
            "status": mem0_status,
            "seeded": mem0_seeded,
            "results": [_normalize_mem0_result(item) for item in mem0_results],
        },
    }


def _normalize_mem0_result(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    text = item.get("memory") or item.get("text") or item.get("content") or item.get("value") or str(item)
    return {
        "text": str(text),
        "score": item.get("score") or item.get("relevance") or item.get("similarity"),
        "memory_type": metadata.get("memory_type") or item.get("memory_type") or "memory",
        "source_id": metadata.get("source_id") or item.get("id") or "",
    }


def _render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    mem0_cards = "\n".join(_mem0_card(item) for item in data["mem0"]["results"]) or _empty_mem0(data["mem0"]["status"])
    seeded = ", ".join(data["mem0"]["seeded"]) or "none"
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLI Journal Knowledge Graph</title>
  <style>
    :root { --bg:#f4f6f5; --paper:#ffffff; --ink:#17202a; --muted:#64707d; --line:#d6dde2; --entity:#a15c12; --thought:#0f766e; --episode:#315f91; --hint:#7c4aa5; --tag:#69721e; --mem0:#b43e68; --edge:#8a97a3; --active:#111827; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(1460px, calc(100% - 32px)); margin:0 auto; padding:24px 0 38px; }
    header { display:grid; grid-template-columns:1fr auto; gap:24px; align-items:end; padding-bottom:18px; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:2rem; line-height:1.08; letter-spacing:0; }
    h2 { margin:0 0 12px; font-size:1rem; letter-spacing:0; }
    p { margin:0 0 10px; color:var(--muted); line-height:1.5; }
    code { padding:.12em .34em; border:1px solid var(--line); border-radius:5px; background:#f8fafc; }
    .meta { text-align:right; min-width:260px; }
    .counts { display:grid; grid-template-columns:repeat(7, minmax(104px, 1fr)); gap:10px; margin:16px 0; }
    .count { padding:12px 14px; border:1px solid var(--line); border-radius:8px; background:var(--paper); }
    .count strong { display:block; font-size:1.55rem; line-height:1.1; }
    .workspace { display:grid; grid-template-columns:minmax(0, 1fr) 360px; gap:14px; align-items:stretch; }
    section { border:1px solid var(--line); border-radius:8px; background:var(--paper); box-shadow:0 10px 26px rgba(20,31,43,.06); }
    .graph-shell { min-height:760px; overflow:hidden; position:relative; }
    .toolbar { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center; padding:12px; border-bottom:1px solid var(--line); }
    .legend { display:flex; flex-wrap:wrap; gap:8px; }
    .tools { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:flex-end; }
    button, input { border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); padding:7px 10px; font:inherit; }
    button { cursor:pointer; }
    button[aria-pressed="true"] { border-color:var(--active); box-shadow:inset 0 0 0 1px var(--active); }
    input[type="search"] { width:210px; }
    input[type="range"] { width:120px; padding:0; }
    label { display:flex; align-items:center; gap:7px; color:var(--muted); font-size:.86rem; white-space:nowrap; }
    .swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
    #graph { display:block; width:100%; height:702px; background:#fbfcfd; touch-action:none; cursor:grab; }
    #graph.dragging { cursor:grabbing; }
    .cluster-label { fill:#8a97a3; font-size:11px; font-weight:800; text-transform:uppercase; pointer-events:none; }
    .link { stroke:var(--edge); stroke-opacity:.54; stroke-width:1.35; }
    .link.strong { stroke-opacity:.86; stroke-width:2; }
    .link-label { fill:#617080; font-size:10px; pointer-events:none; paint-order:stroke; stroke:#fff; stroke-width:3px; }
    .node circle { stroke:#fff; stroke-width:2.5; filter:drop-shadow(0 4px 8px rgba(15,23,42,.18)); }
    .node text { fill:#1f2937; font-size:11px; font-weight:750; pointer-events:none; paint-order:stroke; stroke:#fff; stroke-width:4px; }
    .node.dim, .link.dim, .link-label.dim { opacity:.14; }
    .node.hidden, .link.hidden, .link-label.hidden { display:none; }
    .node.selected circle { stroke:#111827; stroke-width:3.5; }
    .node.pinned circle { stroke:#0f172a; stroke-dasharray:3 3; }
    .side { padding:14px; min-height:760px; }
    .detail { padding:12px; border:1px solid var(--line); border-radius:8px; background:#f8fafc; }
    .detail strong { display:block; margin-bottom:6px; }
    .detail .pill { display:inline-block; margin-bottom:9px; padding:3px 8px; border-radius:999px; color:#fff; font-size:.72rem; font-weight:800; text-transform:uppercase; }
    .detail-list { margin:8px 0 0; padding:0; list-style:none; }
    .detail-list li { margin:0 0 7px; color:var(--muted); line-height:1.35; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
    .chip { border:1px solid var(--line); border-radius:999px; padding:3px 8px; color:#344054; background:#fff; font-size:.78rem; }
    .mem0-card { padding:11px; border:1px solid rgba(178,58,111,.24); border-radius:8px; background:#fff; margin-bottom:10px; }
    .mem0-card strong { color:var(--mem0); }
    .footer-data { margin-top:14px; padding:14px; }
    details { color:var(--muted); }
    pre { max-height:260px; overflow:auto; padding:14px; border-radius:8px; background:#111827; color:#e8edf3; font-size:.82rem; }
    @media (max-width: 1080px) { header, .workspace { display:block; } .meta { text-align:left; margin-top:14px; } .counts { grid-template-columns:repeat(2, minmax(0, 1fr)); } .toolbar { display:block; } .tools { justify-content:flex-start; margin-top:10px; } .side { min-height:auto; margin-top:14px; } #graph { height:620px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>CLI Journal Knowledge Graph</h1>
        <p>Thoughts, episodes, entities, semantic hints, and mem0 recall rendered as a connected graph.</p>
      </div>
      <div class="meta">
        <p><strong>Profile</strong><br>__PROFILE__</p>
        <p><strong>Generated</strong><br>__GENERATED__</p>
      </div>
    </header>
    <div class="counts">
      __COUNTS__
    </div>
    <div class="workspace">
      <section class="graph-shell">
        <div class="toolbar">
          <div class="legend" id="legend"></div>
          <div class="tools">
            <input type="search" id="search" placeholder="Search nodes">
            <label>Spacing <input type="range" id="spacing" min="90" max="260" value="165"></label>
            <button type="button" id="fit">Fit</button>
            <button type="button" id="release">Release pins</button>
            <button type="button" id="reset">Reset layout</button>
          </div>
        </div>
        <svg id="graph" role="img" aria-label="Knowledge graph of journal memory"></svg>
      </section>
      <section class="side">
        <h2>Selected Node</h2>
        <div class="detail" id="detail">
          <p>Select a node to inspect its source record and connected edges.</p>
        </div>
        <h2 style="margin-top:16px;">mem0 Recall Layer</h2>
        <p>Status: <code>__MEM0_STATUS__</code></p>
        <p>Seeded semantic facts: <code>__SEEDED__</code></p>
        __MEM0_CARDS__
      </section>
    </div>
    <section class="footer-data">
      <h2>Embedded Data</h2>
      <details>
        <summary>Open JSON payload</summary>
        <pre>__JSON_PRE__</pre>
      </details>
    </section>
  </main>
  <script type="application/json" id="memory-data">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById("memory-data").textContent);
    const colors = { entity:"#a15c12", thought:"#0f766e", episode:"#315f91", hint:"#7c4aa5", tag:"#69721e", mem0:"#b43e68" };
    const radii = { entity:17, thought:11, episode:12, hint:15, tag:9, mem0:12 };
    const masses = { entity:1.45, thought:1, episode:1.08, hint:1.18, tag:.78, mem0:1 };
    const svg = document.getElementById("graph");
    const legend = document.getElementById("legend");
    const detail = document.getElementById("detail");
    const search = document.getElementById("search");
    const spacing = document.getElementById("spacing");
    const activeKinds = new Set(Object.keys(colors));
    let selectedId = null;
    let width = 1000;
    let height = 700;
    let nodes = [];
    let edges = [];
    let simulationId = null;
    let dragging = null;
    let panning = null;
    let transform = { x: 0, y: 0, k: 1 };
    let graphLayer = null;

    function init() {
      resize();
      const anchors = clusterAnchors(width, height);
      nodes = data.nodes.map((node, index) => ({
        ...node,
        x: initialX(node, index, anchors),
        y: initialY(node, index, anchors),
        vx: 0,
        vy: 0,
        pinned: false,
        match: true
      }));
      edges = data.edges.map(edge => ({ ...edge }));
      renderLegend();
      draw();
      for (let i = 0; i < 260; i += 1) tick();
      draw();
      startSimulation();
    }

    function resize() {
      const rect = svg.getBoundingClientRect();
      width = Math.max(360, rect.width);
      height = Math.max(460, rect.height);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    }

    function clusterAnchors(w, h) {
      return {
        commitment: [w * .18, h * .22],
        decision: [w * .5, h * .18],
        task: [w * .82, h * .26],
        risk: [w * .75, h * .72],
        health: [w * .22, h * .74],
        work: [w * .5, h * .5],
        idea: [w * .35, h * .34],
        project: [w * .5, h * .36],
        person: [w * .16, h * .45],
        place: [w * .14, h * .62],
        tool: [w * .72, h * .48],
        concept: [w * .36, h * .72],
        tag: [w * .5, h * .83],
        mem0: [w * .86, h * .56],
        default: [w * .5, h * .5]
      };
    }

    function anchorFor(node) {
      const anchors = clusterAnchors(width, height);
      return anchors[node.type] || anchors[node.cluster] || anchors[node.kind] || anchors.default;
    }

    function initialX(node, index, anchors) {
      const anchor = anchors[node.type] || anchors[node.cluster] || anchors[node.kind] || anchors.default;
      return anchor[0] + Math.cos(index * 1.91) * (28 + (index % 7) * 11);
    }

    function initialY(node, index, anchors) {
      const anchor = anchors[node.type] || anchors[node.cluster] || anchors[node.kind] || anchors.default;
      return anchor[1] + Math.sin(index * 1.53) * (28 + (index % 5) * 13);
    }

    function renderLegend() {
      legend.innerHTML = "";
      Object.keys(colors).forEach(kind => {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("aria-pressed", "true");
        button.innerHTML = `<span class="swatch" style="background:${colors[kind]}"></span>${kind}`;
        button.addEventListener("click", () => {
          if (activeKinds.has(kind) && activeKinds.size > 1) activeKinds.delete(kind);
          else activeKinds.add(kind);
          button.setAttribute("aria-pressed", activeKinds.has(kind) ? "true" : "false");
          draw();
        });
        legend.appendChild(button);
      });
    }

    function tick() {
      const byId = new Map(nodes.map(node => [node.id, node]));
      const targetSpacing = Number(spacing.value || 165);
      nodes.forEach((a, i) => {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const b = nodes[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let distance = Math.max(1, Math.hypot(dx, dy));
          const minDistance = (radii[a.kind] || 10) + (radii[b.kind] || 10) + 34;
          const repel = (targetSpacing * targetSpacing * 0.23) / (distance * distance);
          const collide = distance < minDistance ? (minDistance - distance) * 0.035 : 0;
          const force = repel + collide;
          if (!a.pinned) {
            a.vx += (dx / distance) * force / (masses[a.kind] || 1);
            a.vy += (dy / distance) * force / (masses[a.kind] || 1);
          }
          if (!b.pinned) {
            b.vx -= (dx / distance) * force / (masses[b.kind] || 1);
            b.vy -= (dy / distance) * force / (masses[b.kind] || 1);
          }
        }
      });
      edges.forEach(edge => {
        const a = byId.get(edge.from);
        const b = byId.get(edge.to);
        if (!a || !b) return;
        const target = edge.label === "became" ? targetSpacing * .62 : edge.label === "tagged" ? targetSpacing * .72 : targetSpacing;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const force = (distance - target) * 0.012;
        if (!a.pinned) {
          a.vx += (dx / distance) * force;
          a.vy += (dy / distance) * force;
        }
        if (!b.pinned) {
          b.vx -= (dx / distance) * force;
          b.vy -= (dy / distance) * force;
        }
      });
      nodes.forEach(node => {
        if (!node.pinned) {
          const anchor = anchorFor(node);
          node.vx += (anchor[0] - node.x) * 0.0028;
          node.vy += (anchor[1] - node.y) * 0.0028;
          node.vx *= 0.84;
          node.vy *= 0.84;
          node.x = Math.min(width - 42, Math.max(42, node.x + node.vx));
          node.y = Math.min(height - 42, Math.max(42, node.y + node.vy));
        }
      });
    }

    function draw() {
      const byId = new Map(nodes.map(node => [node.id, node]));
      const visibleNode = node => activeKinds.has(node.kind);
      const connected = new Set();
      if (selectedId) {
        edges.forEach(edge => {
          if (edge.from === selectedId || edge.to === selectedId) {
            connected.add(edge.from);
            connected.add(edge.to);
          }
        });
      }
      svg.innerHTML = "";
      graphLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const labelLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const clusterLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      graphLayer.append(clusterLayer, edgeLayer, labelLayer, nodeLayer);
      svg.append(graphLayer);
      applyTransform();

      Object.entries(clusterAnchors(width, height)).forEach(([name, point]) => {
        if (name === "default") return;
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "cluster-label");
        label.setAttribute("x", point[0]);
        label.setAttribute("y", point[1]);
        label.setAttribute("text-anchor", "middle");
        label.textContent = name;
        clusterLayer.appendChild(label);
      });

      edges.forEach(edge => {
        const a = byId.get(edge.from);
        const b = byId.get(edge.to);
        if (!a || !b || !visibleNode(a) || !visibleNode(b)) return;
        const dim = selectedId && !(edge.from === selectedId || edge.to === selectedId);
        const hidden = !a.match || !b.match;
        const strong = edge.label === "supports" || edge.label === "recalls";
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", `link${dim ? " dim" : ""}${hidden ? " hidden" : ""}${strong ? " strong" : ""}`);
        line.setAttribute("x1", a.x);
        line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x);
        line.setAttribute("y2", b.y);
        edgeLayer.appendChild(line);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", `link-label${dim ? " dim" : ""}${hidden ? " hidden" : ""}`);
        label.setAttribute("x", (a.x + b.x) / 2);
        label.setAttribute("y", (a.y + b.y) / 2 - 4);
        label.textContent = edge.label;
        labelLayer.appendChild(label);
      });

      nodes.forEach(node => {
        if (!visibleNode(node)) return;
        const dim = selectedId && selectedId !== node.id && !connected.has(node.id);
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", `node${dim ? " dim" : ""}${selectedId === node.id ? " selected" : ""}${node.pinned ? " pinned" : ""}${node.match ? "" : " hidden"}`);
        group.setAttribute("transform", `translate(${node.x},${node.y})`);
        group.style.cursor = "pointer";
        group.addEventListener("pointerdown", event => startDrag(event, node));
        group.addEventListener("dblclick", event => {
          event.stopPropagation();
          node.pinned = !node.pinned;
          draw();
        });
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("r", radii[node.kind] || 11);
        circle.setAttribute("fill", colors[node.kind] || "#64748b");
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${node.kind}: ${node.label}`;
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", 0);
        text.setAttribute("y", -(radii[node.kind] || 11) - 8);
        text.setAttribute("text-anchor", "middle");
        text.textContent = compactLabel(node.label || node.kind);
        group.append(title, circle, text);
        nodeLayer.appendChild(group);
      });
    }

    function showDetail(node) {
      if (!node) {
        detail.innerHTML = "<p>Select a node to inspect its source record and connected edges.</p>";
        return;
      }
      const related = edges.filter(edge => edge.from === node.id || edge.to === node.id);
      const tags = Array.isArray(node.tags) ? node.tags : [];
      const meta = [
        node.type ? ["Type", node.type] : null,
        node.salience !== undefined ? ["Salience", Number(node.salience).toFixed(2)] : null,
        node.confidence !== undefined ? ["Confidence", Number(node.confidence).toFixed(2)] : null,
        node.support_count !== undefined ? ["Support", node.support_count] : null,
        node.status ? ["Status", node.status] : null,
        node.created_at ? ["Created", node.created_at] : null,
        node.occurred_at ? ["Occurred", node.occurred_at] : null
      ].filter(Boolean);
      detail.innerHTML = `
        <span class="pill" style="background:${colors[node.kind] || "#64748b"}">${escapeHtml(node.kind)}</span>
        <strong>${escapeHtml(node.label)}</strong>
        <p>${escapeHtml(node.detail)}</p>
        ${tags.length ? `<div class="chips">${tags.map(tag => `<span class="chip">#${escapeHtml(tag)}</span>`).join("")}</div>` : ""}
        ${node.significance ? `<p><strong>Significance</strong></p><p>${escapeHtml(node.significance)}</p>` : ""}
        ${node.rationale ? `<p><strong>Rationale</strong></p><p>${escapeHtml(node.rationale)}</p>` : ""}
        <p><code>${escapeHtml(node.id)}</code></p>
        ${meta.length ? `<ul class="detail-list">${meta.map(item => `<li><strong>${escapeHtml(item[0])}</strong> ${escapeHtml(item[1])}</li>`).join("")}</ul>` : ""}
        <p><strong>Edges</strong></p>
        ${related.length ? related.map(edge => `<p><code>${escapeHtml(shortId(edge.from))}</code> ${escapeHtml(edge.label)} <code>${escapeHtml(shortId(edge.to))}</code></p>`).join("") : "<p>No connected edges.</p>"}
      `;
    }

    function startSimulation() {
      if (simulationId) cancelAnimationFrame(simulationId);
      let frames = 0;
      const step = () => {
        for (let i = 0; i < 2; i += 1) tick();
        draw();
        frames += 1;
        if (frames < 220 || dragging) simulationId = requestAnimationFrame(step);
      };
      simulationId = requestAnimationFrame(step);
    }

    function startDrag(event, node) {
      event.preventDefault();
      event.stopPropagation();
      const point = toGraphPoint(event);
      dragging = { node, dx: node.x - point.x, dy: node.y - point.y, moved: false };
      node.pinned = true;
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("dragging");
    }

    function startPan(event) {
      if (event.target.closest && event.target.closest(".node")) return;
      panning = { x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("dragging");
    }

    function pointerMove(event) {
      if (dragging) {
        const point = toGraphPoint(event);
        dragging.node.x = Math.min(width - 30, Math.max(30, point.x + dragging.dx));
        dragging.node.y = Math.min(height - 30, Math.max(30, point.y + dragging.dy));
        dragging.node.vx = 0;
        dragging.node.vy = 0;
        dragging.moved = true;
        draw();
        return;
      }
      if (panning) {
        transform.x = panning.tx + event.clientX - panning.x;
        transform.y = panning.ty + event.clientY - panning.y;
        applyTransform();
      }
    }

    function pointerUp(event) {
      if (dragging) {
        const node = dragging.node;
        const moved = dragging.moved;
        dragging = null;
        svg.classList.remove("dragging");
        if (!moved) {
          selectedId = selectedId === node.id ? null : node.id;
          showDetail(selectedId ? node : null);
        }
        draw();
      }
      if (panning) {
        panning = null;
        svg.classList.remove("dragging");
      }
      try { svg.releasePointerCapture(event.pointerId); } catch {}
    }

    function toGraphPoint(event) {
      const rect = svg.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left - transform.x) / transform.k,
        y: (event.clientY - rect.top - transform.y) / transform.k
      };
    }

    function applyTransform() {
      if (graphLayer) graphLayer.setAttribute("transform", `translate(${transform.x},${transform.y}) scale(${transform.k})`);
    }

    function fitToGraph() {
      const visible = nodes.filter(node => activeKinds.has(node.kind) && node.match);
      if (!visible.length) return;
      const minX = Math.min(...visible.map(node => node.x)) - 60;
      const maxX = Math.max(...visible.map(node => node.x)) + 60;
      const minY = Math.min(...visible.map(node => node.y)) - 60;
      const maxY = Math.max(...visible.map(node => node.y)) + 60;
      const scale = Math.min(1.6, Math.max(.45, Math.min(width / Math.max(1, maxX - minX), height / Math.max(1, maxY - minY))));
      transform.k = scale;
      transform.x = (width - (minX + maxX) * scale) / 2;
      transform.y = (height - (minY + maxY) * scale) / 2;
      applyTransform();
    }

    function updateSearch() {
      const query = search.value.trim().toLowerCase();
      nodes.forEach(node => {
        const haystack = [
          node.id,
          node.kind,
          node.label,
          node.detail,
          node.type,
          ...(node.tags || []),
          ...(node.aliases || [])
        ].join(" ").toLowerCase();
        node.match = !query || haystack.includes(query);
      });
      draw();
    }

    function compactLabel(value) {
      const text = String(value);
      return text.length > 30 ? `${text.slice(0, 27)}...` : text;
    }

    function shortId(value) {
      const text = String(value);
      return text.length > 22 ? `${text.slice(0, 19)}...` : text;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[char]));
    }

    document.getElementById("reset").addEventListener("click", () => {
      selectedId = null;
      showDetail(null);
      transform = { x: 0, y: 0, k: 1 };
      init();
    });
    document.getElementById("release").addEventListener("click", () => {
      nodes.forEach(node => node.pinned = false);
      startSimulation();
      draw();
    });
    document.getElementById("fit").addEventListener("click", fitToGraph);
    search.addEventListener("input", updateSearch);
    spacing.addEventListener("input", startSimulation);
    svg.addEventListener("pointerdown", startPan);
    svg.addEventListener("pointermove", pointerMove);
    svg.addEventListener("pointerup", pointerUp);
    svg.addEventListener("pointercancel", pointerUp);
    svg.addEventListener("wheel", event => {
      event.preventDefault();
      const before = toGraphPoint(event);
      const factor = event.deltaY < 0 ? 1.08 : .92;
      transform.k = Math.min(2.2, Math.max(.38, transform.k * factor));
      transform.x = event.clientX - svg.getBoundingClientRect().left - before.x * transform.k;
      transform.y = event.clientY - svg.getBoundingClientRect().top - before.y * transform.k;
      applyTransform();
    }, { passive:false });
    window.addEventListener("resize", () => {
      resize();
      fitToGraph();
      draw();
    });
    init();
    setTimeout(fitToGraph, 80);
  </script>
</body>
</html>
"""
    return (
        template.replace("__PROFILE__", _e(data["profile_id"]))
        .replace("__GENERATED__", _e(data["generated_at"]))
        .replace(
            "__COUNTS__",
            "\n      ".join(
                [
                    _count("Thoughts", data["counts"]["thoughts"]),
                    _count("Episodes", data["counts"]["episodes"]),
                    _count("Entities", data["counts"]["entities"]),
                    _count("Hints", data["counts"]["hints"]),
                    _count("Tags", data["counts"]["tags"]),
                    _count("Sessions", data["counts"]["sessions"]),
                    _count("mem0 hits", data["counts"]["mem0_results"]),
                ]
            ),
        )
        .replace("__MEM0_STATUS__", _e(data["mem0"]["status"]))
        .replace("__SEEDED__", _e(seeded))
        .replace("__MEM0_CARDS__", mem0_cards)
        .replace("__JSON_PRE__", _e(json.dumps(data, indent=2, ensure_ascii=True)))
        .replace("__PAYLOAD__", payload.replace("</", "<\\/"))
    )


def _node_card(node: dict[str, Any]) -> str:
    return (
        f'<article class="node {_e(node["kind"])}">'
        f'<span class="kind">{_e(node["kind"])}</span>'
        f'<h3>{_e(node["label"])}</h3>'
        f'<p>{_e(node["detail"])}</p>'
        f"<p><code>{_e(node['id'])}</code></p>"
        "</article>"
    )


def _edge_row(edge: dict[str, str]) -> str:
    return f"<tr><td><code>{_e(edge['from'])}</code></td><td>{_e(edge['label'])}</td><td><code>{_e(edge['to'])}</code></td></tr>"


def _mem0_card(item: dict[str, Any]) -> str:
    score = f" score={item['score']}" if item.get("score") is not None else ""
    source = f" source={item['source_id']}" if item.get("source_id") else ""
    return (
        '<article class="mem0-card">'
        f"<strong>{_e(item['memory_type'])}</strong>"
        f"<p>{_e(item['text'])}</p>"
        f"<p><code>{_e(score.strip() or 'live recall')}</code> <code>{_e(source.strip() or 'mem0')}</code></p>"
        "</article>"
    )


def _empty_mem0(status: str) -> str:
    return f'<article class="mem0-card"><strong>No mem0 results rendered</strong><p>{_e(status)}</p></article>'


def _count(label: str, value: int) -> str:
    return f'<div class="count"><strong>{value}</strong><span>{_e(label)}</span></div>'


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
