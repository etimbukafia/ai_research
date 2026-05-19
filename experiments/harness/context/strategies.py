"""Compression strategies for context management.

Implements six core compression techniques for AI agent context engineering,
based on 2025-2026 state-of-the-art research:

    1. HierarchicalSummarizationStrategy  – rolling multi-tier summarization
    2. JITRetrievalStrategy               – just-in-time pointer-based retrieval
    3. ObservationMaskingStrategy         – mask/replace stale tool outputs
    4. TokenPruningStrategy               – LLMLingua-style low-info token removal
    5. TaskBoundaryStrategy               – compress at task seams, not token thresholds
    6. ImportanceScoredStrategy           – multi-factor importance scoring with keyword boost

Plus:
    - DeduplicationStrategy               – Jaccard near-duplicate removal (utility)
    - SlidingWindowStrategy               – simple recency window (fallback)
    - HybridStrategy                      – pipeline combinator with production preset
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class CompressionStrategy(ABC):
    """Abstract base class for all compression strategies."""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Compress a message list to fit within *target_tokens*.

        Args:
            messages:      Ordered list of chat message dicts (role/content).
            target_tokens: Maximum token budget for the returned list.
            tokenizer:     Object exposing ``count_message_tokens(msg)``
                           and ``count_messages_tokens(msgs)`` methods.

        Returns:
            Compressed message list, preserving original chronological order.
        """


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_text(msg: Dict[str, Any]) -> str:
    """Extract plain-text content from a message dict."""
    content = msg.get("content") or ""
    return str(content).strip()


def _jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard token similarity between two strings."""
    if not text_a or not text_b:
        return 0.0
    words_a: Set[str] = set(re.findall(r"\w+", text_a.lower()))
    words_b: Set[str] = set(re.findall(r"\w+", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _is_tool_message(msg: Dict[str, Any]) -> bool:
    """Return True if the message is a tool/function result."""
    return msg.get("role") in ("tool", "function")


def _safe_count(tokenizer: Any, messages: List[Dict[str, Any]]) -> int:
    """Count tokens, tolerating both single-message and multi-message APIs."""
    try:
        return tokenizer.count_messages_tokens(messages)
    except AttributeError:
        return sum(tokenizer.count_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# 1. Hierarchical Summarization Strategy
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalSummarizationStrategy(CompressionStrategy):
    """Rolling multi-tier summarization.

    Maintains three layers of context fidelity:
      - **Recent** (verbatim)  – last ``recent_window`` non-system turns.
      - **Episode summary**    – LLM-generated summary of the mid-range history.
      - **System anchor**      – system messages, always preserved verbatim.

    The ``summarizer`` callable receives a list of messages and returns a
    single summary string.  Wire in any LLM call here.

    Attributes:
        summarizer:     ``(messages) -> str`` function that produces a summary.
        recent_window:  Number of most-recent messages to keep verbatim.
        summary_role:   Role tag for injected summary messages.
        summary_prefix: Prefix prepended to every generated summary.
    """

    summarizer: Callable[[List[Dict[str, Any]]], str]
    recent_window: int = 10
    summary_role: str = "system"
    summary_prefix: str = "[Conversation Summary]\n"

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Summarize older messages, keep recent ones verbatim."""
        if not messages:
            return []

        system_msgs: List[Dict[str, Any]] = []
        non_system: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        # If already within budget, return as-is.
        if _safe_count(tokenizer, messages) <= target_tokens:
            return messages

        # Split non-system into "to summarize" vs "recent verbatim".
        if len(non_system) <= self.recent_window:
            return messages  # nothing old enough to summarize

        to_summarize = non_system[: -self.recent_window]
        recent = non_system[-self.recent_window :]

        summary_text = self.summarizer(to_summarize)
        summary_msg: Dict[str, Any] = {
            "role": self.summary_role,
            "content": f"{self.summary_prefix}{summary_text}",
        }

        result = system_msgs + [summary_msg] + recent

        # If still over budget, recursively summarize (handles deep histories).
        if _safe_count(tokenizer, result) > target_tokens and len(recent) > 1:
            return self.compress(result, target_tokens, tokenizer)

        return result


# ---------------------------------------------------------------------------
# 2. JIT Retrieval Strategy
# ---------------------------------------------------------------------------

@dataclass
class JITRetrievalStrategy(CompressionStrategy):
    """Just-in-Time pointer-based retrieval.

    Instead of carrying large tool outputs or document chunks in-context,
    this strategy replaces them with lightweight reference stubs:

        ``[REF:<id>] <excerpt>``

    The ``retrieve`` callable is the agent's lookup function that can
    hydrate a reference when it is actually needed for reasoning.

    Attributes:
        retrieve:         ``(ref_id: str) -> str`` hydration function.
        store:            ``(content: str) -> str`` returns a ref_id.
        excerpt_chars:    How many characters of the original to keep as a hint.
        min_tokens:       Only offload messages that exceed this token count.
    """

    retrieve: Callable[[str], str]
    store: Callable[[str], str]
    excerpt_chars: int = 120
    min_tokens: int = 200

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Offload large messages to external store; inject reference stubs."""
        if not messages:
            return []

        result: List[Dict[str, Any]] = []

        for msg in messages:
            msg_tokens = tokenizer.count_message_tokens(msg)

            # Only offload large, non-system messages.
            if (
                msg_tokens >= self.min_tokens
                and msg.get("role") != "system"
                and _get_text(msg)
            ):
                text = _get_text(msg)
                ref_id = self.store(text)
                excerpt = text[: self.excerpt_chars].replace("\n", " ")
                stub: Dict[str, Any] = {
                    **msg,
                    "content": f"[REF:{ref_id}] {excerpt}…",
                    "_jit_ref": ref_id,      # internal metadata
                }
                result.append(stub)
            else:
                result.append(msg)

        return result

    def hydrate(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Restore a stub message to its full content on demand.

        Args:
            msg: A message that may contain a ``_jit_ref`` key.

        Returns:
            Message with full content restored.
        """
        ref_id = msg.get("_jit_ref")
        if ref_id:
            full_content = self.retrieve(ref_id)
            return {k: v for k, v in msg.items() if k != "_jit_ref"} | {
                "content": full_content
            }
        return msg


# ---------------------------------------------------------------------------
# 3. Observation Masking Strategy
# ---------------------------------------------------------------------------

@dataclass
class ObservationMaskingStrategy(CompressionStrategy):
    """Replace stale tool/observation outputs with compact placeholders.

    Preserves the agent's reasoning trajectory (which tools were called and
    why) while eliminating raw observation bloat from older turns.

    Attributes:
        keep_recent_observations: Number of most-recent tool messages to
            keep verbatim (so the agent retains fresh grounding).
        placeholder_template:     Format string for masked messages.
            Use ``{role}`` and ``{excerpt}`` as placeholders.
        excerpt_chars:            Characters of original content to keep.
    """

    keep_recent_observations: int = 3
    placeholder_template: str = "[OBSERVATION MASKED – {role}: {excerpt}…]"
    excerpt_chars: int = 80

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Mask old tool outputs; keep recent ones and all non-tool messages."""
        if not messages:
            return []

        # Find indices of tool/observation messages, oldest-first.
        tool_indices = [
            i for i, m in enumerate(messages) if _is_tool_message(m)
        ]

        # The most-recent N are kept verbatim.
        keep_verbatim: Set[int] = set(tool_indices[-self.keep_recent_observations :])

        result: List[Dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if _is_tool_message(msg) and i not in keep_verbatim:
                text = _get_text(msg)
                excerpt = text[: self.excerpt_chars].replace("\n", " ")
                masked: Dict[str, Any] = {
                    **msg,
                    "content": self.placeholder_template.format(
                        role=msg.get("role", "tool"),
                        excerpt=excerpt,
                    ),
                }
                result.append(masked)
            else:
                result.append(msg)

        # If still over budget, iteratively mask more recent observations.
        if _safe_count(tokenizer, result) > target_tokens and keep_verbatim:
            reduced = self.keep_recent_observations - 1
            return ObservationMaskingStrategy(
                keep_recent_observations=max(0, reduced),
                placeholder_template=self.placeholder_template,
                excerpt_chars=self.excerpt_chars,
            ).compress(messages, target_tokens, tokenizer)

        return result


# ---------------------------------------------------------------------------
# 4. Token Pruning Strategy  (LLMLingua-style)
# ---------------------------------------------------------------------------

@dataclass
class TokenPruningStrategy(CompressionStrategy):
    """Intra-message low-information token removal.

    Approximates LLMLingua-style compression: sentences and phrases are
    scored by word rarity (inverse document frequency within the message
    list) and low-scoring ones are pruned or truncated.

    This is a *heuristic* approximation. A production deployment
    should route through an actual LLMLingua endpoint or equivalent model.

    Attributes:
        compression_ratio:  Target fraction of content to retain (0.0-1.0).
        min_sentence_score: Sentences below this IDF score are dropped.
        protect_roles:      Roles whose messages are never pruned.
    """

    compression_ratio: float = 0.6
    min_sentence_score: float = 0.2
    protect_roles: Tuple[str, ...] = ("system",)

    # ---- internal helpers --------------------------------------------------

    def _idf_scores(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute a simple IDF score per word across all messages."""
        doc_count = len(messages)
        word_doc_freq: Counter = Counter()
        for msg in messages:
            words = set(re.findall(r"\w+", _get_text(msg).lower()))
            word_doc_freq.update(words)
        return {
            w: (doc_count / (freq + 1))
            for w, freq in word_doc_freq.items()
        }

    def _score_sentence(self, sentence: str, idf: Dict[str, float]) -> float:
        """Return mean IDF score for the words in a sentence."""
        words = re.findall(r"\w+", sentence.lower())
        if not words:
            return 0.0
        return sum(idf.get(w, 0.0) for w in words) / len(words)

    def _prune_content(self, text: str, idf: Dict[str, float]) -> str:
        """Remove lowest-scoring sentences to hit the compression ratio."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            # Fall back to character-level truncation for single sentences.
            keep_chars = max(20, int(len(text) * self.compression_ratio))
            return text[:keep_chars] + ("…" if len(text) > keep_chars else "")

        scored = [
            (s, self._score_sentence(s, idf)) for s in sentences if s.strip()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        keep_n = max(1, int(len(scored) * self.compression_ratio))
        kept_sentences = {s for s, _ in scored[:keep_n]}

        # Reassemble in original order.
        return " ".join(s for s in sentences if s in kept_sentences)

    # ---- public interface --------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Prune low-information content from messages."""
        if not messages:
            return []

        if _safe_count(tokenizer, messages) <= target_tokens:
            return messages

        idf = self._idf_scores(messages)
        result: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role in self.protect_roles or not _get_text(msg):
                result.append(msg)
                continue

            pruned_content = self._prune_content(_get_text(msg), idf)
            result.append({**msg, "content": pruned_content})

        return result


# ---------------------------------------------------------------------------
# 5. Task-Boundary Compression Strategy
# ---------------------------------------------------------------------------

@dataclass
class TaskBoundaryStrategy(CompressionStrategy):
    """Compress at natural task seams rather than arbitrary token thresholds.

    Identifies task boundary markers in message content (e.g., "Task complete",
    "Step N done", structured JSON checkpoints) and compresses the completed
    sub-task history into a single checkpoint summary.

    Pairs well with a ``summarizer`` function — typically the same LLM call
    used in ``HierarchicalSummarizationStrategy``.

    Attributes:
        summarizer:         ``(messages) -> str`` produces a task checkpoint.
        boundary_patterns:  Regex patterns that signal a task boundary.
        checkpoint_role:    Role for injected checkpoint messages.
        checkpoint_prefix:  Prefix for checkpoint content.
        min_boundary_count: Minimum number of boundaries before compressing.
    """

    summarizer: Callable[[List[Dict[str, Any]]], str]
    boundary_patterns: Tuple[str, ...] = (
        r"(?i)task\s+(complete|done|finished)",
        r"(?i)step\s+\d+\s+(complete|done)",
        r"(?i)\[checkpoint\]",
        r"(?i)\b(completed|accomplished|resolved)\b",
    )
    checkpoint_role: str = "system"
    checkpoint_prefix: str = "[Task Checkpoint]\n"
    min_boundary_count: int = 1

    def _find_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Return indices of messages that mark a task boundary."""
        patterns = [re.compile(p) for p in self.boundary_patterns]
        return [
            i
            for i, msg in enumerate(messages)
            if any(p.search(_get_text(msg)) for p in patterns)
        ]

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Compress completed task segments into checkpoint summaries."""
        if not messages:
            return []

        if _safe_count(tokenizer, messages) <= target_tokens:
            return messages

        boundary_indices = self._find_boundaries(messages)

        if len(boundary_indices) < self.min_boundary_count:
            # No clear task boundaries — fall back gracefully (no-op here;
            # let the HybridStrategy chain pick this up).
            return messages

        # Compress everything up to (and including) the last boundary.
        split_point = boundary_indices[-1] + 1
        to_compress = messages[:split_point]
        remainder = messages[split_point:]

        system_msgs = [m for m in to_compress if m.get("role") == "system"]
        history = [m for m in to_compress if m.get("role") != "system"]

        if not history:
            return messages

        checkpoint_text = self.summarizer(history)
        checkpoint_msg: Dict[str, Any] = {
            "role": self.checkpoint_role,
            "content": f"{self.checkpoint_prefix}{checkpoint_text}",
        }

        return system_msgs + [checkpoint_msg] + remainder


# ---------------------------------------------------------------------------
# 6. Importance-Scored Retention Strategy  (enhanced)
# ---------------------------------------------------------------------------

@dataclass
class ImportanceScoredStrategy(CompressionStrategy):
    """Multi-factor importance scoring with keyword and entity boosting.

    Scores each message across five axes:
      1. **Role weight**    – system > user > assistant > tool
      2. **Recency decay**  – exponential decay with configurable rate
      3. **Content density** – penalise very short / empty messages
      4. **Structural signal** – boost tool-call messages and JSON content
      5. **Keyword boost**  – domain-critical terms increase score

    Attributes:
        system_weight:      Score weight for system messages (always 1.0).
        user_weight:        Score weight for user messages.
        assistant_weight:   Score weight for assistant messages.
        tool_weight:        Score weight for tool result messages.
        recency_decay:      Exponential decay factor (0.0-1.0).  Higher = less decay.
        min_score:          Drop messages below this threshold before greedy selection.
        keyword_boost:      Per-keyword score multiplier for domain-critical terms.
        domain_keywords:    Set of keywords that trigger a boost when found.
        always_keep_system: Unconditionally include system messages.
        always_keep_last_n: Always keep the N most-recent messages verbatim.
    """

    system_weight: float = 1.0
    user_weight: float = 0.85
    assistant_weight: float = 0.65
    tool_weight: float = 0.45
    recency_decay: float = 0.96
    min_score: float = 0.08
    keyword_boost: float = 1.35
    domain_keywords: Set[str] = field(default_factory=set)
    always_keep_system: bool = True
    always_keep_last_n: int = 4

    def _score(self, msg: Dict[str, Any], index: int, total: int) -> float:
        """Compute a composite importance score for a single message."""
        role = msg.get("role", "")

        # 1. Role weight
        role_weights: Dict[str, float] = {
            "system": self.system_weight,
            "user": self.user_weight,
            "assistant": self.assistant_weight,
            "tool": self.tool_weight,
            "function": self.tool_weight,
        }
        base = role_weights.get(role, 0.5)

        # 2. Recency decay  (index=0 is oldest, index=total-1 is newest)
        position_ratio = index / max(total - 1, 1)          # 0.0 → 1.0
        recency = self.recency_decay ** ((1.0 - position_ratio) * 10)

        # 3. Content density
        text = _get_text(msg)
        length = len(text)
        if length < 10:
            density = 0.4
        elif length < 60:
            density = 0.75
        else:
            density = 1.0

        # 4. Structural signal
        structural = 1.0
        if msg.get("tool_calls"):
            structural *= 1.25
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            structural *= 1.1   # likely JSON payload

        # 5. Keyword boost
        keyword_factor = 1.0
        if self.domain_keywords:
            words = set(re.findall(r"\w+", text.lower()))
            if words & {kw.lower() for kw in self.domain_keywords}:
                keyword_factor = self.keyword_boost

        return base * recency * density * structural * keyword_factor

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Retain highest-scoring messages within the token budget."""
        if not messages:
            return []

        if _safe_count(tokenizer, messages) <= target_tokens:
            return messages

        protected: Set[int] = set()

        # Pin system messages.
        if self.always_keep_system:
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    protected.add(i)

        # Pin last N messages verbatim.
        tail_start = max(0, len(messages) - self.always_keep_last_n)
        protected.update(range(tail_start, len(messages)))

        # Score the remainder.
        scored: List[Tuple[int, float]] = []
        for i, msg in enumerate(messages):
            if i in protected:
                continue
            score = self._score(msg, i, len(messages))
            if score >= self.min_score:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Greedily fill the budget.
        selected: Set[int] = set(protected)
        current_tokens = _safe_count(
            tokenizer, [messages[i] for i in sorted(protected)]
        )

        for i, _ in scored:
            msg_tokens = tokenizer.count_message_tokens(messages[i])
            if current_tokens + msg_tokens <= target_tokens:
                selected.add(i)
                current_tokens += msg_tokens

        return [messages[i] for i in sorted(selected)]


# ---------------------------------------------------------------------------
# Utility: Deduplication Strategy
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationStrategy(CompressionStrategy):
    """Remove near-duplicate messages using Jaccard similarity.

    Useful when tool loops or agent self-loops produce repeated outputs.

    Attributes:
        similarity_threshold: Jaccard ratio above which two messages are
            considered duplicates (0.0-1.0).
        keep_latest:          Keep the newer of two duplicates.
        compare_roles:        Only compare messages sharing the same role.
    """

    similarity_threshold: float = 0.85
    keep_latest: bool = True
    compare_roles: bool = True

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Deduplicate, then fall back to simple eviction if still over budget."""
        if not messages:
            return []

        keep: Set[int] = set(range(len(messages)))

        for i in range(len(messages)):
            if i not in keep:
                continue
            for j in range(i + 1, len(messages)):
                if j not in keep:
                    continue
                if self.compare_roles and messages[i].get("role") != messages[j].get("role"):
                    continue
                sim = _jaccard(_get_text(messages[i]), _get_text(messages[j]))
                if sim >= self.similarity_threshold:
                    keep.discard(i if self.keep_latest else j)

        result = [messages[i] for i in sorted(keep)]

        # Safety-net: evict oldest non-system messages if still over budget.
        while _safe_count(tokenizer, result) > target_tokens and result:
            for idx, msg in enumerate(result):
                if msg.get("role") != "system":
                    result.pop(idx)
                    break
            else:
                break  # only system messages remain

        return result


# ---------------------------------------------------------------------------
# Utility: Sliding Window Strategy  (simple fallback)
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowStrategy(CompressionStrategy):
    """Keep the most recent N messages, optionally anchoring system messages.

    The simplest and most predictable strategy.  Use as a final safety-net
    in a ``HybridStrategy`` chain.

    Attributes:
        keep_system:     Always keep system messages.
        keep_recent:     Number of recent non-system messages to retain.
        keep_first_user: Anchor the very first user message (original request).
    """

    keep_system: bool = True
    keep_recent: int = 20
    keep_first_user: bool = True

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Apply sliding window compression."""
        if not messages:
            return []

        protected: Set[int] = set()
        first_user_seen = False

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if self.keep_system and role == "system":
                protected.add(i)
            if self.keep_first_user and role == "user" and not first_user_seen:
                protected.add(i)
                first_user_seen = True

        non_protected = [i for i in range(len(messages)) if i not in protected]
        recent_indices = non_protected[-self.keep_recent :]

        result: List[Dict[str, Any]] = []
        for i in sorted(protected):
            result.append(messages[i])

        for i in recent_indices:
            candidate = messages[i]
            probe = result + [candidate]
            if _safe_count(tokenizer, probe) <= target_tokens:
                result.append(candidate)

        result.sort(key=lambda m: next(
            (i for i, orig in enumerate(messages) if orig is m), len(messages)
        ))
        return result


# ---------------------------------------------------------------------------
# StrategySelector  – memory-type-aware strategy routing
# ---------------------------------------------------------------------------

class StrategySelector:
    """Maps ``MemoryType`` values to compression strategies.

    When injected into ``HybridStrategy``, the selector is applied *before*
    the main strategy chain.  It partitions the mixed message list by
    ``memory_type``, compresses each partition with its designated strategy,
    and reassembles the result in the original chronological order.

    Messages without a ``memory_type`` key are treated as the ``default``
    type and handled by the fallback strategy.

    Usage::

        from experiments.harness.context.memory import MemoryType

        selector = StrategySelector(
            mapping={
                MemoryType.EPISODIC:   HierarchicalSummarizationStrategy(summarizer=llm_summarize),
                MemoryType.SEMANTIC:   DeduplicationStrategy(),
                MemoryType.SCRATCHPAD: SlidingWindowStrategy(keep_recent=2),
                MemoryType.PROCEDURAL: SlidingWindowStrategy(keep_recent=50),  # near-lossless
            },
            default=ImportanceScoredStrategy(),
        )

        hybrid = HybridStrategy(strategies=[...], selector=selector)

    Attributes:
        mapping:  Dict mapping ``MemoryType`` (or its string value) to a strategy.
        default:  Fallback strategy for untagged or unrecognised memory types.
    """

    def __init__(
        self,
        mapping: Dict[Any, CompressionStrategy],
        default: CompressionStrategy,
    ) -> None:
        # Normalise keys to string values so both enum instances and raw
        # strings ("working", "episodic", …) work as look-up keys.
        self._map: Dict[str, CompressionStrategy] = {
            (k.value if hasattr(k, "value") else str(k)): v
            for k, v in mapping.items()
        }
        self.default = default

    def select(self, memory_type: Any) -> CompressionStrategy:
        """Return the strategy registered for *memory_type*, or the default."""
        key = memory_type.value if hasattr(memory_type, "value") else str(memory_type)
        return self._map.get(key, self.default)

    def compress_partitioned(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Partition by memory type, compress each partition, reassemble.

        Each message may carry a ``"memory_type"`` key (string or
        ``MemoryType`` enum value).  Messages are grouped, compressed
        independently, then merged back in their original order.

        Args:
            messages:      Full mixed-type message list.
            target_tokens: Token budget forwarded to each per-type strategy.
            tokenizer:     Shared tokenizer instance.

        Returns:
            Reassembled message list in original chronological order.
        """
        if not messages:
            return []

        # Group messages by memory_type while preserving original position.
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for i, msg in enumerate(messages):
            raw_type = msg.get("memory_type", "")
            key = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            groups.setdefault(key, []).append((i, msg))

        # Compress each group with its designated strategy.
        survivors: List[Tuple[int, Dict[str, Any]]] = []
        for key, indexed_msgs in groups.items():
            strategy = self._map.get(key, self.default)
            raw_msgs = [m for _, m in indexed_msgs]

            # Allocate a proportional token budget for this partition.
            partition_ratio = len(raw_msgs) / len(messages)
            partition_budget = max(1, int(target_tokens * partition_ratio))

            compressed = strategy.compress(raw_msgs, partition_budget, tokenizer)
            compressed_set = {id(m) for m in compressed}

            # Re-attach original indices for those messages that survived.
            for orig_i, orig_msg in indexed_msgs:
                if id(orig_msg) in compressed_set:
                    survivors.append((orig_i, orig_msg))

        # Restore original chronological order.
        survivors.sort(key=lambda x: x[0])
        return [msg for _, msg in survivors]


# ---------------------------------------------------------------------------
# HybridStrategy  – pipeline combinator with production preset
# ---------------------------------------------------------------------------

@dataclass
class HybridStrategy(CompressionStrategy):
    """Apply multiple compression strategies in sequence.

    If a ``StrategySelector`` is provided, it runs *first* as a pre-pass:
    messages are partitioned by ``memory_type``, each partition is
    compressed by its designated strategy, and the result is reassembled
    before the main chain runs.  This lets the chain focus on overall
    budget enforcement while the selector handles type-specific logic.

    Attributes:
        strategies: Ordered list of strategies to apply after the selector.
        selector:   Optional memory-type-aware routing layer.
        early_exit: Stop the chain once budget is met (default True).
    """

    strategies: List[CompressionStrategy]
    selector: Optional[StrategySelector] = None
    early_exit: bool = True

    def __init__(
        self,
        strategies: List[CompressionStrategy],
        selector: Optional[StrategySelector] = None,
        early_exit: bool = True,
    ) -> None:
        self.strategies = strategies
        self.selector = selector
        self.early_exit = early_exit

    def compress(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Run selector pre-pass (if set), then the strategy chain."""
        result = messages

        # Pre-pass: partition by memory type and compress each group.
        if self.selector is not None:
            result = self.selector.compress_partitioned(
                result, target_tokens, tokenizer
            )

        # Main chain: apply strategies in order until budget is met.
        for strategy in self.strategies:
            if self.early_exit and _safe_count(tokenizer, result) <= target_tokens:
                break
            result = strategy.compress(result, target_tokens, tokenizer)

        return result


def production_preset(
    summarizer: Callable[[List[Dict[str, Any]]], str],
    domain_keywords: Optional[Set[str]] = None,
    selector: Optional[StrategySelector] = None,
) -> HybridStrategy:
    """Return the recommended production compression pipeline.

    Chain order (research-backed):
      1. StrategySelector pre-pass   – per-memory-type compression (optional)
      2. DeduplicationStrategy       – strip near-duplicates first
      3. ObservationMaskingStrategy  – compress stale tool outputs
      4. TaskBoundaryStrategy        – checkpoint finished sub-tasks
      5. ImportanceScoredStrategy    – multi-factor budget-aware selection
      6. HierarchicalSummarizationStrategy – summarize remaining history
      7. SlidingWindowStrategy       – final safety-net

    Args:
        summarizer:       LLM call for summarization steps.
        domain_keywords:  Optional set of domain-critical terms for boosting.
        selector:         Optional ``StrategySelector`` for memory-type routing.
                          If omitted, all messages are treated uniformly.

    Returns:
        A configured ``HybridStrategy`` ready for use.
    """
    return HybridStrategy(
        strategies=[
            DeduplicationStrategy(similarity_threshold=0.85, keep_latest=True),
            ObservationMaskingStrategy(keep_recent_observations=3),
            TaskBoundaryStrategy(summarizer=summarizer),
            ImportanceScoredStrategy(
                domain_keywords=domain_keywords or set(),
                always_keep_last_n=4,
            ),
            HierarchicalSummarizationStrategy(
                summarizer=summarizer,
                recent_window=8,
            ),
            SlidingWindowStrategy(keep_system=True, keep_recent=12),
        ],
        selector=selector,
        early_exit=True,
    )