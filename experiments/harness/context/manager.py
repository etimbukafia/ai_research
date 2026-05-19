"""Context manager for the agent harness.

Owns the live message history and named memory blocks. Acts as the single
orchestration point for token accounting, overflow detection, and strategy-
driven compression.

Typical usage::

    from experiments.harness.context.manager import ContextManager
    from experiments.harness.context.strategies import production_preset
    from experiments.harness.context.tokenizer import get_tokenizer

    def my_summarizer(msgs):
        ...  # call your LLM here
        return summary_text

    cm = ContextManager(
        context_window=128_000,
        tokenizer=get_tokenizer("gpt-4o"),
        strategy=production_preset(summarizer=my_summarizer),
    )

    cm.add_message({"role": "system", "content": "You are a helpful agent."})
    cm.add_message({"role": "user",   "content": "Hello!"})

    # before every LLM call:
    payload = cm.get_context()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .memory import MemoryBlock
from .strategies import CompressionStrategy, SlidingWindowStrategy
from .tokenizer import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

@dataclass
class ContextManager:
    """Manages the agent's live message history and named memory blocks.

    Attributes:
        context_window:        Total token budget for the model (prompt + output).
        reserve_output_tokens: Tokens reserved for the model's reply.
        warn_threshold:        Usage ratio that triggers a near-overflow warning.
        tokenizer:             Token-counting backend (auto-selected if omitted).
        strategy:              Compression strategy (SlidingWindow if omitted).
    """

    context_window: int = 8_192
    reserve_output_tokens: int = 1_024
    warn_threshold: float = 0.85
    tokenizer: Tokenizer = field(default_factory=get_tokenizer)
    strategy: CompressionStrategy = field(
        default_factory=lambda: SlidingWindowStrategy(keep_system=True, keep_recent=20)
    )

    # Private state — not dataclass fields so they're excluded from __repr__.
    _messages: List[Dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _blocks: Dict[str, MemoryBlock] = field(
        default_factory=dict, init=False, repr=False
    )
    _cached_token_count: int = field(default=0, init=False, repr=False)
    _dirty: bool = field(default=True, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _token_budget(self) -> int:
        """Tokens available for the prompt (context_window minus output reserve)."""
        return max(0, self.context_window - self.reserve_output_tokens)

    def _recount(self) -> int:
        """Recompute and cache the token count of the current message list."""
        self._cached_token_count = self.tokenizer.count_messages_tokens(self._messages)
        self._dirty = False
        return self._cached_token_count

    def _mark_dirty(self) -> None:
        self._dirty = True

    # ------------------------------------------------------------------
    # Token accounting (public)
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Current token count of all messages (lazy, cached)."""
        if self._dirty:
            self._recount()
        return self._cached_token_count

    @property
    def usage_ratio(self) -> float:
        """Fraction of the token budget currently consumed (0.0 – 1.0+)."""
        budget = self._token_budget
        return self.token_count / budget if budget > 0 else 1.0

    def is_near_overflow(self, threshold: Optional[float] = None) -> bool:
        """Return True when usage exceeds *threshold* (default: ``warn_threshold``)."""
        return self.usage_ratio >= (threshold if threshold is not None else self.warn_threshold)

    def is_over_budget(self) -> bool:
        """Return True when the message list exceeds the token budget."""
        return self.token_count > self._token_budget

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, message: Dict[str, Any]) -> None:
        """Append a single message and invalidate the token cache."""
        self._messages.append(message)
        self._mark_dirty()

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Append multiple messages in one call."""
        self._messages.extend(messages)
        self._mark_dirty()

    def pop_message(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """Remove and return the message at *index* (default: last).

        Returns ``None`` if the history is empty.
        """
        if not self._messages:
            return None
        msg = self._messages.pop(index)
        self._mark_dirty()
        return msg

    def clear(self) -> None:
        """Remove all messages from the history."""
        self._messages.clear()
        self._cached_token_count = 0
        self._dirty = False

    # ------------------------------------------------------------------
    # Memory block management
    # ------------------------------------------------------------------

    def add_block(self, block: MemoryBlock) -> None:
        """Register a ``MemoryBlock`` under its ``name``.

        Raises:
            ValueError: If a block with that name is already registered.
        """
        if block.name in self._blocks:
            raise ValueError(f"Block already registered: {block.name!r}")
        # Sync initial token count.
        block._token_count = self.tokenizer.count_tokens(block.content)
        self._blocks[block.name] = block

    def update_block(self, block_name: str, content: str) -> None:
        """Replace the content of an existing block and refresh its token count.

        Raises:
            KeyError:   If no block with that name exists.
            ValueError: If the block is read-only.
        """
        block = self._get_block_or_raise(block_name)
        block.update(content)
        block._token_count = self.tokenizer.count_tokens(content)

    def get_block(self, block_name: str) -> Optional[MemoryBlock]:
        """Return the block for *block_name*, or ``None`` if not found."""
        return self._blocks.get(block_name)

    def delete_block(self, block_name: str) -> None:
        """Remove a block from the registry.

        Raises:
            KeyError: If no block with that name exists.
        """
        if block_name not in self._blocks:
            raise KeyError(f"No block registered: {block_name!r}")
        del self._blocks[block_name]

    def compress_block(self, block_name: str) -> None:
        """Run the block's own strategy to compress its content, if applicable.

        No-op when the block has no strategy or is within its token budget.

        Raises:
            KeyError: If no block with that name exists.
        """
        block = self._get_block_or_raise(block_name)
        if block.strategy is None or not block.should_compress():
            return

        msgs = [block.to_message()]
        budget = block.max_tokens or self._token_budget
        compressed = block.strategy.compress(msgs, budget, self.tokenizer)
        if compressed:
            block.update(compressed[0].get("content", block.content))
            block.mark_compressed()
            block._token_count = self.tokenizer.count_tokens(block.content)

    def _get_block_or_raise(self, name: str) -> MemoryBlock:
        block = self._blocks.get(name)
        if block is None:
            raise KeyError(f"No block registered: {name!r}")
        return block

    # ------------------------------------------------------------------
    # Overflow handling
    # ------------------------------------------------------------------

    def check_overflow(self) -> None:
        """Log a warning if usage is near or over the token budget."""
        ratio = self.usage_ratio
        if ratio > 1.0:
            logger.warning(
                "Context over budget: %d / %d tokens (%.0f%%)",
                self.token_count,
                self._token_budget,
                ratio * 100,
            )
        elif ratio >= self.warn_threshold:
            logger.warning(
                "Context near overflow: %d / %d tokens (%.0f%%)",
                self.token_count,
                self._token_budget,
                ratio * 100,
            )

    def truncate(self) -> None:
        """Run the compression strategy to bring the history within budget.

        No-op if already within budget.
        """
        if not self.is_over_budget():
            return

        logger.debug(
            "Compressing context: %d tokens → target %d",
            self.token_count,
            self._token_budget,
        )
        self._messages = self.strategy.compress(
            self._messages, self._token_budget, self.tokenizer
        )
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """A shallow copy of the current message list."""
        return list(self._messages)

    @property
    def blocks(self) -> Dict[str, MemoryBlock]:
        """A shallow copy of the block registry."""
        return dict(self._blocks)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return the current message list (same as ``messages`` property)."""
        return list(self._messages)

    def get_context(self, auto_compress: bool = True) -> List[Dict[str, Any]]:
        """Return the LLM-ready message list, compressing first if needed.

        Args:
            auto_compress: When True (default), ``truncate()`` is called
                automatically before returning if the history is over budget.

        Returns:
            Ordered list of message dicts ready to pass to the LLM.
        """
        if auto_compress and self.is_over_budget():
            self.truncate()
        self.check_overflow()
        return list(self._messages)

    def clear_all(self) -> None:
        """Clear messages *and* all registered memory blocks."""
        self.clear()
        self._blocks.clear()

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._messages[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._messages)

    def __str__(self) -> str:
        return (
            f"ContextManager("
            f"messages={len(self._messages)}, "
            f"tokens={self.token_count}/{self._token_budget}, "
            f"blocks={list(self._blocks)})"
        )

    def __repr__(self) -> str:
        return (
            f"ContextManager("
            f"context_window={self.context_window}, "
            f"reserve_output_tokens={self.reserve_output_tokens}, "
            f"messages={len(self._messages)}, "
            f"blocks={list(self._blocks)}, "
            f"tokenizer={self.tokenizer!r}, "
            f"strategy={self.strategy!r})"
        )
