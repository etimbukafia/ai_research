from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING, Dict, List, Any


if TYPE_CHECKING:
    from .strategies import CompressionStrategy

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    WORKING = "working"       # agent’s active short-term memory.
    EPISODIC = "episodic"     # long/medium term memory of past experiences.
    SEMANTIC = "semantic"     # long-term memory of facts, stable truths, and knowledge.
    PROCEDURAL = "procedural" # long-term memory of how to do things.
    SCRATCHPAD = "scratchpad" # temporary reasoning space


@dataclass
class MemoryBlock:
    """
    Structured memory block for agent context management.

    Supports:
    - token limits
    - compression strategies
    - retrieval metadata
    - dependency tracking
    - trust boundaries
    - checkpoint persistence
    - active compression workflows
    """

    # Core identity
    name: str
    memory_type: MemoryType = MemoryType.WORKING
    content: str = ""

    # Token management
    max_tokens: Optional[int] = None
    priority: float = 0.5
    importance: float = 0.5

    # Compression
    strategy: Optional["CompressionStrategy"] = None
    compressed: bool = False
    summary_version: int = 0

    # Access control
    read_only: bool = False
    trusted: bool = True

    # Description
    description: str = ""

    # Retrieval metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Internal state
    _token_count: int = field(default=0, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def _ensure_writable(self) -> None:
        """
        Checks whether the block is allowed to be modified.
        """
        if self.read_only:
            raise ValueError(
                f"Cannot modify read-only block: {self.name}"
            )

    def _touch(self) -> None:
        """
        Updates internal tracking whenever content changes.
        """
        self.updated_at = datetime.utcnow()  # Tracks when the block was last modified.
        self._dirty = True                   # Indicates the block has been modified since the last checkpoint and should be saved/checkpointed.

    def update(self, content: str) -> None:
        """
        Completely replaces the block content.
        """
        self._ensure_writable() # Verify writability.
        self.content = content  # Replace old content.
        self.compressed = False # Mark compression as invalid.
        self._touch()           # Update timestamps and dirty state.

    def append(
        self,
        content: str,
        separator: str = "\n"
    ) -> None:
        """
        Appends new content to the existing content.
        
        Args:
            content: Content to append.
            separator: Separator to use between existing and new content.

        Why: 
           Perfect for logs, chat history, running notes, observations, event tracking
        """
        self._ensure_writable()

        if self.content:
            self.content = f"{self.content}{separator}{content}"
        else:
            self.content = content

        self._touch()

    def clear(self) -> None:
        """
        Removes all content from the block.
        """
        self._ensure_writable()
        self.content = ""
        self._token_count = 0
        self.compressed = False
        self._touch()

    def mark_compressed(self) -> None:
        """
        Marks the block as compressed.
        """
        self.compressed = True
        self.summary_version += 1
        self._touch()

    def add_dependency(self, block_name: str) -> None:
        """
        Creates a relationship between memory blocks.

        Args:
            block_name: Name of the block to depend on.

        Why: 
           This is useful for creating a hierarchy of memory blocks, 
           where one block depends on another.
           This supports dependency-aware retrieval
        """
        if block_name not in self.depends_on:
            self.depends_on.append(block_name)
            self._touch()

    def to_message(self, role: str = "system") -> dict:
        """
        Converts memory into LLM chat format
        """
        return {
            "role": role,
            "content": self.content
        }

    def should_compress(self) -> bool:
        """
        Decides whether the block exceeds its token budget
        """
        if self.max_tokens is None:
            return False
        return self._token_count > self.max_tokens

    def __len__(self) -> int:
        """
        Returns the length of the memory block.
        """
        return len(self.content)