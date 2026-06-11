"""Frozen experiment constants shared across runtime modules."""

from typing import Final

CLASSIFICATION_LABELS: Final[tuple[str, ...]] = (
    "supported",
    "weakly_supported",
    "unsupported",
    "insufficient_evidence",
)

MEMORY_CONDITIONS: Final[tuple[str, ...]] = (
    "no_memory",
    "raw_memory",
    "validated_lessons",
)
