"""Compaction strategies for message history.

Includes both generic strategies that work with any SDK message type,
and SDK-specific strategies that understand message structure.
"""

from .generic import (
    DropOldestUntilFits,
    KeepFirstLast,
    KeepRecentMessages,
    SlidingWindow,
)

__all__ = [
    # Generic strategies
    "KeepRecentMessages",
    "DropOldestUntilFits",
    "KeepFirstLast",
    "SlidingWindow",
]

# Optional: pydantic-ai strategies
try:
    from .pydantic_ai import (
        ChainedStrategy,  # noqa: F401
        DropThinking,  # noqa: F401
        KeepToolCalls,  # noqa: F401
        SummarizeMiddle,  # noqa: F401
    )

    __all__.extend(
        [
            "SummarizeMiddle",
            "KeepToolCalls",
            "DropThinking",
            "ChainedStrategy",
        ]
    )
except ImportError:
    pass
