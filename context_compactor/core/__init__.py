"""Core compaction components."""

from .compactor import ContextCompactor
from .hooks import CallbackHook, CompactionHook, CompactionResult, LoggingHook
from .protocols import CompactionStrategy, MessageT, TokenCounter
from .types import ClaudeAgentMessage, OpenAIAgentsMessage, PydanticAIMessage

__all__ = [
    # Core compactor
    "ContextCompactor",
    # Protocols
    "TokenCounter",
    "CompactionStrategy",
    "MessageT",
    # Hooks
    "CompactionHook",
    "CompactionResult",
    "LoggingHook",
    "CallbackHook",
    # Type aliases
    "PydanticAIMessage",
    "OpenAIAgentsMessage",
    "ClaudeAgentMessage",
]
