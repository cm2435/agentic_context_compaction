"""Core compaction components."""

from .compactor import ContextCompactor
from .protocols import CompactionStrategy, MessageT, TokenCounter
from .types import ClaudeAgentMessage, OpenAIAgentsMessage, PydanticAIMessage

__all__ = [
    "ContextCompactor",
    "TokenCounter",
    "CompactionStrategy",
    "MessageT",
    "PydanticAIMessage",
    "OpenAIAgentsMessage",
    "ClaudeAgentMessage",
]
