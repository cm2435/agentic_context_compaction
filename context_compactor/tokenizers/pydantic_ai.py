"""Token counter for pydantic-ai messages.

Requires: pip install context-compactor[pydantic-ai]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_core import ErrorDetails


@dataclass
class PydanticAITokenCounter:
    """
    Token counter for pydantic-ai ModelRequest and ModelResponse messages.

    Internally extracts text from all part types and counts tokens.

    Args:
        chars_per_token: Average characters per token (default 4.0)
        model: Optional model name for more accurate counting (future: tiktoken)

    Example:
        ```python
        from context_compactor.tokenizers.pydantic_ai import PydanticAITokenCounter

        counter = PydanticAITokenCounter()
        tokens = counter.count_messages(messages)
        ```
    """

    chars_per_token: float = 4.0
    model: str | None = None

    def count_messages(self, messages: list[ModelRequest | ModelResponse]) -> int:
        """Count total tokens across all messages."""
        return sum(self.count_single(msg) for msg in messages)

    def count_single(self, message: ModelRequest | ModelResponse) -> int:
        """Count tokens in a single message."""
        text = self._extract_text(message)
        # Add overhead for message structure
        overhead = 4
        return int(len(text) / self.chars_per_token) + overhead

    def _extract_text(self, message: ModelRequest | ModelResponse) -> str:
        """Extract text from all parts of a message."""
        texts = []
        for part in message.parts:
            text = self._extract_part(part)
            if text:
                texts.append(text)
        return " ".join(texts)

    def _extract_part(self, part: ModelRequestPart | ModelResponsePart) -> str:
        """Extract text from a single message part (exhaustive match)."""
        # Request parts
        match part:
            case SystemPromptPart(content=content):
                return content
            case UserPromptPart(content=content):
                return self._extract_user_content(content)
            case ToolReturnPart(content=content, tool_name=name):
                return f"{name}: {self._serialize_any(content)}"
            case RetryPromptPart(content=content):
                return self._serialize_retry_content(content)
            # Response parts
            case TextPart(content=content):
                return content
            case ThinkingPart(content=content):
                return content
            case ToolCallPart(tool_name=name, args=args):
                return f"{name}({args})"
            case BuiltinToolCallPart(tool_name=name, args=args):
                return f"{name}({args})"
            case BuiltinToolReturnPart(content=content):
                return self._serialize_any(content)
            case FilePart():
                # Binary content - skip for token counting
                return ""
            case _:
                # Unknown part type - convert to string
                return str(part)

    def _extract_user_content(self, content: str | Sequence[UserContent]) -> str:
        """Extract text from UserPromptPart content."""
        if isinstance(content, str):
            return content
        # Multimodal content - extract text parts only
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            # Skip ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent, CachePoint
        return " ".join(texts)

    def _serialize_retry_content(self, content: list[ErrorDetails] | str) -> str:
        """Serialize RetryPromptPart content."""
        if isinstance(content, str):
            return content
        # ErrorDetails list - extract message strings
        return " ".join(error.get("msg", str(error)) for error in content)

    def _serialize_any(self, content: Any) -> str:
        """Serialize tool return content (can be any type)."""
        if isinstance(content, str):
            return content
        return str(content)
