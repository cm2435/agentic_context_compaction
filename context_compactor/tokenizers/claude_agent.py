"""Token counter for claude-agent-sdk messages.

Requires: pip install context-compactor[claude-agent]
"""

from __future__ import annotations

from dataclasses import dataclass

from claude_agent_sdk.types import (
    AssistantMessage,
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# Type alias for all message types we handle
ClaudeAgentMessage = UserMessage | AssistantMessage


@dataclass
class ClaudeAgentTokenCounter:
    """
    Token counter for claude-agent-sdk UserMessage and AssistantMessage.

    Handles all ContentBlock types within messages.

    Args:
        chars_per_token: Average characters per token (default 4.0)
        model: Optional model name for more accurate counting

    Example:
        ```python
        from context_compactor.tokenizers.claude_agent import ClaudeAgentTokenCounter

        counter = ClaudeAgentTokenCounter()
        tokens = counter.count_messages(messages)
        ```
    """

    chars_per_token: float = 4.0
    model: str | None = None

    def count_messages(self, messages: list[ClaudeAgentMessage]) -> int:
        """Count total tokens across all messages."""
        return sum(self.count_single(msg) for msg in messages)

    def count_single(self, message: ClaudeAgentMessage) -> int:
        """Count tokens in a single message."""
        text = self._extract_text(message)
        overhead = 4
        return int(len(text) / self.chars_per_token) + overhead

    def _extract_text(self, message: ClaudeAgentMessage) -> str:
        """Extract text from a message."""
        match message:
            case UserMessage(content=str(text)):
                return text
            case UserMessage(content=list() as blocks):
                return self._extract_blocks(blocks)
            case AssistantMessage(content=blocks):
                return self._extract_blocks(blocks)
            case _:
                return str(message)

    def _extract_blocks(self, blocks: list[ContentBlock]) -> str:
        """Extract text from a list of content blocks."""
        texts = []
        for block in blocks:
            text = self._extract_block(block)
            if text:
                texts.append(text)
        return " ".join(texts)

    def _extract_block(self, block: ContentBlock) -> str:
        """Extract text from a single content block (exhaustive match)."""
        match block:
            case TextBlock(text=text):
                return text
            case ThinkingBlock(thinking=thinking):
                return thinking
            case ToolUseBlock(name=name, input=input_data):
                return f"{name}({input_data})"
            case ToolResultBlock(content=str(content)):
                return content
            case ToolResultBlock(content=list() as parts):
                return self._extract_tool_result_parts(parts)
            case ToolResultBlock(content=None):
                return ""
            case _:
                return str(block)

    def _extract_tool_result_parts(self, parts: list) -> str:
        """Extract text from tool result content parts."""
        texts = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts)
