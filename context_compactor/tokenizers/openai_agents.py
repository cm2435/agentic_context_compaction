"""Token counter for openai-agents SDK messages.

Requires: pip install context-compactor[openai-agents]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    MCPListToolsItem,
    MessageOutputItem,
    ReasoningItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpCall,
)

# Type alias for all message types we handle
OpenAIAgentsMessage = RunItem | TResponseInputItem


@dataclass
class OpenAIAgentsTokenCounter:
    """
    Token counter for openai-agents RunItem and input item messages.

    Handles both the wrapper types (MessageOutputItem, ToolCallItem, etc.)
    and raw OpenAI SDK types.

    Args:
        chars_per_token: Average characters per token (default 4.0)
        model: Optional model name for more accurate counting

    Example:
        ```python
        from context_compactor.tokenizers.openai_agents import OpenAIAgentsTokenCounter

        counter = OpenAIAgentsTokenCounter()
        tokens = counter.count_messages(run_items)
        ```
    """

    chars_per_token: float = 4.0
    model: str | None = None

    def count_messages(self, messages: list[OpenAIAgentsMessage]) -> int:
        """Count total tokens across all messages."""
        return sum(self.count_single(msg) for msg in messages)

    def count_single(self, message: OpenAIAgentsMessage) -> int:
        """Count tokens in a single message."""
        text = self._extract_text(message)
        overhead = 4
        return int(len(text) / self.chars_per_token) + overhead

    def _extract_text(self, message: OpenAIAgentsMessage) -> str:
        """Extract text from a message (handles all wrapper and raw types)."""
        match message:
            # Wrapper types (RunItem variants)
            case MessageOutputItem(raw_item=msg):
                return self._extract_output_message(msg)
            case ToolCallItem(raw_item=call):
                return self._extract_tool_call(call)
            case ToolCallOutputItem(output=output):
                return str(output)
            case ReasoningItem(raw_item=reasoning):
                return self._extract_reasoning(reasoning)
            case HandoffCallItem(raw_item=call):
                return f"handoff: {call.name}"
            case HandoffOutputItem():
                return ""  # Metadata only
            case MCPListToolsItem() | MCPApprovalRequestItem() | MCPApprovalResponseItem():
                return ""  # MCP protocol items - minimal text
            # Raw input item (dict)
            case dict() as d:
                return self._extract_input_item(d)
            case _:
                return str(message)

    def _extract_output_message(self, msg: ResponseOutputMessage) -> str:
        """Extract text from ResponseOutputMessage."""
        texts = []
        for content in msg.content:
            match content:
                case ResponseOutputText(text=text):
                    texts.append(text)
                case ResponseOutputRefusal(refusal=refusal):
                    texts.append(refusal)
                case _:
                    pass  # Skip unknown content types
        return " ".join(texts)

    def _extract_tool_call(self, call: Any) -> str:
        """Extract text from various tool call types."""
        match call:
            case ResponseFunctionToolCall(name=name, arguments=args):
                return f"{name}({args})"
            case ResponseComputerToolCall(action=action):
                return f"computer: {action}"
            case ResponseFileSearchToolCall():
                return "file_search"
            case ResponseFunctionWebSearch():
                return "web_search"
            case ResponseCodeInterpreterToolCall():
                return "code_interpreter"
            case ImageGenerationCall():
                return "image_generation"
            case LocalShellCall():
                return "local_shell"
            case McpCall():
                return "mcp_call"
            case dict() as d:
                name = d.get("name", d.get("type", "unknown"))
                args = d.get("arguments", d.get("input", ""))
                return f"{name}({args})"
            case _:
                return str(call)

    def _extract_reasoning(self, reasoning: ResponseReasoningItem) -> str:
        """Extract text from reasoning item."""
        if reasoning.summary:
            texts = []
            for summary in reasoning.summary:
                if hasattr(summary, "text"):
                    texts.append(summary.text)
            return " ".join(texts)
        return ""

    def _extract_input_item(self, item: dict) -> str:
        """Extract text from a raw input item dict."""
        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("input_text", "text"):
                        texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(texts)
        return str(content)
