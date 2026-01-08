"""Claude Agent SDK integration adapter.

Provides integration with claude-agent-sdk's PreCompact hook.

Requires: pip install context-compactor[claude-agent]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_agent_sdk.types import (
    AssistantMessage,
    HookInput,
    HookJSONOutput,
    SyncHookJSONOutput,
    UserMessage,
)

if TYPE_CHECKING:
    from ..core.compactor import ContextCompactor

# Type alias
ClaudeAgentMessage = UserMessage | AssistantMessage


def claude_agent_adapter(
    compactor: ContextCompactor[ClaudeAgentMessage],
) -> tuple[str, list]:
    """
    Create a PreCompact hook configuration for claude-agent-sdk.

    The Claude Agent SDK has native support for context compaction via
    the PreCompact hook event. This adapter integrates our compactor
    with that hook system.

    Args:
        compactor: A ContextCompactor configured for claude-agent-sdk messages

    Returns:
        A tuple of (hook_event, hook_matchers) for ClaudeAgentOptions.hooks

    Example:
        ```python
        from context_compactor import ContextCompactor
        from context_compactor.tokenizers.claude_agent import ClaudeAgentTokenCounter
        from context_compactor.strategies.generic import KeepFirstLast
        from context_compactor.adapters.claude_agent import claude_agent_adapter
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

        compactor = ContextCompactor(
            max_context_tokens=200_000,
            strategy=KeepFirstLast(keep_first=2, keep_last=10),
            token_counter=ClaudeAgentTokenCounter(),
        )

        # The PreCompact hook is triggered by Claude Agent SDK
        # when context needs compaction
        async def pre_compact_hook(
            input: HookInput,
            tool_use_id: str | None,
            context: dict,
        ) -> HookJSONOutput:
            # Return custom instructions for the compaction
            return {
                "continue_": True,
                "hookSpecificOutput": {
                    "hookEventName": "PreCompact",
                    "additionalContext": "Focus on preserving tool interactions.",
                },
            }

        options = ClaudeAgentOptions(
            hooks={
                "PreCompact": [
                    HookMatcher(hooks=[pre_compact_hook]),
                ],
            },
        )
        ```

    Note:
        The Claude Agent SDK handles context compaction natively via the
        PreCompact hook. This adapter provides a way to customize the
        compaction behavior, but the actual compaction is performed by
        the Claude Agent SDK itself.

        For full control over compaction, you may need to intercept
        messages at a different level in your application.
    """
    from claude_agent_sdk.types import HookMatcher

    async def pre_compact_callback(
        input: HookInput,
        tool_use_id: str | None,
        context: dict[str, Any],
    ) -> HookJSONOutput:
        """
        PreCompact hook callback.

        The Claude Agent SDK triggers this before performing compaction.
        We can provide custom instructions for how compaction should proceed.
        """
        if not isinstance(input, dict) or input.get("hook_event_name") != "PreCompact":
            return {"continue_": True}

        # Get trigger information
        trigger = input.get("trigger", "auto")
        custom_instructions = input.get("custom_instructions")

        # Build compaction guidance based on our strategy
        guidance = _build_compaction_guidance(compactor, trigger, custom_instructions)

        result: SyncHookJSONOutput = {
            "continue_": True,
        }

        if guidance:
            result["hookSpecificOutput"] = {
                "hookEventName": "PreCompact",
                "additionalContext": guidance,
            }

        return result

    hook_matcher = HookMatcher(hooks=[pre_compact_callback])

    return ("PreCompact", [hook_matcher])


def _build_compaction_guidance(
    compactor: ContextCompactor[ClaudeAgentMessage],
    trigger: str,
    custom_instructions: str | None,
) -> str:
    """Build guidance text for the compaction process."""
    parts = []

    # Add info about our configuration
    parts.append(
        f"Target context: {compactor.max_context_tokens:,} tokens, "
        f"trigger at {compactor.trigger_at_percent * 100:.0f}%"
    )

    # Add strategy name if available
    strategy_name = type(compactor.strategy).__name__
    parts.append(f"Strategy: {strategy_name}")

    # Add custom instructions if provided
    if custom_instructions:
        parts.append(f"User guidance: {custom_instructions}")

    return ". ".join(parts)
