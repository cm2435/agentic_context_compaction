"""Compaction lifecycle hooks.

Provides hook protocols and built-in implementations for monitoring
and reacting to compaction events.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Contains statistics about what happened during compaction,
    passed to `CompactionHook.on_end()`.

    Attributes:
        original_tokens: Token count before compaction
        compacted_tokens: Token count after compaction
        tokens_saved: Number of tokens removed (original - compacted)
        original_message_count: Number of messages before compaction
        compacted_message_count: Number of messages after compaction
        strategy_name: Name of the strategy class used
    """

    original_tokens: int
    compacted_tokens: int
    tokens_saved: int
    original_message_count: int
    compacted_message_count: int
    strategy_name: str


@runtime_checkable
class CompactionHook(Protocol):
    """Protocol for compaction lifecycle hooks.

    Implement this protocol to receive notifications when compaction
    starts and ends. Multiple hooks can be registered with a single
    `ContextCompactor`.

    Example:
        ```python
        @dataclass
        class MyHook:
            async def on_start(self) -> None:
                print("Compaction starting...")

            async def on_end(self, result: CompactionResult) -> None:
                print(f"Saved {result.tokens_saved} tokens")

        compactor = ContextCompactor(
            ...,
            hooks=[MyHook()],
        )
        ```
    """

    async def on_start(self) -> None:
        """Called when compaction begins.

        Use this to notify UI, start timers, or prepare for compaction.
        """
        ...

    async def on_end(self, result: CompactionResult) -> None:
        """Called when compaction completes.

        Args:
            result: Statistics about the compaction that was performed
        """
        ...


@dataclass
class LoggingHook:
    """Hook that prints compaction events to stdout.

    Useful for debugging and development.

    Args:
        prefix: String prefix for log messages (default: "[Compactor]")

    Example:
        ```python
        compactor = ContextCompactor(
            ...,
            hooks=[LoggingHook(prefix="[MyApp]")],
        )
        ```
    """

    prefix: str = "[Compactor]"

    async def on_start(self) -> None:
        """Log compaction start."""
        print(f"{self.prefix} Compaction started...")

    async def on_end(self, result: CompactionResult) -> None:
        """Log compaction results."""
        reduction_pct = (result.tokens_saved / result.original_tokens) * 100
        print(
            f"{self.prefix} Compaction complete: "
            f"{result.original_tokens:,} → {result.compacted_tokens:,} tokens "
            f"({reduction_pct:.1f}% reduction, {result.tokens_saved:,} saved)"
        )


@dataclass
class CallbackHook:
    """Hook that delegates to user-provided async functions.

    Convenience wrapper when you don't want to define a full class.

    Args:
        on_start_callback: Async function called when compaction starts
        on_end_callback: Async function called when compaction ends

    Example:
        ```python
        async def notify_ui():
            await websocket.send({"type": "compacting"})

        async def notify_done(result):
            await websocket.send({"type": "done", "saved": result.tokens_saved})

        compactor = ContextCompactor(
            ...,
            hooks=[CallbackHook(on_start_callback=notify_ui, on_end_callback=notify_done)],
        )
        ```
    """

    on_start_callback: Callable[[], Awaitable[None]] | None = None
    on_end_callback: Callable[[CompactionResult], Awaitable[None]] | None = None

    async def on_start(self) -> None:
        """Call the start callback if provided."""
        if self.on_start_callback:
            await self.on_start_callback()

    async def on_end(self, result: CompactionResult) -> None:
        """Call the end callback if provided."""
        if self.on_end_callback:
            await self.on_end_callback(result)
