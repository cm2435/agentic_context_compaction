"""Tests for ContextCompactor core functionality."""

from __future__ import annotations

import pytest

from context_compactor import ContextCompactor
from context_compactor.strategies import KeepRecentMessages

from .conftest import MockMessage, MockTokenCounter


class TestContextCompactorInit:
    """Tests for ContextCompactor initialization."""

    def test_init_with_required_args(self, mock_counter: MockTokenCounter):
        """Test initialization with required arguments."""
        strategy = KeepRecentMessages(keep_count=10)
        compactor = ContextCompactor(
            max_context_tokens=128_000,
            strategy=strategy,
            token_counter=mock_counter,
        )
        assert compactor.max_context_tokens == 128_000
        assert compactor.trigger_at_percent == 0.8  # default
        assert compactor.strategy is strategy
        assert compactor.token_counter is mock_counter

    def test_init_custom_trigger_percent(self, mock_counter: MockTokenCounter):
        """Test custom trigger percentage."""
        compactor = ContextCompactor(
            max_context_tokens=100_000,
            trigger_at_percent=0.9,
            strategy=KeepRecentMessages(),
            token_counter=mock_counter,
        )
        assert compactor.trigger_at_percent == 0.9
        assert compactor.trigger_threshold == 90_000

    def test_trigger_threshold_calculation(self, mock_counter: MockTokenCounter):
        """Test that trigger threshold is calculated correctly."""
        compactor = ContextCompactor(
            max_context_tokens=128_000,
            trigger_at_percent=0.8,
            strategy=KeepRecentMessages(),
            token_counter=mock_counter,
        )
        assert compactor.trigger_threshold == 102_400  # 128000 * 0.8

    def test_initial_stats_are_zero(self, mock_counter: MockTokenCounter):
        """Test that stats start at zero."""
        compactor = ContextCompactor(
            max_context_tokens=128_000,
            strategy=KeepRecentMessages(),
            token_counter=mock_counter,
        )
        assert compactor.compactions_performed == 0
        assert compactor.tokens_saved == 0


class TestContextCompactorMaybeCompact:
    """Tests for the maybe_compact method."""

    @pytest.mark.asyncio
    async def test_no_compaction_under_threshold(
        self,
        sample_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Messages under threshold should NOT be compacted."""
        compactor = ContextCompactor(
            max_context_tokens=1_000_000,  # Very high threshold
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(sample_messages)

        assert result == sample_messages  # Unchanged
        assert compactor.compactions_performed == 0

    @pytest.mark.asyncio
    async def test_compaction_over_threshold(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Messages over threshold should be compacted."""
        compactor = ContextCompactor(
            max_context_tokens=100,  # Very low to force compaction
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        result = await compactor.maybe_compact(long_messages)

        assert len(result) <= 5
        assert compactor.compactions_performed == 1

    @pytest.mark.asyncio
    async def test_compaction_updates_stats(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
    ):
        """Compaction should update statistics."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
        )

        original_tokens = mock_counter.count_messages(long_messages)
        result = await compactor.maybe_compact(long_messages)
        result_tokens = mock_counter.count_messages(result)

        assert compactor.compactions_performed == 1
        assert compactor.tokens_saved == original_tokens - result_tokens


class TestContextCompactorStats:
    """Tests for statistics methods."""

    def test_get_stats(self, mock_counter: MockTokenCounter):
        """Test stats retrieval."""
        compactor = ContextCompactor(
            max_context_tokens=100_000,
            strategy=KeepRecentMessages(),
            token_counter=mock_counter,
        )
        compactor.compactions_performed = 3
        compactor.tokens_saved = 5000

        stats = compactor.get_stats()

        assert stats["compactions_performed"] == 3
        assert stats["tokens_saved"] == 5000
        assert stats["max_context_tokens"] == 100_000
        assert stats["trigger_threshold"] == 80_000

    def test_reset_stats(self, mock_counter: MockTokenCounter):
        """Test stats reset."""
        compactor = ContextCompactor(
            max_context_tokens=100_000,
            strategy=KeepRecentMessages(),
            token_counter=mock_counter,
        )
        compactor.compactions_performed = 5
        compactor.tokens_saved = 10000

        compactor.reset_stats()

        assert compactor.compactions_performed == 0
        assert compactor.tokens_saved == 0


class TestContextCompactorVerbose:
    """Tests for verbose mode."""

    @pytest.mark.asyncio
    async def test_verbose_mode_prints(
        self,
        long_messages: list[MockMessage],
        mock_counter: MockTokenCounter,
        capsys,
    ):
        """Verbose mode should print debug info."""
        compactor = ContextCompactor(
            max_context_tokens=100,
            strategy=KeepRecentMessages(keep_count=5),
            token_counter=mock_counter,
            verbose=True,
        )

        await compactor.maybe_compact(long_messages)

        captured = capsys.readouterr()
        assert "[Compactor]" in captured.out
