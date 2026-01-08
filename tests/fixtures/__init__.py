"""Test fixtures for SDK-specific message types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

MessageT = TypeVar("MessageT")


# =============================================================================
# Fixture Protocol - All SDK fixtures implement this
# =============================================================================


@dataclass
class FixtureSet:
    """Container for a set of SDK-specific test fixtures."""

    sdk_name: str
    sample_messages: list
    long_messages: list
    mixed_messages: list
    token_counter: object


class FixtureProvider(Protocol):
    """Protocol for SDK fixture providers."""

    def get_fixtures(self) -> FixtureSet: ...
