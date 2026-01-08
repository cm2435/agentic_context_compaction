"""Pytest configuration and fixtures.

This module provides:
1. Mock message types for testing without SDK dependencies
2. Parametrized fixtures that run tests across all SDK fixture sets
3. Common test utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import pytest

MessageT = TypeVar("MessageT")


# =============================================================================
# Mock Types for SDK-Independent Testing
# =============================================================================


@dataclass
class MockMessage:
    """
    Mock message type for testing generic compaction.

    Simulates the structure of SDK messages without requiring
    actual SDK dependencies.
    """

    role: str
    content: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockMessage):
            return NotImplemented
        return self.role == other.role and self.content == other.content

    def __hash__(self) -> int:
        return hash((self.role, self.content))


@dataclass
class MockTokenCounter:
    """
    Mock token counter for testing.

    Uses character-based approximation (4 chars per token).
    """

    chars_per_token: float = 4.0
    overhead_per_message: int = 4

    def count_messages(self, messages: list[MockMessage]) -> int:
        """Count total tokens across all messages."""
        return sum(self.count_single(msg) for msg in messages)

    def count_single(self, message: MockMessage) -> int:
        """Count tokens in a single message."""
        text_tokens = int(len(message.content) / self.chars_per_token)
        return text_tokens + self.overhead_per_message


# =============================================================================
# Base Fixtures (Mock Types)
# =============================================================================


@pytest.fixture
def mock_counter() -> MockTokenCounter:
    """Provide a mock token counter."""
    return MockTokenCounter()


@pytest.fixture
def sample_messages() -> list[MockMessage]:
    """Basic conversation (~11 messages) for testing."""
    return [
        MockMessage(role="system", content="You are a helpful assistant."),
        MockMessage(role="user", content="Hello, how are you?"),
        MockMessage(role="assistant", content="I'm doing well, thank you!"),
        MockMessage(role="user", content="Can you help me with Python?"),
        MockMessage(role="assistant", content="Of course! I'd be happy to help."),
        MockMessage(role="user", content="How do I read a file?"),
        MockMessage(role="assistant", content="To read a file, use open()..."),
        MockMessage(role="user", content="What about writing?"),
        MockMessage(role="assistant", content="For writing, use open('w')..."),
        MockMessage(role="user", content="Thanks!"),
        MockMessage(role="assistant", content="You're welcome!"),
    ]


@pytest.fixture
def long_messages(sample_messages: list[MockMessage]) -> list[MockMessage]:
    """Extended message history (~51 messages) for compaction testing."""
    messages = sample_messages.copy()
    for i in range(20):
        messages.append(MockMessage(role="user", content=f"Question {i}: How does feature X work?"))
        messages.append(
            MockMessage(role="assistant", content=f"Answer {i}: Feature X works by doing Y and Z.")
        )
    return messages


@pytest.fixture
def mixed_messages() -> list[MockMessage]:
    """
    Mixed message types simulating multimodal content.

    In real SDK tests, these would include images, tool calls, etc.
    Here we use content markers to simulate.
    """
    return [
        MockMessage(role="system", content="You are a coding assistant."),
        MockMessage(role="user", content="[IMAGE: screenshot.png] What's wrong here?"),
        MockMessage(role="assistant", content="I see an error in line 5..."),
        MockMessage(
            role="assistant",
            content="[TOOL_CALL: read_file(path='main.py')]",
        ),
        MockMessage(
            role="tool",
            content="[TOOL_RESULT: def main():\\n    print('hello')]",
        ),
        MockMessage(role="assistant", content="The code looks fine. Let me check more."),
        MockMessage(
            role="assistant",
            content="[THINKING] Analyzing the error message...",
        ),
        MockMessage(role="assistant", content="The issue is with the import statement."),
        MockMessage(role="user", content="Can you fix it?"),
        MockMessage(
            role="assistant",
            content="[TOOL_CALL: edit_file(path='main.py', content='...')]",
        ),
        MockMessage(role="tool", content="[TOOL_RESULT: File updated successfully]"),
        MockMessage(role="assistant", content="Done! I've fixed the import."),
        MockMessage(role="user", content="[IMAGE: result.png] Does this look right?"),
        MockMessage(role="assistant", content="Yes, the output looks correct now!"),
    ]


# =============================================================================
# SDK-Specific Fixture Parametrization
# =============================================================================

# We use pytest_generate_tests to dynamically generate tests for each SDK
# when the SDK is available. For now, we use mock types.


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "sdk(name): mark test to run with specific SDK fixtures")


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def empty_messages() -> list[MockMessage]:
    """Empty message list for edge case testing."""
    return []


@pytest.fixture
def single_message() -> list[MockMessage]:
    """Single message for edge case testing."""
    return [MockMessage(role="user", content="Hello!")]


@pytest.fixture
def large_message() -> MockMessage:
    """A single message with very long content."""
    return MockMessage(role="assistant", content="x" * 10000)
