"""
Conversation history management for LLM players.

Manages message history across multiple decide() calls within a turn,
with automatic clearing at turn boundaries.
"""

from typing import List, Any


class ConversationHistoryManager:
    """
    Manages conversation history for LLM agent across multiple calls.

    In Catan, a single "turn" may involve multiple decide() calls:
    - Rolling dice
    - Building multiple things
    - Trading with other players
    - Playing development cards

    This manager:
    - Keeps conversation context within a turn for coherent decision making
    - Clears history at turn boundaries to prevent context overflow
    - Provides hooks for future extensions (summarization, etc.)
    """

    def __init__(self):
        """Initialize the history manager."""
        self.messages: List[Any] = []  # PydanticAI message objects
        self.current_turn: int = -1

    def is_new_turn(self, turn_number: int) -> bool:
        """
        Check if the given turn number represents a new turn.

        Args:
            turn_number: The current game turn number

        Returns:
            True if this is a different turn than the last recorded one
        """
        return turn_number != self.current_turn

    def set_turn(self, turn_number: int) -> None:
        """
        Set the current turn number.

        Args:
            turn_number: The turn number to track
        """
        self.current_turn = turn_number

    def clear(self) -> None:
        """Clear all stored messages."""
        self.messages = []

    def get_messages(self) -> List[Any]:
        """
        Get the current message history.

        Returns:
            List of PydanticAI message objects
        """
        return self.messages

    def update(self, new_messages: List[Any]) -> None:
        """
        Update the stored messages with new conversation.

        Args:
            new_messages: The complete message history from the latest agent run
        """
        self.messages = new_messages

    def add_message(self, message: Any) -> None:
        """
        Add a single message to the history.

        Args:
            message: A message to append
        """
        self.messages.append(message)

    @property
    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self.messages)

    def trim_to_last_n(self, n: int) -> None:
        """
        Keep only the last N messages.

        Useful for preventing context overflow in long games.

        Args:
            n: Number of recent messages to keep
        """
        if len(self.messages) > n:
            self.messages = self.messages[-n:]

    def __repr__(self) -> str:
        return f"ConversationHistoryManager(turn={self.current_turn}, messages={len(self.messages)})"
