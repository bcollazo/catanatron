"""
LLM-powered player implementations using PydanticAI.

This package provides LLM-based players that can use existing strategy players
(AlphaBetaPlayer, MCTSPlayer, etc.) as advisors while making decisions via LLM.
"""

from catanatron.players.llm.base import BaseLLMPlayer, CatanDependencies
from catanatron.players.llm.output_types import ActionOutput, ActionByIndex
from catanatron.players.llm.state_formatter import StateFormatter
from catanatron.players.llm.history import ConversationHistoryManager

__all__ = [
    "BaseLLMPlayer",
    "CatanDependencies",
    "ActionOutput",
    "ActionByIndex",
    "StateFormatter",
    "ConversationHistoryManager",
]
