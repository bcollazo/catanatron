"""
Catanatron player implementations.

This package provides various AI player implementations for playing Catan,
from simple random players to sophisticated LLM-powered agents.
"""

# Base players (from models.player)
from catanatron.models.player import (
    Player,
    SimplePlayer,
    RandomPlayer,
    HumanPlayer,
    Color,
)

# Strategy players
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer

# LLM players (optional - may not be available if pydantic-ai not installed)
try:
    from catanatron.players.llm_player import (
        PydanticAIPlayer,
        LLMAlphaBetaPlayer,
        LLMMCTSPlayer,
        LLMValuePlayer,
        LLMPlayer,  # Alias for PydanticAIPlayer
    )

    LLM_PLAYERS_AVAILABLE = True
except ImportError:
    LLM_PLAYERS_AVAILABLE = False
    PydanticAIPlayer = None
    LLMAlphaBetaPlayer = None
    LLMMCTSPlayer = None
    LLMValuePlayer = None
    LLMPlayer = None

__all__ = [
    # Base
    "Player",
    "SimplePlayer",
    "RandomPlayer",
    "HumanPlayer",
    "Color",
    # Strategy
    "WeightedRandomPlayer",
    "VictoryPointPlayer",
    "ValueFunctionPlayer",
    "AlphaBetaPlayer",
    "SameTurnAlphaBetaPlayer",
    "MCTSPlayer",
    "GreedyPlayoutsPlayer",
    # LLM (may be None if not available)
    "PydanticAIPlayer",
    "LLMAlphaBetaPlayer",
    "LLMMCTSPlayer",
    "LLMValuePlayer",
    "LLMPlayer",
    "LLM_PLAYERS_AVAILABLE",
]
