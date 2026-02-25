"""
Concrete LLM-powered player implementations.

Provides ready-to-use player classes that combine LLM decision making
with various strategy advisors (AlphaBeta, MCTS, Value Function).
"""

from typing import Literal, Optional, List

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import Action

from catanatron.players.llm.base import BaseLLMPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.value import ValueFunctionPlayer


class PydanticAIPlayer(BaseLLMPlayer):
    """
    Pure LLM player without a strategy advisor.

    Makes decisions entirely based on LLM reasoning using the available tools.
    Good for testing pure LLM capabilities or when you want maximum flexibility.

    Example:
        player = PydanticAIPlayer(Color.RED, model="anthropic:claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        color: Color,
        model: str = "anthropic:claude-sonnet-4-20250514",
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
    ):
        """
        Initialize a pure LLM player.

        Args:
            color: Player color
            model: LLM model string (e.g., "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o")
            output_mode: "index" for fast mode, "structured" for detailed logging
            is_bot: Whether this is a bot player (always True for LLM)
        """
        super().__init__(color, model=model, output_mode=output_mode, is_bot=is_bot)

    def _get_strategy_recommendation(
        self, game: Game, playable_actions: List[Action]
    ) -> tuple[Optional[Action], Optional[str]]:
        """Pure LLM player has no strategy advisor."""
        return None, None


class LLMAlphaBetaPlayer(BaseLLMPlayer, AlphaBetaPlayer):
    """
    LLM player with AlphaBeta search as strategy advisor.

    The AlphaBeta algorithm looks ahead multiple moves and provides
    a recommendation, which the LLM can follow or override.

    This combines the tactical depth of tree search with the
    strategic reasoning of an LLM.

    Example:
        player = LLMAlphaBetaPlayer(
            Color.RED,
            model="anthropic:claude-sonnet-4-20250514",
            depth=2,
            prunning=True
        )
    """

    def __init__(
        self,
        color: Color,
        model: str = "anthropic:claude-sonnet-4-20250514",
        depth: int = 2,
        prunning: bool = False,
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
    ):
        """
        Initialize an LLM player with AlphaBeta advisor.

        Args:
            color: Player color
            model: LLM model string
            depth: Search depth for AlphaBeta (higher = slower but better recommendations)
            prunning: Whether to use alpha-beta pruning
            timeout: Timeout in seconds for LLM calls (default: 120.0)
            tool_calls_limit: Overall tool call limit per decision (default: 10)
            output_mode: Action output format
            is_bot: Whether this is a bot player
        """
        # Initialize AlphaBeta first (it will call Player.__init__)
        super().__init__(
            color,
            model=model,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            depth=depth,
            prunning=prunning,
        )

    def __repr__(self) -> str:
        base = AlphaBetaPlayer.__repr__(self)
        return f"LLM{base}[{self.model}]"


class LLMMCTSPlayer(BaseLLMPlayer, MCTSPlayer):
    """
    LLM player with Monte Carlo Tree Search as strategy advisor.

    MCTS runs simulations to estimate the value of each action,
    providing a probabilistic recommendation to the LLM.

    Good for situations where deep tactical analysis is important.

    Example:
        player = LLMMCTSPlayer(
            Color.RED,
            model="anthropic:claude-sonnet-4-20250514",
            num_simulations=20
        )
    """

    def __init__(
        self,
        color: Color,
        model: str = "anthropic:claude-sonnet-4-20250514",
        num_simulations: int = 10,
        prunning: bool = False,
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        output_mode: Literal["index", "structured"] = "index",
        is_bot: bool = True,
    ):
        """
        Initialize an LLM player with MCTS advisor.

        Args:
            color: Player color
            model: LLM model string
            num_simulations: Number of MCTS simulations (higher = slower but better)
            prunning: Whether to use action pruning
            timeout: Timeout in seconds for LLM calls (default: 120.0)
            tool_calls_limit: Overall tool call limit per decision (default: 10)
            output_mode: Action output format
            is_bot: Whether this is a bot player
        """
        # Initialize MCTS first
        super().__init__(
            color,
            model=model,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            num_simulations=num_simulations,
            prunning=prunning,
        )

    def __repr__(self) -> str:
        base = MCTSPlayer.__repr__(self)
        return f"LLM{base}[{self.model}]"


class LLMValuePlayer(BaseLLMPlayer, ValueFunctionPlayer):
    """
    LLM player with heuristic value function as strategy advisor.

    The value function provides a fast, greedy recommendation based on
    hand-crafted heuristics. Less computationally expensive than
    AlphaBeta or MCTS.

    Good for faster games or when compute is limited.

    Example:
        player = LLMValuePlayer(Color.RED, model="anthropic:claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        color: Color,
        model: str = "anthropic:claude-sonnet-4-20250514",
        value_fn_builder_name: Optional[str] = None,
        output_mode: Literal["index", "structured"] = "index",
        timeout: Optional[float] = 120.0,
        tool_calls_limit: int = 10,
        is_bot: bool = True,
    ):
        """
        Initialize an LLM player with Value Function advisor.

        Args:
            color: Player color
            model: LLM model string
            value_fn_builder_name: Which value function to use ("C" for contender, None for base)
            output_mode: Action output format
            timeout: Timeout in seconds for LLM calls (default: 120.0)
            tool_calls_limit: Overall tool call limit per decision (default: 10)
            is_bot: Whether this is a bot player
        """
        # Initialize ValueFunctionPlayer first
        super().__init__(
            color,
            model=model,
            output_mode=output_mode,
            is_bot=is_bot,
            timeout=timeout,
            tool_calls_limit=tool_calls_limit,
            value_fn_builder_name=value_fn_builder_name,
        )

    def __repr__(self) -> str:
        base = ValueFunctionPlayer.__str__(self)
        return f"LLM{base}[{self.model}]"


# Aliases for convenience
LLMPlayer = PydanticAIPlayer
