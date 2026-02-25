import os
from collections import namedtuple

from rich.table import Table

from catanatron.models.player import Color, HumanPlayer, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer


# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])


# ============= LLM Player Factory Functions =============
# These are lazy-loaded to avoid import errors when pydantic-ai is not installed


def _get_default_model():
    """Get default LLM model from environment or use Claude."""
    return os.environ.get("CATAN_LLM_MODEL", "anthropic:claude-sonnet-4-20250514")


def create_llm_player(color, model=None):
    """Factory for pure LLM player."""
    from catanatron.players.llm_player import PydanticAIPlayer

    model = model or _get_default_model()
    return PydanticAIPlayer(color, model=model)


def create_llm_alphabeta_player(color, model=None, depth="2", prunning="False"):
    """Factory for LLM + AlphaBeta hybrid player."""
    from catanatron.players.llm_player import LLMAlphaBetaPlayer

    model = model or _get_default_model()
    return LLMAlphaBetaPlayer(
        color,
        model=model,
        depth=int(depth),
        prunning=prunning.lower() == "true",
    )


def create_llm_mcts_player(color, model=None, num_simulations="10"):
    """Factory for LLM + MCTS hybrid player."""
    from catanatron.players.llm_player import LLMMCTSPlayer

    model = model or _get_default_model()
    return LLMMCTSPlayer(
        color,
        model=model,
        num_simulations=int(num_simulations),
    )


def create_llm_value_player(color, model=None):
    """Factory for LLM + Value Function hybrid player."""
    from catanatron.players.llm_player import LLMValuePlayer

    model = model or _get_default_model()
    return LLMValuePlayer(color, model=model)
CLI_PLAYERS = [
    CliPlayer(
        "H", "HumanPlayer", "Human player, uses input() to get action.", HumanPlayer
    ),
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
    # LLM Players (require pydantic-ai, anthropic, or openai packages)
    CliPlayer(
        "LLM",
        "PydanticAIPlayer",
        "Pure LLM player using PydanticAI. Set ANTHROPIC_API_KEY or OPENAI_API_KEY env var. "
        "Optional param: MODEL (e.g., LLM:openai:gpt-4o)",
        create_llm_player,
    ),
    CliPlayer(
        "LLMAB",
        "LLMAlphaBetaPlayer",
        "LLM with AlphaBeta strategy advisor. Params: MODEL, DEPTH, PRUNNING. "
        "Example: LLMAB:anthropic:claude-sonnet-4-20250514:3:True or LLMAB::3 to use default model",
        create_llm_alphabeta_player,
    ),
    CliPlayer(
        "LLMM",
        "LLMMCTSPlayer",
        "LLM with MCTS strategy advisor. Params: MODEL, NUM_SIMULATIONS. "
        "Example: LLMM:openai:gpt-4o:20 or LLMM::20 to use default model",
        create_llm_mcts_player,
    ),
    CliPlayer(
        "LLMV",
        "LLMValuePlayer",
        "LLM with Value Function strategy advisor. Params: MODEL. "
        "Example: LLMV:anthropic:claude-sonnet-4-20250514 or just LLMV for default",
        create_llm_value_player,
    ),
]


def parse_cli_string(player_string):
    players = []
    player_keys = player_string.split(",")
    colors = [c for c in Color]
    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        for cli_player in CLI_PLAYERS:
            if cli_player.code == code:
                params = [colors[i]] + parts[1:]
                player = cli_player.import_fn(*params)
                players.append(player)
                break
    return players


def register_cli_player(code, player_class):
    CLI_PLAYERS.append(
        CliPlayer(
            code,
            player_class.__name__,
            player_class.__doc__,
            player_class,
        ),
    )


CUSTOM_ACCUMULATORS = []


def register_cli_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table
