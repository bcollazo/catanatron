from collections import namedtuple
import sys
import logging

from rich.table import Table

from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

try:
    from catanatron_experimental.player_adapter import RustCompatibleMixin
    from catanatron_experimental.player_registry import register_rust_compatible
    RUST_ADAPTER_AVAILABLE = True
except ImportError:
    RUST_ADAPTER_AVAILABLE = False
    # Create dummy functions if the modules aren't available
    class RustCompatibleMixin:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    def register_rust_compatible(cls):
        return cls

# Try to see if Rust backend is available (we don't need RandomPlayer directly)
RUST_AVAILABLE = False
try:
    from catanatron_experimental.rust_bridge import is_rust_available
    RUST_AVAILABLE = is_rust_available()
    print(f"Rust backend available: {RUST_AVAILABLE}")
except ImportError as e:
    print(f"Error checking if Rust is available: {e}")

# Create a proxy for the Rust RandomPlayer
@register_rust_compatible
class RustRandomPlayerProxy(RustCompatibleMixin, RandomPlayer):
    """A proxy that uses the Python RandomPlayer but flags it for Rust optimization"""
    
    def __init__(self, color, name=None):
        self.logger = logging.getLogger(__name__)
        
        if not RUST_AVAILABLE:
            self.logger.warning("Rust backend not available, RustRandomPlayer will use Python implementation")
            
        # Store the name as an attribute since RandomPlayer doesn't handle it
        self.name = name if name is not None else f"RustRandom {color}"
        
        # Call the parent constructor with just the color parameter
        super().__init__(color)
        
        # Add a special attribute to mark this player for Rust optimization
        self.is_rust_player = True
        
        # Add Rust-specific color attribute
        # Map color to numeric value for Rust (RED=0, BLUE=1, ORANGE=2, WHITE=3)
        color_map = {
            "RED": 0,
            "BLUE": 1,
            "ORANGE": 2,
            "WHITE": 3,
        }
        
        # Extract the color name - handle both enum.name and string representations
        if hasattr(self.color, 'name'):
            color_name = str(self.color.name)
        else:
            color_name = str(self.color)
        
        if color_name in color_map:
            self._rust_color = color_map[color_name]
            self.logger.debug(f"Set _rust_color={self._rust_color} for player {self.name} with color {color_name}")
        else:
            # Default to RED if color not recognized
            self._rust_color = 0
            self.logger.warning(f"Unrecognized color {color_name}, defaulting to RED (0) for Rust")
            
        # Ensure the color attribute is accessible directly as needed
        if not hasattr(self, 'color') or self.color is None:
            self.logger.warning(f"Player {self.name} has no color attribute, setting explicitly")
            self.color = color
        
    def decide(self, game, playable_actions):
        """
        Overridden decide method that delegates to RandomPlayer's implementation
        This ensures compatibility with both Python and Rust backends
        """
        return super().decide(game, playable_actions)
    
    def __str__(self):
        return f"RustRandom({self.name}, _rust_color={self._rust_color})"
        
    def __repr__(self):
        return f"RustRandomPlayerProxy(color={self.color}, name='{self.name}', _rust_color={self._rust_color})"

# from catanatron_experimental.mcts_score_collector import (
#     MCTSScoreCollector,
#     MCTSPredictor,
# )
# from catanatron_experimental.machine_learning.players.reinforcement import (
#     QRLPlayer,
#     TensorRLPlayer,
#     VRLPlayer,
#     PRLPlayer,
# )
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    SameTurnAlphaBetaPlayer,
)
from catanatron.players.search import VictoryPointPlayer
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.machine_learning.players.playouts import (
    GreedyPlayoutsPlayer,
)

# from catanatron_experimental.machine_learning.players.online_mcts_dqn import (
#     OnlineMCTSDQNPlayer,
# )

# PLAYER_CLASSES = {
#     "O": OnlineMCTSDQNPlayer,
#     "S": ScikitPlayer,
#     "Y": MyPlayer,
#     # Used like: --players=V:path/to/model.model,T:path/to.model
#     "C": ForcePlayer,
#     "VRL": VRLPlayer,
#     "Q": QRLPlayer,
#     "P": PRLPlayer,
#     "T": TensorRLPlayer,
#     "D": DQNPlayer,
#     "CO": MCTSScoreCollector,
#     "COP": MCTSPredictor,
# }

# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])
CLI_PLAYERS = [
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "RR",
        "RustRandomPlayer", 
        "Faster random player for use with the Rust backend.", 
        RustRandomPlayerProxy,
    ),
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
]


def register_player(code):
    def decorator(player_class):
        CLI_PLAYERS.append(
            CliPlayer(
                code,
                player_class.__name__,
                player_class.__doc__,
                player_class,
            ),
        )

    return decorator


CUSTOM_ACCUMULATORS = []


def register_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    table.add_column("RUST COMPATIBLE", justify="center", style="green")
    
    for player in CLI_PLAYERS:
        rust_compatible = "✓" if player.code == "RR" else "✗"
        if player.code == "RR" and not RUST_AVAILABLE:
            rust_compatible = "✗ (Rust not installed)"
            
        table.add_row(player.code, player.name, player.description, rust_compatible)
    
    return table

def parse_player_arg(players_str):
    """Parse the players command line argument into a list of (code, params) tuples."""
    player_keys = players_str.split(",")
    result = []
    for key in player_keys:
        parts = key.split(":")
        code = parts[0]
        params = parts[1:] if len(parts) > 1 else []
        result.append((code, params))
    return result

def get_player(player_code):
    """Get the CLI player definition for the given code."""
    for cli_player in CLI_PLAYERS:
        if cli_player.code == player_code:
            return cli_player
    raise ValueError(f"Unknown player code: {player_code}")
