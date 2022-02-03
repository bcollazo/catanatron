from collections import namedtuple

from rich.table import Table

from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_experimental.my_player import MyPlayer
from catanatron_experimental.mcts_score_collector import (
    MCTSScoreCollector,
    MCTSPredictor,
)
from catanatron_experimental.machine_learning.players.reinforcement import (
    QRLPlayer,
    TensorRLPlayer,
    VRLPlayer,
    PRLPlayer,
)
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)
from catanatron.players.search import (
    VictoryPointPlayer,
)
from catanatron_experimental.machine_learning.players.mcts import MCTSPlayer
from catanatron_experimental.machine_learning.players.playouts import (
    GreedyPlayoutsPlayer,
)
from catanatron_experimental.machine_learning.players.online_mcts_dqn import (
    OnlineMCTSDQNPlayer,
)

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
        "Y",
        "MyPlayer",
        "Uses catanatron_experimental/catanatron_experimental/my_player.py. "
        + "Edit this file with your own strategy and test it out here!",
        MyPlayer,
    ),
]


player_help_table = Table(title="Player Legend")
player_help_table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
player_help_table.add_column("PLAYER")
player_help_table.add_column("DESCRIPTION")
for player in CLI_PLAYERS:
    player_help_table.add_row(player.code, player.name, player.description)
