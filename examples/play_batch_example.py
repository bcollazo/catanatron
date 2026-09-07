from catanatron.cli.play import GameConfigOptions, OutputOptions, play_batch
from catanatron.registry import REGISTRY


players = REGISTRY.build_all("AB:2,AB:2")
output_options = OutputOptions(None, None, False, False, False)
game_config = GameConfigOptions(7, 10, "BASE")
play_batch(
    5,
    players,
    output_options,
    game_config,
    True,
)
