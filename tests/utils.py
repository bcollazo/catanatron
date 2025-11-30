from catanatron.game import Game
from catanatron.models.enums import Action, ActionType


def build_initial_placements(
    game: Game,
    p0_actions=[0, (0, 1), 2, (1, 2)],
    p1_actions=[24, (24, 25), 26, (25, 26)],
):
    p0_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, p0_actions[0]))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, p0_actions[1]))

    game.execute(Action(p1_color, ActionType.BUILD_SETTLEMENT, p1_actions[0]))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, p1_actions[1]))
    game.execute(Action(p1_color, ActionType.BUILD_SETTLEMENT, p1_actions[2]))
    game.execute(Action(p1_color, ActionType.BUILD_ROAD, p1_actions[3]))

    game.execute(Action(p0_color, ActionType.BUILD_SETTLEMENT, p0_actions[2]))
    game.execute(Action(p0_color, ActionType.BUILD_ROAD, p0_actions[3]))


def advance_to_play_turn(game: Game):
    game.execute(Action(game.state.current_color(), ActionType.ROLL, None))
    while game.playable_actions[0].action_type in [
        ActionType.DISCARD,
        ActionType.MOVE_ROBBER,
    ]:
        game.execute(game.playable_actions[0])


def end_turn(game: Game):
    game.execute(Action(game.current_color(), ActionType.END_TURN, None))
