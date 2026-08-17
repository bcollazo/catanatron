"""
The engine-side adapter that feeds fair agents Observations.

The game loop always hands Player.decide a perfect-information Game.
PerspectivePlayer converts that Game into an Observation for the wrapped
agent on every decision, so the agent never perceives more than a real
player could observe.
"""

from catanatron.features import create_sample
from catanatron.models.enums import Action, ActionRecord, ActionType, DEVELOPMENT_CARDS
from catanatron.models.observation import Observation
from catanatron.models.player import Player
from catanatron.state_functions import (
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)


def _sanitize_record(record, observer_color):
    """Returns a copy of the record with hidden identities redacted.

    Full detail is retained for the observer's own records. For opponent
    records: development-card purchases are redacted (both channels), and
    stolen-card identities are redacted unless the observer was the victim.
    Discards are public per tournament convention.
    """
    action = record.action
    if action.color == observer_color:
        return record

    action_type = action.action_type
    if action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return ActionRecord(Action(action.color, action_type, None), None)
    elif action_type == ActionType.MOVE_ROBBER:
        robbed_color = action.value[1] if action.value is not None else None
        if robbed_color == observer_color:
            return record
        return ActionRecord(action, None)
    return record


def _sanitize_history(game, observer_color):
    """Sanitizes the full-truth action log for the given observer color."""
    return [
        _sanitize_record(record, observer_color) for record in game.state.action_records
    ]


def _build_public_state(game):
    """Projects the engine state down to a pure-data snapshot of public facts.

    Everything here is knowable with certainty by every player: the buildings
    map, roads, robber, longest-road holder, and per-color public counts. It is
    keyed absolutely (node/edge ids and Colors), unlike ``features`` which is
    keyed relative to the observing color. Opponent hand identities and actual
    victory points are never included.
    """
    state = game.state
    board = state.board

    roads = {}
    for edge, color in board.roads.items():
        node_a, node_b = edge
        if node_a < node_b:
            roads[(node_a, node_b)] = color

    players = {}
    for color in state.colors:
        key = player_key(state, color)
        player_state = state.player_state
        players[color] = {
            "public_vps": player_state[f"{key}_VICTORY_POINTS"],
            "has_army": player_state[f"{key}_HAS_ARMY"],
            "has_road": player_state[f"{key}_HAS_ROAD"],
            "longest_road_length": player_state[f"{key}_LONGEST_ROAD_LENGTH"],
            "roads_left": player_state[f"{key}_ROADS_AVAILABLE"],
            "settlements_left": player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
            "cities_left": player_state[f"{key}_CITIES_AVAILABLE"],
            "has_rolled": player_state[f"{key}_HAS_ROLLED"],
            "hand_resource_count": player_num_resource_cards(state, color),
            "hand_dev_count": player_num_dev_cards(state, color),
        }
        for card in DEVELOPMENT_CARDS:
            players[color][f"played_{card.lower()}"] = player_state[
                f"{key}_PLAYED_{card}"
            ]

    return {
        "board": {
            "buildings": dict(board.buildings),
            "roads": roads,
            "robber_coordinate": board.robber_coordinate,
            "longest_road_color": board.road_color,
            "longest_road_length": board.road_length,
        },
        "players": players,
    }


class PerspectivePlayer(Player):
    """Player that adapts an ObservationAgent into the perfect-info game loop.

    On every decision the given game is projected down to the agent's color
    as an Observation (features, sanitized history, trade state, and a
    structured public-state snapshot) and the agent's decide_observation is
    called. The agent never receives the Game.

    Args:
        agent (ObservationAgent): the fair agent to drive.
        is_bot (bool): whether this player is a bot. Defaults to True.
    """

    def __init__(self, agent, is_bot=True):
        self.agent = agent
        super().__init__(agent.color, is_bot)

    def decide(self, game, playable_actions):
        color = self.agent.color
        observation = Observation(
            color=color,
            features=create_sample(game, color),
            public_history=_sanitize_history(game, color),
            current_prompt=game.state.current_prompt,
            current_trade=game.state.current_trade,
            acceptees=game.state.acceptees,
            public_state=_build_public_state(game),
        )
        return self.agent.decide_observation(observation, playable_actions)

    def reset_state(self):
        self.agent.reset_state()
