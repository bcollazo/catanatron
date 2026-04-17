"""
Classes to encode/decode catanatron classes to JSON format.
"""

import json
from enum import Enum

from catanatron.models.map import Water, Port, LandTile
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType
from catanatron.state_functions import get_longest_road_length, get_state_index
from catanatron.state import State
from catanatron.apply_action import apply_action


def longest_roads_by_player(state: State):
    result = dict()
    for color in state.colors:
        result[color.value] = get_longest_road_length(state, color)
    return result


def action_from_json(data) -> Action:
    color = Color[data[0]]
    action_type = ActionType[data[1]]
    if action_type == ActionType.BUILD_ROAD:
        action = Action(color, action_type, tuple(data[2]))
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = tuple(data[2])
        if len(resources) not in [1, 2]:
            raise ValueError("Year of Plenty action must have 1 or 2 resources")
        action = Action(color, action_type, resources)
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = data[2]
        coordinate = tuple(coordinate)
        victim = Color[victim] if victim else None
        action = Action(color, action_type, (coordinate, victim))
    elif action_type == ActionType.MARITIME_TRADE:
        value = tuple(data[2])
        action = Action(color, action_type, value)
    else:
        action = Action(color, action_type, data[2])
    return action


class GameEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
            return obj
        if isinstance(obj, Game):
            nodes = {}
            edges = {}
            for coordinate, tile in obj.state.board.map.tiles.items():
                for direction, node_id in tile.nodes.items():
                    building = obj.state.board.buildings.get(node_id, None)
                    color = None if building is None else building[0]
                    building_type = None if building is None else building[1]
                    nodes[node_id] = {
                        "id": node_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building_type),
                        "color": self.default(color),
                    }
                for direction, edge in tile.edges.items():
                    color = obj.state.board.roads.get(edge, None)
                    edge_id = tuple(sorted(edge))
                    edges[edge_id] = {
                        "id": edge_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "color": self.default(color),
                    }
            return {
                "tiles": [
                    {"coordinate": coordinate, "tile": self.default(tile)}
                    for coordinate, tile in obj.state.board.map.tiles.items()
                ],
                "adjacent_tiles": obj.state.board.map.adjacent_tiles,
                "nodes": nodes,
                "edges": list(edges.values()),
                "action_records": [self.default(a) for a in obj.state.action_records],
                "policy_debug_records": getattr(obj.state, "policy_debug_records", None),
                "player_state": obj.state.player_state,
                "colors": obj.state.colors,
                "bot_colors": list(
                    map(
                        lambda p: p.color, filter(lambda p: p.is_bot, obj.state.players)
                    )
                ),
                "bot_labels": {
                    p.color.value: getattr(p, "ui_label", "BOT")
                    for p in obj.state.players
                    if p.is_bot
                },
                "is_initial_build_phase": obj.state.is_initial_build_phase,
                "robber_coordinate": obj.state.board.robber_coordinate,
                "current_color": obj.state.current_color(),
                "current_prompt": obj.state.current_prompt,
                "current_playable_actions": obj.playable_actions,
                "longest_roads_by_player": longest_roads_by_player(obj.state),
                "winning_color": obj.winning_color(),
                "state_index": get_state_index(obj.state),
            }
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            return {
                "id": obj.id,
                "type": "PORT",
                "direction": self.default(obj.direction),
                "resource": self.default(obj.resource),
            }
        if isinstance(obj, LandTile):
            if obj.resource is None:
                return {"id": obj.id, "type": "DESERT"}
            return {
                "id": obj.id,
                "type": "RESOURCE_TILE",
                "resource": self.default(obj.resource),
                "number": obj.number,
            }
        return json.JSONEncoder.default(self, obj)


def fullstate_GameEncoder(game: Game):
    """Return a list of encoded board states, one after each action.

    The first element corresponds to the initial state *before* any actions
    have been applied. Each subsequent element is the result of replaying
    the next recorded action on a freshly reconstructed State that shares
    the same map, players, and victory-point configuration as ``game``.

    This function is intended for analysis / visualization tooling that
    needs a JSON-serializable snapshot of the board and player state at
    every step of the game, rather than only the final state plus the
    action log.
    """
    # Reconstruct an initial State that matches the logical start of the game.
    # We use the existing players, map and discard limit; randomness-dependent
    # events (dice rolls, dev-card draws, discards, robber steals) are replayed
    # deterministically from the stored ActionRecords via apply_action's
    # ``action_record`` parameter.
    initial_state = State(
        players=game.state.players,
        catan_map=game.state.board.map,
        discard_limit=game.state.discard_limit,
    )

    replay_game = Game(players=[], initialize=False)
    replay_game.seed = getattr(game, "seed", None)
    replay_game.id = game.id
    replay_game.vps_to_win = game.vps_to_win
    replay_game.state = initial_state
    # We don't need accurate playable_actions for encoding board snapshots;
    # leave it empty to avoid errors during initial build-phase replay.
    replay_game.playable_actions = []

    encoder = GameEncoder()
    snapshots = []

    # State before any actions.
    snapshots.append(encoder.default(replay_game))

    # Apply each recorded action using its stored result to ensure
    # deterministic replay, capturing the board/state after each step.
    for action_record in game.state.action_records:
        apply_action(replay_game.state, action_record.action, action_record)
        snapshots.append(encoder.default(replay_game))

    return snapshots
