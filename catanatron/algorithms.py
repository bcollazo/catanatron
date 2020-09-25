from collections import defaultdict
import itertools
from typing import Iterable

from catanatron.models.actions import ActionType, Action
from catanatron.models.map import Port, Water
from catanatron.models.board import Board, Graph
from catanatron.models.player import Player, Color


def longest_road(board: Board, players: Iterable[Player], actions: Iterable[Action]):
    """
    For each connected subgraph (made by single-colored roads) find
    the longest path. Take max of all these candidates.

    Returns (color, path) tuple where
        color -- color of player whose longest path belongs.
        longest -- list of edges (all from a single color)
    """
    max_count = 0
    max_paths_by_player = dict()
    for player in players:
        for path in continuous_roads_by_player(board, player):
            count = len(path)
            if count < 5:
                continue
            if count > max_count:
                max_count = count
                max_paths_by_player = dict()
                max_paths_by_player[player.color] = path
            elif count == max_count:
                max_paths_by_player[player.color] = path

    if len(max_paths_by_player) == 0:
        return (None, None)

    # find first player that got to that point
    road_building_actions_by_candidates = list(
        filter(
            lambda a: a.action_type == ActionType.BUILD_ROAD
            and a.player.color in max_paths_by_player.keys(),
            actions,
        )
    )
    while len(max_paths_by_player) > 1:
        action = road_building_actions_by_candidates.pop()
        del max_paths_by_player[action.player.color]
    return max_paths_by_player.popitem()


def continuous_roads_by_player(board: Board, player: Player):
    paths = []
    components = board.find_connected_components(player.color)
    for component in components:
        paths.append(longest_acyclic_path(component))
    return paths


def longest_acyclic_path(subgraph: Graph):
    paths = []
    for start_node, connections in subgraph.items():
        # do DFS when reach leaf node, stop and add to paths
        paths_from_this_node = []
        agenda = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()

            able_to_navigate = False
            for edge, neighbor_node in subgraph[node].items():
                if edge not in path_thus_far:
                    agenda.insert(0, (neighbor_node, path_thus_far + [edge]))
                    able_to_navigate = True

            if not able_to_navigate:  # then it is leaf node
                paths_from_this_node.append(path_thus_far)

        paths.extend(paths_from_this_node)

    return max(paths, key=len)
