from collections import defaultdict
from typing import Iterable, Set

from catanatron.models.actions import ActionType, Action
from catanatron.models.board import Board, STATIC_GRAPH
from catanatron.models.player import Color, Player
from catanatron.models.enums import DevelopmentCard


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
        for path in board.continuous_roads_by_player(player.color):
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
            and a.color in max_paths_by_player.keys(),
            actions,
        )
    )
    while len(max_paths_by_player) > 1:
        action = road_building_actions_by_candidates.pop()
        if action.color in max_paths_by_player:
            del max_paths_by_player[action.color]
    return max_paths_by_player.popitem()


# def continuous_roads_by_player(board: Board, color: Color):
#     paths = []
#     components = board.find_connected_components(color)
#     for component in components:
#         paths.append(longest_acyclic_path(board, component, color))
#     return paths


# def longest_acyclic_path(board: Board, node_set: Set[int], color: Color):
#     global STATIC_GRAPH

#     paths = []
#     for start_node in node_set:
#         # do DFS when reach leaf node, stop and add to paths
#         paths_from_this_node = []
#         agenda = [(start_node, [])]
#         while len(agenda) > 0:
#             node, path_thus_far = agenda.pop()

#             able_to_navigate = False
#             for neighbor_node in STATIC_GRAPH.neighbors(node):
#                 edge_color = board.get_edge_color((node, neighbor_node))
#                 if edge_color != color:
#                     continue

#                 neighbor_color = board.get_node_color(neighbor_node)
#                 if neighbor_color is not None and neighbor_color != color:
#                     continue  # enemy-owned, cant use this to navigate.
#                 edge = tuple(sorted((node, neighbor_node)))
#                 if edge not in path_thus_far:
#                     agenda.insert(0, (neighbor_node, path_thus_far + [edge]))
#                     able_to_navigate = True

#             if not able_to_navigate:  # then it is leaf node
#                 paths_from_this_node.append(path_thus_far)

#         paths.extend(paths_from_this_node)

#     return max(paths, key=len)


def largest_army(players: Iterable[Player], actions: Iterable[Action]):
    num_knights_to_players = defaultdict(set)
    for player in players:
        num_knight_played = player.played_development_cards.count(
            DevelopmentCard.KNIGHT
        )
        num_knights_to_players[num_knight_played].add(player.color)

    max_count = max(num_knights_to_players.keys())
    if max_count < 3:
        return (None, None)

    candidates = num_knights_to_players[max_count]
    knight_actions = list(
        filter(
            lambda a: a.action_type == ActionType.PLAY_KNIGHT_CARD
            and a.color in candidates,
            actions,
        )
    )
    while len(candidates) > 1:
        action = knight_actions.pop()
        if action.color in candidates:
            candidates.remove(action.color)

    return candidates.pop(), max_count
