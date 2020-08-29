from collections import defaultdict
import itertools

from catanatron.models.enums import BuildingType
from catanatron.models.map import Port, Water
from catanatron.models.board import BuildingType


def buildable_nodes(board, color, initial_placement=False):
    buildable = set()

    def is_buildable(node):
        """true if this and neighboring nodes are empty"""
        under_consideration = [node] + list(board.graph[node].values())
        has_building = map(
            lambda n: board.buildings.get(n) is None,
            under_consideration,
        )
        return all(has_building)

    # if initial-placement, iterate over non-water/port tiles, for each
    # of these nodes check if its a buildable node.
    if initial_placement:
        for (coordinate, tile) in board.tiles.items():
            if isinstance(tile, Port) or isinstance(tile, Water):
                continue

            for (noderef, node) in tile.nodes.items():
                if is_buildable(node):
                    buildable.add(node)

    # if not initial-placement, find all connected components. For each
    #   node in this connected subgraph, iterate checking buildability
    connected_components = find_connected_components(board, color)
    for subgraph in connected_components:
        for node in subgraph.keys():
            # by definition node is "connected", so only need to check buildable
            if is_buildable(node):
                buildable.add(node)

    return buildable


def buildable_edges(board, color):
    def is_buildable(edge):
        a_node, b_node = edge.nodes
        a_color = board.get_color(a_node)
        b_color = board.get_color(b_node)

        # check if buildable. buildable if nothing there, connected (one end_has_color)
        nothing_there = board.buildings.get(edge) is None
        one_end_has_color = board.is_color(a_node, color) or board.is_color(
            b_node, color
        )
        a_connected = any(
            [
                board.is_color(e, color)
                for e in board.graph.get(a_node).keys()
                if e != edge
            ]
        )
        b_connected = any(
            [
                board.is_color(e, color)
                for e in board.graph.get(b_node).keys()
                if e != edge
            ]
        )
        enemy_on_a = a_color is not None and a_color != color
        enemy_on_b = b_color is not None and b_color != color

        can_build = nothing_there and (
            one_end_has_color  # helpful for initial_placements
            or (a_connected and not enemy_on_a)
            or (b_connected and not enemy_on_b)
        )
        return can_build

    buildable = set()
    for edge in board.edges.values():
        if is_buildable(edge):
            buildable.add(edge)

    return buildable


def find_connected_components(board, color):
    """returns connected subgraphs for a given player

    algorithm goes like: find all nodes where color has buildings.
    start a BFS from any of these nodes, only following edges color owns,
    appending to subgraph and eliminating from agenda if builded there.
    repeat until list of settled_nodes is empty.

    Args:
        color (Color): [description]

    Returns:
        [list of self.graph-like objects]: connected subgraphs. subgraph
            might include nodes that color doesnt own (on the way and on ends),
            just to make it is "closed" and easier for buildable_nodes to operate.
    """
    settled_edges = set(
        edge for edge in board.edges.values() if board.is_color(edge, color)
    )

    subgraphs = []
    while len(settled_edges) > 0:
        tmp_subgraph = defaultdict(dict)

        # start bfs
        agenda = [settled_edges.pop()]
        visited = set()
        while len(agenda) > 0:
            edge = agenda.pop()
            visited.add(edge)
            if edge in settled_edges:
                settled_edges.remove(edge)

            # add to subgraph
            tmp_subgraph[edge.nodes[0]][edge] = edge.nodes[1]
            tmp_subgraph[edge.nodes[1]][edge] = edge.nodes[0]

            # edges to add to exploration are ones we are connected to.
            candidates = set()  # will be explorable "5-edge star" around edge
            a_color = board.get_color(edge.nodes[0])
            if a_color is None or a_color == color:  # enemy is not blocking
                for candidate_edge in board.graph[edge.nodes[0]].keys():
                    candidates.add(candidate_edge)
            b_color = board.get_color(edge.nodes[1])
            if b_color is None or b_color == color:  # enemy is not blocking
                for candidate_edge in board.graph[edge.nodes[1]].keys():
                    candidates.add(candidate_edge)

            for candidate_edge in candidates:
                if (
                    candidate_edge not in visited
                    and candidate_edge not in agenda
                    and candidate_edge != edge
                    and board.is_color(candidate_edge, color)
                ):
                    agenda.append(candidate_edge)

        subgraphs.append(dict(tmp_subgraph))
    return subgraphs


def longest_road(board):
    """
    For each connected subgraph (made by single-colored roads) find
    the longest path. Take max of all these candidates.

    Returns (path, color) tuple where
    longest -- list of edges (all from a single color)
    color -- color of player whose longest path belongs.
    """
    max_count = 0
    max_players = set([])
    for color in board.players:
        components = find_connected_components(board, color)
        for component in components:
            count = longest_acyclic_path(board, component)
            if count < 5:
                continue
            if count > max_count:
                max_count = count
                max_players = set([color])
            elif count == max_count:
                max_players.add(color)

    print(max_players)
    if len(max_players) == 0:
        return None

    # find first player that got to that point
    road_building_actions_by_winners = list(
        filter(
            lambda a: a.building.building_type == BuildingType.ROAD
            and a.color in max_players,
            board.actions,
        )
    ).reverse()
    while len(max_players) > 1:
        for action in road_building_actions_by_winners:
            max_players.remove(action.color)
    return max_players.pop()


def longest_acyclic_path(board, subgraph):
    """Brute-force, tries all edge-orderings, discards non-paths, and takes max

    Args:
        board ([type]): [description]
        subgraph ([type]): { node => {edge => node}}
    """
    # Consider all orderings of owned edges participating in this subgraph.
    all_edges = set()
    for node, connections in subgraph.items():
        for edge, b_node in connections.items():
            all_edges.add(edge)
    permutations = []
    for i in range(len(all_edges)):
        permutations.extend(list(itertools.permutations(all_edges, i + 1)))

    # Eliminate all that are not a path.
    paths = []

    def get_connection_node(a_edge, b_edge):
        if a_edge.nodes[0] == b_edge.nodes[0]:
            return a_edge.nodes[0]
        elif a_edge.nodes[1] == b_edge.nodes[0]:
            return a_edge.nodes[1]
        elif a_edge.nodes[0] == b_edge.nodes[1]:
            return a_edge.nodes[0]
        elif a_edge.nodes[1] == b_edge.nodes[1]:
            return a_edge.nodes[1]
        else:
            return None

    for permutation in permutations:
        if len(permutation) == 1:
            paths.append(permutation)
            continue

        first_edge = permutation[0]
        second_edge = permutation[1]
        last_edge = first_edge
        next_connection_node = get_connection_node(first_edge, second_edge)
        if next_connection_node is None:
            continue  # not a path, the first two are not connected

        is_path = True  # assume the best
        for edge in permutation[1:]:
            connection_node = get_connection_node(last_edge, edge)
            if connection_node == next_connection_node:
                next_connection_node = list(
                    filter(lambda n: n != connection_node, edge.nodes)
                )[0]
                last_edge = edge
            else:
                is_path = False
                break

        if is_path:
            paths.append(permutation)

    return len(max(paths, key=len))


def largest_army(board):
    """
    Count the number of robbers activated by each player.

    Return (largest, color) tuple where
    largest -- number of knights activated
    color -- color of player who owns largest army
    """
    raise NotImplemented
