import copy
import itertools
import operator as op
from functools import reduce
from enum import Enum
from collections import namedtuple

from catanatron.models.decks import ResourceDeck
from catanatron.models.board import BuildingType
from catanatron.models.enums import Resource


class ActionPrompt(Enum):
    BUILD_FIRST_SETTLEMENT = "BUILD_FIRST_SETTLEMENT"
    BUILD_SECOND_SETTLEMENT = "BUILD_SECOND_SETTLEMENT"
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"
    ROLL = "ROLL"
    PLAY_TURN = "PLAY_TURN"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"


class ActionType(Enum):
    ROLL = "ROLL"  # value is None or rolled value.
    MOVE_ROBBER = "MOVE_ROBBER"  # value is (coordinate, Player|None).
    DISCARD = "DISCARD"  # value is None or discarded cards

    # Building/Buying
    BUILD_FIRST_SETTLEMENT = "BUILD_FIRST_SETTLEMENT"  # value is node_id
    BUILD_SECOND_SETTLEMENT = "BUILD_SECOND_SETTLEMENT"  # value is node id
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"  # value is edge id
    BUILD_ROAD = "BUILD_ROAD"  # value is edge id
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is node id
    BUILD_CITY = "BUILD_CITY"  # value is node id
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"  # value is None

    # Dev Card Plays
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"  # value is (coordinate, player)
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"  # value is (edge_1, edge_2)

    # Trade
    MARITIME_TRADE = "MARITIME_TRADE"  # value is TradeOffer
    # TODO: Domestic trade. Im thinking should contain SUGGEST_TRADE, ACCEPT_TRADE actions...

    END_TURN = "END_TURN"


# TODO: Distinguish between PossibleAction and FinalizedAction?
Action = namedtuple("Action", ["player", "action_type", "value"])

TradeOffer = namedtuple(
    "TradeOffer", ["offering", "asking", "tradee"]
)  # offering and asking are Resource lists. tradee is a Player; None if bank


def monopoly_possible_actions(player):
    return [
        Action(player, ActionType.PLAY_MONOPOLY, card_type) for card_type in Resource
    ]


def year_of_plenty_possible_actions(player, resource_deck):
    possible_combinations = set()
    actions = []
    for first_card in Resource:
        for second_card in Resource:
            if (
                resource_deck.can_draw(1, first_card)
                and resource_deck.can_draw(1, second_card)
                and (second_card, first_card) not in possible_combinations
            ):
                possible_combinations.add((first_card, second_card))
                cards_selected = ResourceDeck()
                cards_selected.replenish(1, first_card)
                cards_selected.replenish(1, second_card)
                actions.append(
                    Action(player, ActionType.PLAY_YEAR_OF_PLENTY, cards_selected)
                )

    # TODO: If none of the combinations are possible due to shortages
    # in the deck, allow player to draw one card
    return actions


def road_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.road_cost())

    roads = board.get_player_buildings(player.color, BuildingType.ROAD)
    has_roads_available = len(roads) < 15

    if has_money and has_roads_available:
        buildable_edges = board.buildable_edges(player.color)
        return [
            Action(player, ActionType.BUILD_ROAD, edge.id) for edge in buildable_edges
        ]
    else:
        return []


def settlement_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.settlement_cost())

    settlements = board.get_player_buildings(player.color, BuildingType.SETTLEMENT)
    has_settlements_available = len(settlements) < 5

    if has_money and has_settlements_available:
        buildable_nodes = board.buildable_nodes(player.color)
        return [
            Action(player, ActionType.BUILD_SETTLEMENT, node.id)
            for node in buildable_nodes
        ]
    else:
        return []


def city_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.city_cost())

    cities = board.get_player_buildings(player.color, BuildingType.CITY)
    has_cities_available = len(cities) < 4

    if has_money and has_cities_available:
        settlements = board.get_player_buildings(player.color, BuildingType.SETTLEMENT)
        return [
            Action(player, ActionType.BUILD_CITY, node.id) for (node, _) in settlements
        ]
    else:
        return []


def robber_possibilities(player, board, players, is_dev_card):
    action_type = ActionType.PLAY_KNIGHT_CARD if is_dev_card else ActionType.MOVE_ROBBER

    players_by_color = {p.color: p for p in players}
    actions = []
    for coordinate, tile in board.resource_tiles():
        if coordinate == board.robber_coordinate:
            continue  # ignore. must move robber.

        # each tile can yield a (move-but-cant-steal) action or
        #   several (move-and-steal-from-x) actions.
        players_to_steal_from = set()
        for node_ref, node in tile.nodes.items():
            building = board.buildings.get(node)
            if building is not None:
                candidate = players_by_color[building.color]
                if (
                    candidate.resource_deck.num_cards() >= 1
                    and candidate.color != player.color  # can't play yourself
                ):
                    players_to_steal_from.add(candidate)

        if len(players_to_steal_from) == 0:
            actions.append(Action(player, action_type, (coordinate, None)))
        else:
            for p in players_to_steal_from:
                actions.append(Action(player, action_type, (coordinate, p)))

    return actions


def initial_settlement_possibilites(player, board, is_first):
    action_type = (
        ActionType.BUILD_FIRST_SETTLEMENT
        if is_first
        else ActionType.BUILD_SECOND_SETTLEMENT
    )
    buildable_nodes = board.buildable_nodes(player.color, initial_build_phase=True)
    return list(map(lambda node: Action(player, action_type, node.id), buildable_nodes))


def initial_road_possibilities(player, board, actions):
    # Must be connected to last settlement
    node_building_actions_by_player = filter(
        lambda action: action.player == player
        and action.action_type == ActionType.BUILD_FIRST_SETTLEMENT
        or action.action_type == ActionType.BUILD_SECOND_SETTLEMENT,
        actions,
    )
    last_settlement_node_id = list(node_building_actions_by_player)[-1].value
    last_settlement_node = board.get_node_by_id(last_settlement_node_id)

    buildable_edges = filter(
        lambda edge: last_settlement_node in edge.nodes,
        board.buildable_edges(player.color),
    )
    return list(
        map(
            lambda edge: Action(player, ActionType.BUILD_INITIAL_ROAD, edge.id),
            buildable_edges,
        )
    )


def discard_possibilities(player):
    hand = player.resource_deck.to_array()
    num_cards = player.resource_deck.num_cards()
    num_to_discard = num_cards // 2

    num_possibilities = ncr(num_cards, num_to_discard)
    if num_possibilities > 100:  # if too many, just take first N
        return [Action(player, ActionType.DISCARD, hand[:num_to_discard])]

    to_discard = itertools.combinations(hand, num_to_discard)
    return list(
        map(
            lambda combination: Action(player, ActionType.DISCARD, combination),
            to_discard,
        )
    )


def ncr(n, r):
    """n choose r. helper for discard_possibilities"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


# TODO: Remove possibilities if bank doesnt have them.
def maritime_trade_possibilities(player, bank_resource_cards, board):
    possibilities = []
    # 4:1 trade
    for resource in Resource:
        if player.resource_deck.count(resource) >= 4:
            for j_resource in Resource:
                if resource != j_resource:  # cant trade for same resource
                    trade_offer = TradeOffer([resource] * 4, [j_resource], None)
                    possibilities.append(
                        Action(player, ActionType.MARITIME_TRADE, trade_offer)
                    )

    port_resources = board.get_player_port_resources(player.color)
    for port_resource in port_resources:
        if port_resource is None:  # then has 3:1
            for resource in Resource:
                if player.resource_deck.count(resource) >= 3:
                    for j_resource in Resource:
                        if resource != j_resource:  # cant trade for same resource
                            trade_offer = TradeOffer([resource] * 3, [j_resource], None)
                            possibilities.append(
                                Action(player, ActionType.MARITIME_TRADE, trade_offer)
                            )
        else:  # has 2:1
            if player.resource_deck.count(port_resource) >= 2:
                for j_resource in Resource:
                    if port_resource != j_resource:  # cant trade for same resource
                        trade_offer = TradeOffer(
                            [port_resource] * 2, [j_resource], None
                        )
                        possibilities.append(
                            Action(player, ActionType.MARITIME_TRADE, trade_offer)
                        )

    return possibilities


def road_building_possibilities(player, board):
    """
    On purpose we _dont_ remove equivalent possibilities, since we need to be
    able to handle high branching degree anyway in AI.
    """
    first_edges = board.buildable_edges(player.color)
    possibilities = []
    for first_edge in first_edges:
        board_copy = copy.deepcopy(board)
        first_edge_copy = board_copy.get_edge_by_id(first_edge.id)
        board_copy.build_road(player.color, first_edge_copy)
        second_edges_copy = board_copy.buildable_edges(player.color)

        for second_edge_copy in second_edges_copy:
            second_edge = board.get_edge_by_id(second_edge_copy.id)
            possibilities.append(
                Action(player, ActionType.PLAY_ROAD_BUILDING, (first_edge, second_edge))
            )

    return possibilities
