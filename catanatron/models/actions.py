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
    """
    Action types are associated with a "value" that can be seen as the "params"
    of such action. They usually hold None for to-be-defined values by the
    execution of the action. After execution, the Actions will be hydrated
    so that they can be used in reproducing a game.
    """

    ROLL = "ROLL"  # value is None|(int, int)
    MOVE_ROBBER = "MOVE_ROBBER"  # value is (coordinate, Color|None, None|Resource)
    DISCARD = "DISCARD"  # value is None|Resource[]

    # Building/Buying
    BUILD_FIRST_SETTLEMENT = "BUILD_FIRST_SETTLEMENT"  # value is node_id
    BUILD_SECOND_SETTLEMENT = "BUILD_SECOND_SETTLEMENT"  # value is node_id
    BUILD_INITIAL_ROAD = "BUILD_INITIAL_ROAD"  # value is edge_id
    BUILD_ROAD = "BUILD_ROAD"  # value is edge_id
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is node_id
    BUILD_CITY = "BUILD_CITY"  # value is node_id
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"  # value is None|DevelopmentCard

    # Dev Card Plays
    PLAY_KNIGHT_CARD = (
        "PLAY_KNIGHT_CARD"  # value is (coordinate, Color|None, None|Resource)
    )
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"  # value is [Resource, Resource]
    PLAY_MONOPOLY = "PLAY_MONOPOLY"  # value is Resource
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"  # value is (edge_id1, edge_id2)

    # Trade
    MARITIME_TRADE = "MARITIME_TRADE"  # value is TradeOffer(offering=Resource[], asking=Resource, tradee=None)
    # TODO: Domestic trade. Im thinking should contain SUGGEST_TRADE, ACCEPT_TRADE actions...

    END_TURN = "END_TURN"  # value is None


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
    actions = []
    resource_list = list(Resource)
    for i, first_card in enumerate(resource_list):
        for j in range(i, len(resource_list)):
            second_card = resource_list[i]  # doing it this way to not repeat
            if resource_deck.can_draw(1, first_card) and resource_deck.can_draw(
                1, second_card
            ):
                actions.append(
                    Action(
                        player,
                        ActionType.PLAY_YEAR_OF_PLENTY,
                        [first_card, second_card],
                    )
                )

    # TODO: If none of the combinations are possible due to shortages
    # in the deck, allow player to draw one card
    return actions


def road_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.road_cost())
    has_roads_available = player.roads_available > 0

    if has_money and has_roads_available:
        buildable_edge_ids = board.buildable_edge_ids(player.color)
        return [
            Action(player, ActionType.BUILD_ROAD, edge_id)
            for edge_id in buildable_edge_ids
        ]
    else:
        return []


def settlement_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.settlement_cost())
    has_settlements_available = player.settlements_available > 0

    if has_money and has_settlements_available:
        buildable_node_ids = board.buildable_node_ids(player.color)
        return [
            Action(player, ActionType.BUILD_SETTLEMENT, node_id)
            for node_id in buildable_node_ids
        ]
    else:
        return []


def city_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.city_cost())
    has_cities_available = player.cities_available > 0

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
        to_steal_from = set()  # set of player_indexs
        for _, node in tile.nodes.items():
            building = board.buildings.get(node)
            if building is not None:
                candidate = players_by_color[building.color]
                if (
                    candidate.resource_deck.num_cards() >= 1
                    and candidate.color != player.color  # can't play yourself
                ):
                    to_steal_from.add(candidate.color)

        if len(to_steal_from) == 0:
            actions.append(Action(player, action_type, (coordinate, None, None)))
        else:
            for color in to_steal_from:
                actions.append(Action(player, action_type, (coordinate, color, None)))

    return actions


def initial_settlement_possibilites(player, board, is_first):
    action_type = (
        ActionType.BUILD_FIRST_SETTLEMENT
        if is_first
        else ActionType.BUILD_SECOND_SETTLEMENT
    )
    buildable_node_ids = board.buildable_node_ids(
        player.color, initial_build_phase=True
    )
    return list(
        map(lambda node_id: Action(player, action_type, node_id), buildable_node_ids)
    )


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

    buildable_edge_ids = filter(
        lambda edge_id: last_settlement_node in board.get_edge_by_id(edge_id).nodes,
        board.buildable_edge_ids(player.color),
    )
    return list(
        map(
            lambda edge_id: Action(player, ActionType.BUILD_INITIAL_ROAD, edge_id),
            buildable_edge_ids,
        )
    )


def discard_possibilities(player):
    return [Action(player, ActionType.DISCARD, None)]
    # TODO: Be robust to high dimensionality of DISCARD
    # hand = player.resource_deck.to_array()
    # num_cards = player.resource_deck.num_cards()
    # num_to_discard = num_cards // 2

    # num_possibilities = ncr(num_cards, num_to_discard)
    # if num_possibilities > 100:  # if too many, just take first N
    #     return [Action(player, ActionType.DISCARD, hand[:num_to_discard])]

    # to_discard = itertools.combinations(hand, num_to_discard)
    # return list(
    #     map(
    #         lambda combination: Action(player, ActionType.DISCARD, combination),
    #         to_discard,
    #     )
    # )


def ncr(n, r):
    """n choose r. helper for discard_possibilities"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def maritime_trade_possibilities(player, bank, board):
    possibilities = []
    # 4:1 trade
    for resource in Resource:
        if player.resource_deck.count(resource) >= 4:
            for j_resource in Resource:
                # cant trade for same resource, and bank must have enough
                if resource != j_resource and bank.count(j_resource) > 0:
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
                        # cant trade for same resource, and bank must have enough
                        if resource != j_resource and bank.count(j_resource) > 0:
                            trade_offer = TradeOffer([resource] * 3, [j_resource], None)
                            possibilities.append(
                                Action(player, ActionType.MARITIME_TRADE, trade_offer)
                            )
        else:  # has 2:1
            if player.resource_deck.count(port_resource) >= 2:
                for j_resource in Resource:
                    # cant trade for same resource, and bank must have enough
                    if port_resource != j_resource and bank.count(j_resource) > 0:
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
    first_edge_ids = board.buildable_edge_ids(player.color)
    possibilities = []
    for first_edge_id in first_edge_ids:
        board_copy = copy.deepcopy(board)
        first_edge_copy = board_copy.get_edge_by_id(first_edge_id)
        board_copy.build_road(player.color, first_edge_copy)
        second_edge_ids_copy = board_copy.buildable_edge_ids(player.color)

        for second_edge_id_copy in second_edge_ids_copy:
            possibilities.append(
                Action(
                    player,
                    ActionType.PLAY_ROAD_BUILDING,
                    (first_edge_id, second_edge_id_copy),
                )
            )

    return possibilities
