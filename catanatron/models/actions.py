import operator as op
from functools import reduce
from enum import Enum
from collections import namedtuple

from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import Resource, BuildingType


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
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"  # value is (Resource, Resource)
    PLAY_MONOPOLY = "PLAY_MONOPOLY"  # value is Resource
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"  # value is (edge_id1, edge_id2)

    # Trade
    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = "MARITIME_TRADE"

    # TODO: Domestic trade. Im thinking should contain SUGGEST_TRADE, ACCEPT_TRADE actions...

    END_TURN = "END_TURN"  # value is None


def action_repr(self):
    return f"Action({self.color.value} {self.action_type.value} {self.value})"


# TODO: Distinguish between PossibleAction and FinalizedAction?
Action = namedtuple("Action", ["color", "action_type", "value"])
Action.__repr__ = action_repr


def monopoly_possible_actions(player):
    return [
        Action(player.color, ActionType.PLAY_MONOPOLY, card_type)
        for card_type in Resource
    ]


def year_of_plenty_possible_actions(player, resource_deck: ResourceDeck):
    resource_list = list(Resource)

    options = set()
    for i, first_card in enumerate(resource_list):
        for j in range(i, len(resource_list)):
            second_card = resource_list[j]  # doing it this way to not repeat
            to_draw = ResourceDeck.from_array([first_card, second_card])
            if resource_deck.includes(to_draw):
                options.add((first_card, second_card))
            else:  # try allowing player select 1 card only.
                if resource_deck.can_draw(1, first_card):
                    options.add((first_card,))
                if resource_deck.can_draw(1, second_card):
                    options.add((second_card,))

    return list(
        map(
            lambda cards: Action(
                player.color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(cards)
            ),
            options,
        )
    )


def road_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.road_cost())
    has_roads_available = player.roads_available > 0

    if has_money and has_roads_available:
        buildable_edges = board.buildable_edges(player.color)
        return [
            Action(player.color, ActionType.BUILD_ROAD, edge)
            for edge in buildable_edges
        ]
    else:
        return []


def settlement_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.settlement_cost())
    has_settlements_available = player.settlements_available > 0

    if has_money and has_settlements_available:
        buildable_node_ids = board.buildable_node_ids(player.color)
        return [
            Action(player.color, ActionType.BUILD_SETTLEMENT, node_id)
            for node_id in buildable_node_ids
        ]
    else:
        return []


def city_possible_actions(player):
    has_money = player.resource_deck.includes(ResourceDeck.city_cost())
    has_cities_available = player.cities_available > 0

    if has_money and has_cities_available:
        return [
            Action(player.color, ActionType.BUILD_CITY, node_id)
            for node_id in player.buildings[BuildingType.SETTLEMENT]
        ]
    else:
        return []


def robber_possibilities(player, board, players, is_dev_card):
    action_type = ActionType.PLAY_KNIGHT_CARD if is_dev_card else ActionType.MOVE_ROBBER

    players_by_color = {p.color: p for p in players}
    actions = []
    for coordinate, tile in board.map.resource_tiles:
        if coordinate == board.robber_coordinate:
            continue  # ignore. must move robber.

        # each tile can yield a (move-but-cant-steal) action or
        #   several (move-and-steal-from-x) actions.
        to_steal_from = set()  # set of player_indexs
        for _, node_id in tile.nodes.items():
            building = board.buildings.get(node_id, None)
            if building is not None:
                candidate = players_by_color[building[0]]
                if (
                    candidate.resource_deck.num_cards() >= 1
                    and candidate.color != player.color  # can't play yourself
                ):
                    to_steal_from.add(candidate.color)

        if len(to_steal_from) == 0:
            actions.append(Action(player.color, action_type, (coordinate, None, None)))
        else:
            for color in to_steal_from:
                actions.append(
                    Action(player.color, action_type, (coordinate, color, None))
                )

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
        map(
            lambda node_id: Action(player.color, action_type, node_id),
            buildable_node_ids,
        )
    )


def initial_road_possibilities(player, board, actions):
    # Must be connected to last settlement
    node_building_actions_by_player = filter(
        lambda action: action.color == player.color
        and action.action_type == ActionType.BUILD_FIRST_SETTLEMENT
        or action.action_type == ActionType.BUILD_SECOND_SETTLEMENT,
        actions,
    )
    last_settlement_node_id = list(node_building_actions_by_player)[-1].value

    buildable_edges = filter(
        lambda edge: last_settlement_node_id in edge,
        board.buildable_edges(player.color),
    )
    return [
        Action(player.color, ActionType.BUILD_INITIAL_ROAD, edge)
        for edge in buildable_edges
    ]


def discard_possibilities(player):
    return [Action(player.color, ActionType.DISCARD, None)]
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
                    trade_offer = tuple([resource] * 4 + [j_resource])
                    possibilities.append(
                        Action(player.color, ActionType.MARITIME_TRADE, trade_offer)
                    )

    port_resources = board.get_player_port_resources(player.color)
    for port_resource in port_resources:
        if port_resource is None:  # then has 3:1
            for resource in Resource:
                if player.resource_deck.count(resource) >= 3:
                    for j_resource in Resource:
                        # cant trade for same resource, and bank must have enough
                        if resource != j_resource and bank.count(j_resource) > 0:
                            trade_offer = tuple([resource] * 3 + [None, j_resource])
                            possibilities.append(
                                Action(
                                    player.color, ActionType.MARITIME_TRADE, trade_offer
                                )
                            )
        else:  # has 2:1
            if player.resource_deck.count(port_resource) >= 2:
                for j_resource in Resource:
                    # cant trade for same resource, and bank must have enough
                    if port_resource != j_resource and bank.count(j_resource) > 0:
                        trade_offer = tuple(
                            [port_resource] * 2 + [None, None, j_resource]
                        )
                        possibilities.append(
                            Action(player.color, ActionType.MARITIME_TRADE, trade_offer)
                        )

    return possibilities


def road_building_possibilities(player, board):
    """
    We remove equivalent possibilities, to simplify branching factor.
    """
    first_edges = board.buildable_edges(player.color)
    possibilities = set()
    for first_edge in first_edges:
        board_copy = board.copy()
        board_copy.build_road(player.color, first_edge)

        second_edges_copy = board_copy.buildable_edges(player.color)
        for second_edge_copy in second_edges_copy:
            possibilities.add((first_edge, second_edge_copy))

    # Remove duplicate possibilities (when second road doesnt depend on first).
    dedupped = set()
    for (first, second) in possibilities:
        if second in first_edges:  # deduppable-pair
            dedupped.add(tuple(sorted((first, second))))
        else:
            dedupped.add((first, second))

    return list(
        map(
            lambda possibility: Action(
                player.color,
                ActionType.PLAY_ROAD_BUILDING,
                (possibility[0], possibility[1]),
            ),
            dedupped,
        )
    )
