from enum import Enum
from collections import namedtuple

from catanatron.models.decks import ResourceDeck
from catanatron.models.board import BuildingType
from catanatron.models.enums import Resource


class ActionType(Enum):
    ROLL = "ROLL"  # value is None or rolled value.
    MOVE_ROBBER = "MOVE_ROBBER"  # value is (coordinate, Player|None).
    DISCARD = "DISCARD"  # value is None or discarded cards

    # Building/Buying
    BUILD_ROAD = "BUILD_ROAD"  # value is edge
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value is node
    BUILD_CITY = "BUILD_CITY"
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"

    # Dev Card Plays
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"  # value is (coordinate, player)
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"

    # TRADE: too complicated for now...
    END_TURN = "END_TURN"


Action = namedtuple("Action", ["player", "action_type", "value"])


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
        return [Action(player, ActionType.BUILD_ROAD, edge) for edge in buildable_edges]
    else:
        return []


def settlement_possible_actions(player, board):
    has_money = player.resource_deck.includes(ResourceDeck.settlement_cost())

    settlements = board.get_player_buildings(player.color, BuildingType.SETTLEMENT)
    has_settlements_available = len(settlements) < 5

    if has_money and has_settlements_available:
        buildable_nodes = board.buildable_nodes(player.color)
        return [
            Action(player, ActionType.BUILD_SETTLEMENT, node)
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
            Action(player, ActionType.BUILD_CITY, node) for (node, _) in settlements
        ]
    else:
        return []


def robber_possibilities(player, board, players):
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
            actions.append(Action(player, ActionType.MOVE_ROBBER, (coordinate, None)))
        else:
            for p in players_to_steal_from:
                actions.append(Action(player, ActionType.MOVE_ROBBER, (coordinate, p)))

    return actions
