"""
Move-generation functions (these return a list of actions that can be taken 
by current player). Main function is generate_playable_actions.
"""
import operator as op
from functools import reduce
from typing import Any, Dict, List, Set, Tuple, Union

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_count,
    freqdeck_from_listdeck,
)
from catanatron.models.enums import (
    RESOURCES,
    Action,
    ActionPrompt,
    ActionType,
    BRICK,
    ORE,
    FastResource,
    SETTLEMENT,
    SHEEP,
    WHEAT,
    WOOD,
)
from catanatron.state_functions import (
    get_player_buildings,
    get_player_freqdeck,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_has_rolled,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)


def generate_playable_actions(state) -> List[Action]:
    action_prompt = state.current_prompt
    color = state.current_color()

    if action_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return settlement_possibilities(state, color, True)
    elif action_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return initial_road_possibilities(state, color)
    elif action_prompt == ActionPrompt.MOVE_ROBBER:
        return robber_possibilities(state, color)
    elif action_prompt == ActionPrompt.PLAY_TURN:
        if state.is_road_building:
            actions = road_building_possibilities(state, color, False)
        elif not player_has_rolled(state, color):
            actions = [Action(color, ActionType.ROLL, None)]
            if player_can_play_dev(state, color, "KNIGHT"):
                actions.append(Action(color, ActionType.PLAY_KNIGHT_CARD, None))
        else:
            actions = [Action(color, ActionType.END_TURN, None)]
            actions.extend(road_building_possibilities(state, color))
            actions.extend(settlement_possibilities(state, color))
            actions.extend(city_possibilities(state, color))

            can_buy_dev_card = (
                player_can_afford_dev_card(state, color)
                and len(state.development_listdeck) > 0
            )
            if can_buy_dev_card:
                actions.append(Action(color, ActionType.BUY_DEVELOPMENT_CARD, None))

            # Play Dev Cards
            if player_can_play_dev(state, color, "YEAR_OF_PLENTY"):
                actions.extend(
                    year_of_plenty_possibilities(color, state.resource_freqdeck)
                )
            if player_can_play_dev(state, color, "MONOPOLY"):
                actions.extend(monopoly_possibilities(color))
            if player_can_play_dev(state, color, "KNIGHT"):
                actions.append(Action(color, ActionType.PLAY_KNIGHT_CARD, None))
            if (
                player_can_play_dev(state, color, "ROAD_BUILDING")
                and len(road_building_possibilities(state, color, False)) > 0
            ):
                actions.append(Action(color, ActionType.PLAY_ROAD_BUILDING, None))

            # Trade
            actions.extend(maritime_trade_possibilities(state, color))
        return actions
    elif action_prompt == ActionPrompt.DISCARD:
        return discard_possibilities(color)
    elif action_prompt == ActionPrompt.DECIDE_TRADE:
        actions = [Action(color, ActionType.REJECT_TRADE, state.current_trade)]

        # can only accept if have enough cards
        freqdeck = get_player_freqdeck(state, color)
        asked = state.current_trade[5:10]
        if freqdeck_contains(freqdeck, asked):
            actions.append(Action(color, ActionType.ACCEPT_TRADE, state.current_trade))

        return actions
    elif action_prompt == ActionPrompt.DECIDE_ACCEPTEES:
        # you should be able to accept for each of the "accepting players"
        actions = [Action(color, ActionType.CANCEL_TRADE, None)]

        for other_color, accepted in zip(state.colors, state.acceptees):
            if accepted:
                actions.append(
                    Action(
                        color,
                        ActionType.CONFIRM_TRADE,
                        (*state.current_trade[:10], other_color),
                    )
                )
        return actions
    else:
        raise RuntimeError("Unknown ActionPrompt: " + str(action_prompt))


def monopoly_possibilities(color) -> List[Action]:
    return [Action(color, ActionType.PLAY_MONOPOLY, card) for card in RESOURCES]


def year_of_plenty_possibilities(color, freqdeck: List[int]) -> List[Action]:
    options: Set[Union[Tuple[FastResource, FastResource], Tuple[FastResource]]] = set()
    for i, first_card in enumerate(RESOURCES):
        for j in range(i, len(RESOURCES)):
            second_card = RESOURCES[j]  # doing it this way to not repeat

            to_draw = freqdeck_from_listdeck([first_card, second_card])
            if freqdeck_contains(freqdeck, to_draw):
                options.add((first_card, second_card))
            else:  # try allowing player select 1 card only.
                if freqdeck_can_draw(freqdeck, 1, first_card):
                    options.add((first_card,))
                if freqdeck_can_draw(freqdeck, 1, second_card):
                    options.add((second_card,))

    return list(
        map(
            lambda cards: Action(color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(cards)),
            options,
        )
    )


def road_building_possibilities(state, color, check_money=True) -> List[Action]:
    key = player_key(state, color)

    has_money = player_resource_freqdeck_contains(state, color, ROAD_COST_FREQDECK)
    has_roads_available = state.player_state[f"{key}_ROADS_AVAILABLE"] > 0

    if (not check_money or has_money) and has_roads_available:
        buildable_edges = state.board.buildable_edges(color)
        return [Action(color, ActionType.BUILD_ROAD, edge) for edge in buildable_edges]
    else:
        return []


def settlement_possibilities(state, color, initial_build_phase=False) -> List[Action]:
    if initial_build_phase:
        buildable_node_ids = state.board.buildable_node_ids(
            color, initial_build_phase=True
        )
        return [
            Action(color, ActionType.BUILD_SETTLEMENT, node_id)
            for node_id in buildable_node_ids
        ]
    else:
        key = player_key(state, color)
        has_money = player_resource_freqdeck_contains(
            state, color, SETTLEMENT_COST_FREQDECK
        )
        has_settlements_available = (
            state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] > 0
        )
        if has_money and has_settlements_available:
            buildable_node_ids = state.board.buildable_node_ids(color)
            return [
                Action(color, ActionType.BUILD_SETTLEMENT, node_id)
                for node_id in buildable_node_ids
            ]
        else:
            return []


def city_possibilities(state, color) -> List[Action]:
    key = player_key(state, color)

    has_money = player_resource_freqdeck_contains(state, color, CITY_COST_FREQDECK)
    has_cities_available = state.player_state[f"{key}_CITIES_AVAILABLE"] > 0

    if has_money and has_cities_available:
        return [
            Action(color, ActionType.BUILD_CITY, node_id)
            for node_id in get_player_buildings(state, color, SETTLEMENT)
        ]
    else:
        return []


def robber_possibilities(state, color) -> List[Action]:
    actions = []
    for coordinate, tile in state.board.map.land_tiles.items():
        if coordinate == state.board.robber_coordinate:
            continue  # ignore. must move robber.

        # each tile can yield a (move-but-cant-steal) action or
        #   several (move-and-steal-from-x) actions.
        to_steal_from = set()  # set of player_indexs
        for _, node_id in tile.nodes.items():
            building = state.board.buildings.get(node_id, None)
            if building is not None:
                candidate_color = building[0]
                if (
                    player_num_resource_cards(state, candidate_color) >= 1
                    and color != candidate_color  # can't play yourself
                ):
                    to_steal_from.add(candidate_color)

        if len(to_steal_from) == 0:
            actions.append(
                Action(color, ActionType.MOVE_ROBBER, (coordinate, None, None))
            )
        else:
            for enemy_color in to_steal_from:
                actions.append(
                    Action(
                        color, ActionType.MOVE_ROBBER, (coordinate, enemy_color, None)
                    )
                )

    return actions


def initial_road_possibilities(state, color) -> List[Action]:
    # Must be connected to last settlement
    last_settlement_node_id = state.buildings_by_color[color][SETTLEMENT][-1]

    buildable_edges = filter(
        lambda edge: last_settlement_node_id in edge,
        state.board.buildable_edges(color),
    )
    return [Action(color, ActionType.BUILD_ROAD, edge) for edge in buildable_edges]


def discard_possibilities(color) -> List[Action]:
    return [Action(color, ActionType.DISCARD, None)]
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


def maritime_trade_possibilities(state, color) -> List[Action]:
    hand_freqdeck = [
        player_num_resource_cards(state, color, resource) for resource in RESOURCES
    ]
    port_resources = state.board.get_player_port_resources(color)
    trade_offers = inner_maritime_trade_possibilities(
        hand_freqdeck, state.resource_freqdeck, port_resources
    )

    return list(
        map(lambda t: Action(color, ActionType.MARITIME_TRADE, t), trade_offers)
    )


def inner_maritime_trade_possibilities(hand_freqdeck, bank_freqdeck, port_resources):
    """This inner function is to make this logic more shareable"""
    trade_offers = set()

    # Get lowest rate per resource
    rates: Dict[FastResource, int] = {WOOD: 4, BRICK: 4, SHEEP: 4, WHEAT: 4, ORE: 4}
    if None in port_resources:
        rates = {WOOD: 3, BRICK: 3, SHEEP: 3, WHEAT: 3, ORE: 3}
    for resource in port_resources:
        if resource != None:
            rates[resource] = 2

    # For resource in hand
    for index, resource in enumerate(RESOURCES):
        amount = hand_freqdeck[index]
        if amount >= rates[resource]:
            resource_out: List[Any] = [resource] * rates[resource]
            resource_out += [None] * (4 - rates[resource])
            for j_resource in RESOURCES:
                if (
                    resource != j_resource
                    and freqdeck_count(bank_freqdeck, j_resource) > 0
                ):
                    trade_offer = tuple(resource_out + [j_resource])
                    trade_offers.add(trade_offer)

    return trade_offers
