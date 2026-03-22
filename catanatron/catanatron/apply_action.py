import random
from collections import defaultdict
from typing import Dict

from catanatron.models.board import Board
from catanatron.models.enums import (
    MONOPOLY,
    RESOURCES,
    YEAR_OF_PLENTY,
    SETTLEMENT,
    CITY,
    Action,
    ActionPrompt,
    ActionRecord,
    ActionType,
)
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_draw,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_subtract,
)
from catanatron.models.actions import (
    road_building_possibilities,
)
from catanatron.state import State
from catanatron.state_functions import (
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    maintain_longest_road,
    play_dev_card,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_deck_random_select,
    player_freqdeck_add,
    player_deck_draw,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)
from catanatron.models.player import Color
from catanatron.models.enums import FastResource


def apply_action(
    state: State, action: Action, action_record: ActionRecord = None
) -> ActionRecord:
    """Main controller call. Follows redux-like pattern and
    routes the given action to the appropiate state-changing calls.

    Responsible for maintaining:
        .current_player_index
        .current_turn_index
        .current_prompt (and similars).

    Appends given action to the list of actions, as fully-specified action.

    Args:
        state (State): State to mutate
        action (Action): Action to carry out
        action_record (ActionRecord, optional): In case of replaying functionality.
            Defaults to None.

    Raises:
        ValueError: If invalid action given

    Returns:
        ActionRecord: Fully-specified action
    """

    if action.action_type == ActionType.END_TURN:
        action_record = apply_end_turn(state, action)
    elif action.action_type == ActionType.BUILD_SETTLEMENT:
        action_record = apply_build_settlement(state, action)
    elif action.action_type == ActionType.BUILD_ROAD:
        action_record = apply_build_road(state, action)
    elif action.action_type == ActionType.BUILD_CITY:
        action_record = apply_build_city(state, action)
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        action_record = apply_buy_development_card(state, action, action_record)
    elif action.action_type == ActionType.ROLL:
        action_record = apply_roll(state, action, action_record)
    elif action.action_type == ActionType.DISCARD:
        action_record = apply_discard(state, action, action_record)
    elif action.action_type == ActionType.MOVE_ROBBER:
        action_record = apply_move_robber(state, action, action_record)
    elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
        action_record = apply_play_knight_card(state, action)
    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        action_record = apply_play_year_of_plenty(state, action)
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        action_record = apply_play_monopoly(state, action)
    elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
        action_record = apply_play_road_building(state, action)
    elif action.action_type == ActionType.MARITIME_TRADE:
        action_record = apply_maritime_trade(state, action)
    elif action.action_type == ActionType.OFFER_TRADE:
        action_record = apply_offer_trade(state, action)
    elif action.action_type == ActionType.ACCEPT_TRADE:
        action_record = apply_accept_trade(state, action)
    elif action.action_type == ActionType.REJECT_TRADE:
        action_record = apply_reject_trade(state, action)
    elif action.action_type == ActionType.CONFIRM_TRADE:
        action_record = apply_confirm_trade(state, action)
    elif action.action_type == ActionType.CANCEL_TRADE:
        action_record = apply_cancel_trade(state, action)
    else:
        raise ValueError("Unknown ActionType " + str(action.action_type))

    state.action_records.append(action_record)
    return action_record


# ===== Apply Action Handlers =====
def apply_end_turn(state: State, action: Action):
    player_clean_turn(state, action.color)
    advance_turn(state)
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_build_settlement(state: State, action: Action):
    node_id = action.value
    if state.is_initial_build_phase:
        state.board.build_settlement(action.color, node_id, True)
        build_settlement(state, action.color, node_id, True)
        buildings = state.buildings_by_color[action.color][SETTLEMENT]

        # yield resources if second settlement
        is_second_house = len(buildings) == 2
        if is_second_house:
            key = player_key(state, action.color)
            for tile in state.board.map.adjacent_tiles[node_id]:
                if tile.resource != None:
                    freqdeck_draw(state.resource_freqdeck, 1, tile.resource)  # type: ignore
                    state.player_state[f"{key}_{tile.resource}_IN_HAND"] += 1

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.BUILD_INITIAL_ROAD
    else:
        (
            previous_road_color,
            road_color,
            road_lengths,
        ) = state.board.build_settlement(action.color, node_id, False)
        build_settlement(state, action.color, node_id, False)
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, SETTLEMENT_COST_FREQDECK
        )  # replenish bank
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
    return ActionRecord(action=action, result=None)


def apply_build_road(state: State, action: Action):
    edge = action.value
    if state.is_initial_build_phase:
        state.board.build_road(action.color, edge)
        build_road(state, action.color, edge, True)

        # state.current_player_index depend on what index are we
        # state.current_prompt too
        buildings = [
            len(state.buildings_by_color[color][SETTLEMENT])
            for color in state.color_to_index.keys()
        ]
        num_buildings = sum(buildings)
        num_players = len(buildings)
        going_forward = num_buildings < num_players
        at_the_end = num_buildings == num_players
        if going_forward:
            advance_turn(state)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif at_the_end:
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif num_buildings == 2 * num_players:
            state.is_initial_build_phase = False
            state.current_prompt = ActionPrompt.PLAY_TURN
        else:
            advance_turn(state, -1)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
    elif state.is_road_building and state.free_roads_available > 0:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, True)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        state.free_roads_available -= 1
        if (
            state.free_roads_available == 0
            or len(road_building_possibilities(state, action.color, False)) == 0
        ):
            state.is_road_building = False
            state.free_roads_available = 0
            # state.current_player_index stays the same
            # state.current_prompt stays as PLAY
    else:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, False)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
    return ActionRecord(action=action, result=None)


def apply_build_city(state: State, action: Action):
    node_id = action.value
    state.board.build_city(action.color, node_id)
    build_city(state, action.color, node_id)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, CITY_COST_FREQDECK
    )  # replenish bank

    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    return ActionRecord(action=action, result=None)


def apply_buy_development_card(state: State, action: Action, action_record=None):
    """Optionally takes action_record in case we want to support replay functionality."""
    if len(state.development_listdeck) == 0:
        raise ValueError("No more development cards")
    if not player_can_afford_dev_card(state, action.color):
        raise ValueError("No money to buy development card")

    if action_record is None:
        card = state.development_listdeck.pop()  # already shuffled
    else:
        card = action_record.result
        draw_from_listdeck(state.development_listdeck, 1, card)

    buy_dev_card(state, action.color, card)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK
    )

    action = Action(action.color, action.action_type, card)
    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    return ActionRecord(action=action, result=card)


def apply_roll(state: State, action: Action, action_record=None):
    key = player_key(state, action.color)
    state.player_state[f"{key}_HAS_ROLLED"] = True

    dices = action_record.result if action_record is not None else roll_dice()
    number = dices[0] + dices[1]
    action = Action(action.color, action.action_type, dices)

    if number == 7:
        discarders = [
            player_num_resource_cards(state, color) > state.discard_limit
            for color in state.colors
        ]
        should_enter_discarding_sequence = any(discarders)

        if should_enter_discarding_sequence:
            state.current_player_index = discarders.index(True)
            state.current_prompt = ActionPrompt.DISCARD
            state.is_discarding = True
        else:
            # state.current_player_index stays the same
            state.current_prompt = ActionPrompt.MOVE_ROBBER
            state.is_moving_knight = True
    else:
        payout, _ = yield_resources(state.board, state.resource_freqdeck, number)
        for color, resource_freqdeck in payout.items():
            # Atomically add to player's hand and remove from bank
            player_freqdeck_add(state, color, resource_freqdeck)
            state.resource_freqdeck = freqdeck_subtract(
                state.resource_freqdeck, resource_freqdeck
            )

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN

    return ActionRecord(action=action, result=dices)


def apply_discard(state: State, action: Action, action_record=None):
    hand = player_deck_to_array(state, action.color)
    num_to_discard = len(hand) // 2
    if action_record is None:
        # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
        discarded = random.sample(hand, k=num_to_discard)
    else:
        discarded = action_record.result  # for replay functionality
    to_discard = freqdeck_from_listdeck(discarded)

    player_freqdeck_subtract(state, action.color, to_discard)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, to_discard)
    action = Action(action.color, action.action_type, discarded)

    # Advance turn
    discarders_left = [
        player_num_resource_cards(state, color) > 7 for color in state.colors
    ][state.current_player_index + 1 :]
    if any(discarders_left):
        to_skip = discarders_left.index(True)
        state.current_player_index = state.current_player_index + 1 + to_skip
        # state.current_prompt stays the same
    else:
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_discarding = False
        state.is_moving_knight = True

    return ActionRecord(action=action, result=discarded)


def apply_move_robber(state: State, action: Action, action_record=None):
    (coordinate, robbed_color) = action.value
    robbed_resource = None
    if robbed_color is not None:
        robbed_resource = (
            action_record.result
            if action_record is not None
            else player_deck_random_select(state, robbed_color)
        )
        player_deck_draw(state, robbed_color, robbed_resource)
        player_deck_replenish(state, action.color, robbed_resource)
    state.board.robber_coordinate = coordinate

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN

    return ActionRecord(action=action, result=robbed_resource)


def apply_play_knight_card(state: State, action: Action):
    if not player_can_play_dev(state, action.color, "KNIGHT"):
        raise ValueError("Player cant play knight card now")

    play_dev_card(state, action.color, "KNIGHT")

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.MOVE_ROBBER
    return ActionRecord(action=action, result=None)


def apply_play_year_of_plenty(state: State, action: Action):
    cards_selected = freqdeck_from_listdeck(action.value)
    if not player_can_play_dev(state, action.color, YEAR_OF_PLENTY):
        raise ValueError("Player cant play year of plenty now")
    if not freqdeck_contains(state.resource_freqdeck, cards_selected):
        raise ValueError("Not enough resources of this type (these types?) in bank")
    player_freqdeck_add(state, action.color, cards_selected)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards_selected)
    play_dev_card(state, action.color, YEAR_OF_PLENTY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_play_monopoly(state: State, action: Action):
    mono_resource = action.value
    cards_stolen = [0, 0, 0, 0, 0]
    if not player_can_play_dev(state, action.color, MONOPOLY):
        raise ValueError("Player cant play monopoly now")
    for color in state.colors:
        if not color == action.color:
            key = player_key(state, color)
            number_of_cards_to_steal = state.player_state[
                f"{key}_{mono_resource}_IN_HAND"
            ]
            freqdeck_replenish(cards_stolen, number_of_cards_to_steal, mono_resource)
            player_deck_draw(state, color, mono_resource, number_of_cards_to_steal)
    player_freqdeck_add(state, action.color, cards_stolen)
    play_dev_card(state, action.color, MONOPOLY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_play_road_building(state: State, action: Action):
    if not player_can_play_dev(state, action.color, "ROAD_BUILDING"):
        raise ValueError("Player cant play road building now")

    play_dev_card(state, action.color, "ROAD_BUILDING")
    state.is_road_building = True
    state.free_roads_available = 2

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_maritime_trade(state: State, action: Action):
    trade_offer = action.value
    offering = freqdeck_from_listdeck(filter(lambda r: r is not None, trade_offer[:-1]))
    asking = freqdeck_from_listdeck(trade_offer[-1:])
    if not player_resource_freqdeck_contains(state, action.color, offering):
        raise ValueError("Trying to trade without money")
    if not freqdeck_contains(state.resource_freqdeck, asking):
        raise ValueError("Bank doenst have those cards")
    player_freqdeck_subtract(state, action.color, offering)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, offering)
    player_freqdeck_add(state, action.color, asking)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, asking)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_offer_trade(state: State, action: Action):
    state.is_resolving_trade = True
    state.current_trade = (*action.value, state.current_turn_index)

    # go in seating order; order won't matter because of "acceptees hook"
    state.current_player_index = next(
        i for i, c in enumerate(state.colors) if c != action.color
    )  # cant ask yourself
    state.current_prompt = ActionPrompt.DECIDE_TRADE
    return ActionRecord(action=action, result=None)


def apply_accept_trade(state: State, action: Action):
    # add yourself to self.acceptees
    index = state.colors.index(action.color)
    new_acceptess = list(state.acceptees)
    new_acceptess[index] = True  # type: ignore
    state.acceptees = tuple(new_acceptess)

    try:
        # keep going around table w/o asking yourself or players that have answered
        state.current_player_index = next(
            i
            for i, c in enumerate(state.colors)
            if c != action.color and i > state.current_player_index
        )
        # .is_resolving_trade, .current_trade, .current_prompt, .acceptees stay the same
    except StopIteration:
        # by this action, there is at least 1 acceptee, so go to DECIDE_ACCEPTEES
        # .is_resolving_trade, .current_trade, .acceptees stay the same
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES

    return ActionRecord(action=action, result=None)


def apply_reject_trade(state: State, action: Action):
    try:
        # keep going around table w/o asking yourself or players that have answered
        state.current_player_index = next(
            i
            for i, c in enumerate(state.colors)
            if c != action.color and i > state.current_player_index
        )
        # .is_resolving_trade, .current_trade, .current_prompt, .acceptees stay the same
    except StopIteration:
        # if no acceptees at this point, go back to PLAY_TURN
        if sum(state.acceptees) == 0:
            reset_trading_state(state)

            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.PLAY_TURN
        else:
            # go to offering player with all the answers
            # .is_resolving_trade, .current_trade, .acceptees stay the same
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES

    return ActionRecord(action=action, result=None)


def apply_confirm_trade(state: State, action: Action):
    offering = action.value[:5]
    asking = action.value[5:10]
    enemy_color = action.value[10]
    player_freqdeck_subtract(state, action.color, offering)
    player_freqdeck_add(state, action.color, asking)
    player_freqdeck_subtract(state, enemy_color, asking)
    player_freqdeck_add(state, enemy_color, offering)

    reset_trading_state(state)

    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


def apply_cancel_trade(state: State, action: Action):
    reset_trading_state(state)

    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    return ActionRecord(action=action, result=None)


# ===== Helper Functions =====
def roll_dice():
    """Yields two random numbers

    Returns:
        tuple[int, int]: 2-tuple of random numbers from 1 to 6 inclusive.
    """
    return (random.randint(1, 6), random.randint(1, 6))


def yield_resources(board: Board, resource_freqdeck, number):
    """Computes resource payouts for given board and dice roll number.

    Args:
        board (Board): Board state
        resource_freqdeck (List[int]): Bank's resource freqdeck
        number (int): Sum of dice roll

    Returns:
        (dict, List[int]): 2-tuple.
            First element is color => freqdeck mapping. e.g. {Color.RED: [0,0,0,3,0]}.
            Second is an array of resources that couldn't be yieleded
            because they depleted.
    """
    intented_payout: Dict[Color, Dict[FastResource, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    resource_totals: Dict[FastResource, int] = defaultdict(int)
    for coordinate, tile in board.map.land_tiles.items():
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id, None)
            assert tile.resource is not None
            if building is None:
                continue
            elif building[1] == SETTLEMENT:
                intented_payout[building[0]][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building[1] == CITY:
                intented_payout[building[0]][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource in RESOURCES:
        total = resource_totals[resource]
        if not freqdeck_can_draw(resource_freqdeck, total, resource):
            depleted.append(resource)

    # build final data color => freqdeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = [0, 0, 0, 0, 0]

        for resource, count in player_payout.items():
            if resource not in depleted:
                freqdeck_replenish(payout[player], count, resource)

    return payout, depleted


def advance_turn(state, direction=1):
    """Sets .current_player_index"""
    next_index = next_player_index(state, direction)
    state.current_player_index = next_index
    state.current_turn_index = next_index
    state.num_turns += 1


def next_player_index(state, direction=1):
    return (state.current_player_index + direction) % len(state.colors)


def reset_trading_state(state):
    state.is_resolving_trade = False
    state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    state.acceptees = tuple(False for _ in state.colors)
