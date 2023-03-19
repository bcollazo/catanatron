"""
Module with main State class and main apply_action call (game controller).
"""

import random
import pickle
from collections import defaultdict
from typing import Any, List, Tuple, Dict, Iterable

from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.board import Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    MONOPOLY,
    RESOURCES,
    YEAR_OF_PLENTY,
    SETTLEMENT,
    CITY,
    Action,
    ActionPrompt,
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
    starting_devcard_bank,
    starting_resource_bank,
)
from catanatron.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
)
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
    player_freqdeck_add,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)
from catanatron.models.player import Color, Player
from catanatron.models.enums import FastResource

# These will be prefixed by P0_, P1_, ...
# Create Player State blueprint
PLAYER_INITIAL_STATE = {
    "VICTORY_POINTS": 0,
    "ROADS_AVAILABLE": 15,
    "SETTLEMENTS_AVAILABLE": 5,
    "CITIES_AVAILABLE": 4,
    "HAS_ROAD": False,
    "HAS_ARMY": False,
    "HAS_ROLLED": False,
    "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN": False,
    # de-normalized features (for performance since we think they are good features)
    "ACTUAL_VICTORY_POINTS": 0,
    "LONGEST_ROAD_LENGTH": 0,
}
for resource in RESOURCES:
    PLAYER_INITIAL_STATE[f"{resource}_IN_HAND"] = 0
for dev_card in DEVELOPMENT_CARDS:
    PLAYER_INITIAL_STATE[f"{dev_card}_IN_HAND"] = 0
    PLAYER_INITIAL_STATE[f"PLAYED_{dev_card}"] = 0


class State:
    """Collection of variables representing state

    Attributes:
        players (List[Player]): DEPRECATED. Reference to list of players.
            Use .colors instead, and move this reference to the Game class.
            Deprecated because we want this class to only contain state
            information that can be easily copiable.
        board (Board): Board state. Settlement locations, cities,
            roads, ect... See Board class.
        player_state (Dict[str, Any]): See PLAYER_INITIAL_STATE. It will
            contain one of each key in PLAYER_INITIAL_STATE but prefixed
            with "P<index_of_player>".
            Example: { P0_HAS_ROAD: False, P1_SETTLEMENTS_AVAILABLE: 18, ... }
        color_to_index (Dict[Color, int]): Color to seating location cache
        colors (Tuple[Color]): Represents seating order.
        resource_freqdeck (List[int]): Represents resource cards in the bank.
            Each element is the amount of [WOOD, BRICK, SHEEP, WHEAT, ORE].
        development_listdeck (List[FastDevCard]): Represents development cards in
            the bank. Already shuffled.
        buildings_by_color (Dict[Color, Dict[FastBuildingType, List]]): Cache of
            buildings. Can be used like: `buildings_by_color[Color.RED][SETTLEMENT]`
            to get a list of all node ids where RED has settlements.
        actions (List[Action]): Log of all actions taken. Fully-specified actions.
        num_turns (int): number of turns thus far
        current_player_index (int): index per colors array of player that should be
            making a decision now. Not necesarilly the same as current_turn_index
            because there are out-of-turn decisions like discarding.
        current_turn_index (int): index per colors array of player whose turn is it.
        current_prompt (ActionPrompt): DEPRECATED. Not needed; use is_initial_build_phase,
            is_moving_knight, etc... instead.
        is_discarding (bool): If current player needs to discard.
        is_moving_knight (bool): If current player needs to move robber.
        is_road_building (bool): If current player needs to build free roads per Road
            Building dev card.
        free_roads_available (int): Number of roads available left in Road Building
            phase.
        playable_actions (List[Action]): List of playable actions by current player.
    """

    def __init__(
        self,
        players: List[Player],
        catan_map=None,
        discard_limit=7,
        initialize=True,
    ):
        if initialize:
            self.players = random.sample(players, len(players))
            self.colors = tuple([player.color for player in self.players])
            self.board = Board(catan_map or CatanMap.from_template(BASE_MAP_TEMPLATE))
            self.discard_limit = discard_limit

            # feature-ready dictionary
            self.player_state = dict()
            for index in range(len(self.colors)):
                for key, value in PLAYER_INITIAL_STATE.items():
                    self.player_state[f"P{index}_{key}"] = value
            self.color_to_index = {
                color: index for index, color in enumerate(self.colors)
            }

            self.resource_freqdeck = starting_resource_bank()
            self.development_listdeck = starting_devcard_bank()
            random.shuffle(self.development_listdeck)

            # Auxiliary attributes to implement game logic
            self.buildings_by_color: Dict[Color, Dict[Any, Any]] = {
                p.color: defaultdict(list) for p in players
            }
            self.actions: List[Action] = []  # log of all action taken by players
            self.num_turns = 0  # num_completed_turns

            # Current prompt / player
            # Two variables since there can be out-of-turn plays
            self.current_player_index = 0
            self.current_turn_index = 0

            # TODO: Deprecate self.current_prompt in favor of indicator variables
            self.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
            self.is_initial_build_phase = True
            self.is_discarding = False
            self.is_moving_knight = False
            self.is_road_building = False
            self.free_roads_available = 0

            self.is_resolving_trade = False
            self.current_trade: Tuple = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            self.acceptees = tuple(False for _ in self.colors)

            self.playable_actions = generate_playable_actions(self)

    def current_player(self):
        """Helper for accessing Player instance who should decide next"""
        return self.players[self.current_player_index]

    def current_color(self):
        """Helper for accessing color (player) who should decide next"""
        return self.colors[self.current_player_index]

    def copy(self):
        """Creates a copy of this State class that can be modified without
        repercusions to this one. Immutable values are just copied over.

        Returns:
            State: State copy.
        """
        state_copy = State([], None, initialize=False)
        state_copy.players = self.players
        state_copy.discard_limit = self.discard_limit  # immutable

        state_copy.board = self.board.copy()

        state_copy.player_state = self.player_state.copy()
        state_copy.color_to_index = self.color_to_index
        state_copy.colors = self.colors  # immutable

        state_copy.resource_freqdeck = self.resource_freqdeck.copy()
        state_copy.development_listdeck = self.development_listdeck.copy()

        state_copy.buildings_by_color = pickle.loads(
            pickle.dumps(self.buildings_by_color)
        )
        state_copy.actions = self.actions.copy()
        state_copy.num_turns = self.num_turns

        # Current prompt / player
        # Two variables since there can be out-of-turn plays
        state_copy.current_player_index = self.current_player_index
        state_copy.current_turn_index = self.current_turn_index

        state_copy.current_prompt = self.current_prompt
        state_copy.is_initial_build_phase = self.is_initial_build_phase
        state_copy.is_discarding = self.is_discarding
        state_copy.is_moving_knight = self.is_moving_knight
        state_copy.is_road_building = self.is_road_building
        state_copy.free_roads_available = self.free_roads_available

        state_copy.is_resolving_trade = self.is_resolving_trade
        state_copy.current_trade = self.current_trade
        state_copy.acceptees = self.acceptees

        state_copy.playable_actions = self.playable_actions
        return state_copy


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

        for _, node_id in tile.nodes.items():
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


def apply_action(state: State, action: Action):
    """Main controller call. Follows redux-like pattern and
    routes the given action to the appropiate state-changing calls.

    Responsible for maintaining:
        .current_player_index, .current_turn_index,
        .current_prompt (and similars), .playable_actions.

    Appends given action to the list of actions, as fully-specified action.

    Args:
        state (State): State to mutate
        action (Action): Action to carry out

    Raises:
        ValueError: If invalid action given

    Returns:
        Action: Fully-specified action
    """

    if action.action_type == ActionType.END_TURN:
        player_clean_turn(state, action.color)
        advance_turn(state)
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUILD_SETTLEMENT:
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
            state.playable_actions = generate_playable_actions(state)
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
            state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUILD_ROAD:
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
            state.playable_actions = generate_playable_actions(state)
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
            state.playable_actions = generate_playable_actions(state)
        else:
            result = state.board.build_road(action.color, edge)
            previous_road_color, road_color, road_lengths = result
            build_road(state, action.color, edge, False)
            maintain_longest_road(state, previous_road_color, road_color, road_lengths)

            # state.current_player_index stays the same
            # state.current_prompt stays as PLAY
            state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUILD_CITY:
        node_id = action.value
        state.board.build_city(action.color, node_id)
        build_city(state, action.color, node_id)
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, CITY_COST_FREQDECK
        )  # replenish bank

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        if len(state.development_listdeck) == 0:
            raise ValueError("No more development cards")
        if not player_can_afford_dev_card(state, action.color):
            raise ValueError("No money to buy development card")

        if action.value is None:
            card = state.development_listdeck.pop()  # already shuffled
        else:
            card = action.value
            draw_from_listdeck(state.development_listdeck, 1, card)

        buy_dev_card(state, action.color, card)
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK
        )

        action = Action(action.color, action.action_type, card)
        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.ROLL:
        key = player_key(state, action.color)
        state.player_state[f"{key}_HAS_ROLLED"] = True

        dices = action.value or roll_dice()
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
            state.playable_actions = generate_playable_actions(state)
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
            state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.DISCARD:
        hand = player_deck_to_array(state, action.color)
        num_to_discard = len(hand) // 2
        if action.value is None:
            # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
            discarded = random.sample(hand, k=num_to_discard)
        else:
            discarded = action.value  # for replay functionality
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

        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.MOVE_ROBBER:
        (coordinate, robbed_color, robbed_resource) = action.value
        state.board.robber_coordinate = coordinate
        if robbed_color is not None:
            if robbed_resource is None:
                robbed_resource = player_deck_random_draw(state, robbed_color)
                action = Action(
                    action.color,
                    action.action_type,
                    (coordinate, robbed_color, robbed_resource),
                )
            else:  # for replay functionality
                player_deck_draw(state, robbed_color, robbed_resource)
            player_deck_replenish(state, action.color, robbed_resource)

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
        if not player_can_play_dev(state, action.color, "KNIGHT"):
            raise ValueError("Player cant play knight card now")

        play_dev_card(state, action.color, "KNIGHT")

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        cards_selected = freqdeck_from_listdeck(action.value)
        if not player_can_play_dev(state, action.color, YEAR_OF_PLENTY):
            raise ValueError("Player cant play year of plenty now")
        if not freqdeck_contains(state.resource_freqdeck, cards_selected):
            raise ValueError("Not enough resources of this type (these types?) in bank")
        player_freqdeck_add(state, action.color, cards_selected)
        state.resource_freqdeck = freqdeck_subtract(
            state.resource_freqdeck, cards_selected
        )
        play_dev_card(state, action.color, YEAR_OF_PLENTY)

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.PLAY_MONOPOLY:
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
                freqdeck_replenish(
                    cards_stolen, number_of_cards_to_steal, mono_resource
                )
                player_deck_draw(state, color, mono_resource, number_of_cards_to_steal)
        player_freqdeck_add(state, action.color, cards_stolen)
        play_dev_card(state, action.color, MONOPOLY)

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
        if not player_can_play_dev(state, action.color, "ROAD_BUILDING"):
            raise ValueError("Player cant play road building now")

        play_dev_card(state, action.color, "ROAD_BUILDING")
        state.is_road_building = True
        state.free_roads_available = 2

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.MARITIME_TRADE:
        trade_offer = action.value
        offering = freqdeck_from_listdeck(
            filter(lambda r: r is not None, trade_offer[:-1])
        )
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
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.OFFER_TRADE:
        state.is_resolving_trade = True
        state.current_trade = (*action.value, state.current_turn_index)

        # go in seating order; order won't matter because of "acceptees hook"
        state.current_player_index = next(
            i for i, c in enumerate(state.colors) if c != action.color
        )  # cant ask yourself
        state.current_prompt = ActionPrompt.DECIDE_TRADE

        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.ACCEPT_TRADE:
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

        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.REJECT_TRADE:
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

        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.CONFIRM_TRADE:
        # apply trade
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
    elif action.action_type == ActionType.CANCEL_TRADE:
        reset_trading_state(state)

        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.PLAY_TURN
    else:
        raise ValueError("Unknown ActionType " + str(action.action_type))

    # TODO: Think about possible-action/idea vs finalized-action design
    state.actions.append(action)
    return action


def reset_trading_state(state):
    state.is_resolving_trade = False
    state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    state.acceptees = tuple(False for _ in state.colors)
