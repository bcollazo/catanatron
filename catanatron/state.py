import random
import pickle
from collections import defaultdict

from catanatron.models.map import BaseMap
from catanatron.models.board import Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    MONOPOLY,
    RESOURCES,
    Resource,
    BuildingType,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.decks import DevelopmentDeck, ResourceDeck
from catanatron.models.actions import (
    generate_playable_actions,
    road_possible_actions,
)
from catanatron.state_functions import (
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    mantain_longest_road,
    play_dev_card,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_deck_add,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_deck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_deck_contains,
)


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
    """Small container object to group dynamic variables in state"""

    def __init__(self, players, catan_map=None, initialize=True):
        if initialize:
            self.players = random.sample(players, len(players))
            self.board = Board(catan_map or BaseMap())

            # feature-ready dictionary
            self.player_state = dict()
            for index in range(len(self.players)):
                for key, value in PLAYER_INITIAL_STATE.items():
                    self.player_state[f"P{index}_{key}"] = value
            self.color_to_index = {
                player.color: index for index, player in enumerate(self.players)
            }
            self.colors = tuple([player.color for player in self.players])

            self.resource_deck = ResourceDeck.starting_bank()
            self.development_deck = DevelopmentDeck.starting_bank()

            # Auxiliary attributes to implement game logic
            self.buildings_by_color = {p.color: defaultdict(list) for p in players}
            self.actions = []  # log of all action taken by players
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

            self.playable_actions = generate_playable_actions(self)

    def current_player(self):
        return self.players[self.current_player_index]

    def copy(self):
        state_copy = State(None, None, initialize=False)
        state_copy.players = self.players

        state_copy.board = self.board.copy()

        state_copy.player_state = self.player_state.copy()
        state_copy.color_to_index = self.color_to_index
        state_copy.colors = self.colors  # immutable, so no need to copy

        # TODO: Move Deck to functional code, so as to quick-copy arrays.
        state_copy.resource_deck = pickle.loads(pickle.dumps(self.resource_deck))
        state_copy.development_deck = pickle.loads(pickle.dumps(self.development_deck))

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

        state_copy.playable_actions = self.playable_actions
        return state_copy


def roll_dice():
    return (random.randint(1, 6), random.randint(1, 6))


def yield_resources(board, resource_deck, number):
    """
    Returns:
        (payouts, depleted): tuple where:
        payouts: dictionary of "resource_deck" keyed by player
                e.g. {Color.RED: ResourceDeck({Resource.WEAT: 3})}
            depleted: list of resources that couldn't yield
    """
    intented_payout = defaultdict(lambda: defaultdict(int))
    resource_totals = defaultdict(int)
    for coordinate, tile in board.map.resource_tiles:
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for _, node_id in tile.nodes.items():
            building = board.buildings.get(node_id, None)
            if building is None:
                continue
            elif building[1] == BuildingType.SETTLEMENT:
                intented_payout[building[0]][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building[1] == BuildingType.CITY:
                intented_payout[building[0]][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource in Resource:
        total = resource_totals[resource]
        if not resource_deck.can_draw(total, resource):
            depleted.append(resource)

    # build final data ResourceDeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = ResourceDeck()

        for resource, count in player_payout.items():
            if resource not in depleted:
                payout[player].replenish(count, resource)

    return payout, depleted


def advance_turn(state, direction=1):
    """Sets .current_player_index"""
    next_index = next_player_index(state, direction)
    state.current_player_index = next_index
    state.current_turn_index = next_index
    state.num_turns += 1


def next_player_index(state, direction=1):
    return (state.current_player_index + direction) % len(state.players)


def apply_action(state: State, action: Action):
    """Action router function. Reducer-like function.

    Each branch is responsible of mantaining:
        .current_player_index, .current_turn_index,
        .current_prompt (and similars), .playable_actions
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
            buildings = state.buildings_by_color[action.color][BuildingType.SETTLEMENT]

            # yield resources if second settlement
            is_second_house = len(buildings) == 2
            if is_second_house:
                key = player_key(state, action.color)
                for tile in state.board.map.adjacent_tiles[node_id]:
                    if tile.resource != None:
                        state.resource_deck.draw(1, tile.resource)
                        state.player_state[f"{key}_{tile.resource.value}_IN_HAND"] += 1

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
            state.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
            mantain_longest_road(state, previous_road_color, road_color, road_lengths)

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
                len(state.buildings_by_color[color][BuildingType.SETTLEMENT])
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
            mantain_longest_road(state, previous_road_color, road_color, road_lengths)

            state.free_roads_available -= 1
            if (
                state.free_roads_available == 0
                or len(road_possible_actions(state, action.color)) == 0
            ):
                state.is_road_building = False
                state.free_roads_available == 0
                # state.current_player_index stays the same
                # state.current_prompt stays as PLAY
            state.playable_actions = generate_playable_actions(state)
        else:
            result = state.board.build_road(action.color, edge)
            previous_road_color, road_color, road_lengths = result
            build_road(state, action.color, edge, False)
            mantain_longest_road(state, previous_road_color, road_color, road_lengths)

            # state.current_player_index stays the same
            # state.current_prompt stays as PLAY
            state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUILD_CITY:
        node_id = action.value
        state.board.build_city(action.color, node_id)
        build_city(state, action.color, node_id)
        state.resource_deck += ResourceDeck.city_cost()  # replenish bank

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        if state.development_deck.num_cards() == 0:
            raise ValueError("No more development cards")
        if not player_can_afford_dev_card(state, action.color):
            raise ValueError("No money to buy development card")

        if action.value is None:
            card = state.development_deck.random_draw()
        else:
            card = action.value
            state.development_deck.draw(1, card)

        buy_dev_card(state, action.color, card.value)
        state.resource_deck += ResourceDeck.development_card_cost()

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
                player_num_resource_cards(state, color) > 7 for color in state.colors
            ]
            is_discarding = any(discarders)

            if is_discarding:
                state.current_player_index = discarders.index(True)
                state.current_prompt = ActionPrompt.DISCARD
                state.is_discarding = True
            else:
                # state.current_player_index stays the same
                state.current_prompt = ActionPrompt.MOVE_ROBBER
                state.is_moving_knight = True
            state.playable_actions = generate_playable_actions(state)
        else:
            payout, _ = yield_resources(state.board, state.resource_deck, number)
            for color, resource_deck in payout.items():
                # Atomically add to player's hand and remove from bank
                player_deck_add(state, color, resource_deck)
                state.resource_deck -= resource_deck

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
        to_discard = ResourceDeck.from_array(discarded)

        player_deck_subtract(state, action.color, to_discard)
        state.resource_deck += to_discard
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
                player_deck_draw(state, robbed_color, robbed_resource.value)
            player_deck_replenish(state, action.color, robbed_resource.value)

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
        cards_selected = ResourceDeck.from_array(action.value)
        if not player_can_play_dev(state, action.color, "YEAR_OF_PLENTY"):
            raise ValueError("Player cant play year of plenty now")
        if not state.resource_deck.includes(cards_selected):
            raise ValueError("Not enough resources of this type (these types?) in bank")
        player_deck_add(state, action.color, cards_selected)
        state.resource_deck -= cards_selected
        play_dev_card(state, action.color, "YEAR_OF_PLENTY")

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        mono_resource = action.value
        cards_stolen = ResourceDeck()
        if not player_can_play_dev(state, action.color, "MONOPOLY"):
            raise ValueError("Player cant play monopoly now")
        for color in state.colors:
            if not color == action.color:
                key = player_key(state, color)
                number_of_cards_to_steal = state.player_state[
                    f"{key}_{mono_resource.value}_IN_HAND"
                ]
                cards_stolen.replenish(number_of_cards_to_steal, mono_resource)
                player_deck_draw(
                    state, color, mono_resource.value, number_of_cards_to_steal
                )
        player_deck_add(state, action.color, cards_stolen)
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
        offering = ResourceDeck.from_array(
            filter(lambda r: r is not None, trade_offer[:-1])
        )
        asking = ResourceDeck.from_array(trade_offer[-1:])
        if not player_resource_deck_contains(state, action.color, offering):
            raise ValueError("Trying to trade without money")
        if not state.resource_deck.includes(asking):
            raise ValueError("Bank doenst have those cards")
        player_deck_subtract(state, action.color, offering)
        state.resource_deck += offering
        player_deck_add(state, action.color, asking)
        state.resource_deck -= asking

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    else:
        raise RuntimeError("Unknown ActionType " + str(action.action_type))

    # TODO: Think about possible-action/idea vs finalized-action design
    state.actions.append(action)
    return action
