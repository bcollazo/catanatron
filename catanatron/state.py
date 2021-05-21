import random
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

# buildings_by_color.
def get_longest_road_color(state):
    for index in range(len(state.players)):
        if state.player_state[f"P{index}_HAS_ROAD"]:
            return state.players[index].color
    return None


def get_larget_army_color(state):
    for index in range(len(state.players)):
        if state.player_state[f"P{index}_HAS_ARMY"]:
            return (
                state.players[index].color,
                state.player_state[f"P{index}_PLAYED_KNIGHT"],
            )
    return None, None


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

            self.resource_deck = ResourceDeck.starting_bank()
            self.development_deck = DevelopmentDeck.starting_bank()

            # Auxiliary attributes to implement game logic
            self.current_prompt = None
            self.playable_actions = None
            self.buildings_by_color = {p.color: defaultdict(list) for p in players}
            self.actions = []  # log of all action taken by players
            self.tick_queue = initialize_tick_queue(self.players)
            self.current_player_index = 0
            self.num_turns = 0  # num_completed_turns

    def current_player(self):
        return self.players[self.current_player_index]


def initialize_tick_queue(players):
    """First player goes, settlement and road, ..."""
    tick_queue = []
    for seating in range(len(players)):
        tick_queue.append((seating, ActionPrompt.BUILD_FIRST_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    for seating in range(len(players) - 1, -1, -1):
        tick_queue.append((seating, ActionPrompt.BUILD_SECOND_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    tick_queue.append((0, ActionPrompt.ROLL))
    return tick_queue


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


def apply_action(state, action):
    if action.action_type == ActionType.END_TURN:
        next_player_index = (state.current_player_index + 1) % len(state.players)
        state.current_player_index = next_player_index
        player_clean_turn(state, state.players[next_player_index].color)
        state.tick_queue.append((next_player_index, ActionPrompt.ROLL))
        state.num_turns += 1
    elif action.action_type == ActionType.BUILD_FIRST_SETTLEMENT:
        node_id = action.value
        state.board.build_settlement(action.color, node_id, True)
        apply_settlement(state, action.color, node_id, True)
    elif action.action_type == ActionType.BUILD_SECOND_SETTLEMENT:
        node_id = action.value
        state.board.build_settlement(action.color, node_id, True)
        apply_settlement(state, action.color, node_id, True)
        # yield resources of second settlement
        key = player_key(state, action.color)
        for tile in state.board.map.adjacent_tiles[node_id]:
            if tile.resource != None:
                state.resource_deck.draw(1, tile.resource)
                state.player_state[f"{key}_{resource}_IN_HAND"] += 1
    elif action.action_type == ActionType.BUILD_SETTLEMENT:
        node_id = action.value
        previous_road_color, road_color, road_lengths = state.board.build_settlement(
            action.color, node_id, False
        )
        apply_settlement(state, action.color, node_id, False)
        state.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
        mantain_longest_road(state, previous_road_color, road_color, road_lengths)
    elif action.action_type == ActionType.BUILD_INITIAL_ROAD:
        edge = action.value
        state.board.build_road(action.color, edge)
        apply_road(state, action.color, edge, True)
    elif action.action_type == ActionType.BUILD_ROAD:
        edge = action.value
        previous_road_color, road_color, road_lengths = state.board.build_road(
            action.color, edge
        )
        apply_road(state, action.color, edge, False)
        state.resource_deck += ResourceDeck.road_cost()  # replenish bank
        mantain_longest_road(state, previous_road_color, road_color, road_lengths)
    elif action.action_type == ActionType.BUILD_CITY:
        node_id = action.value
        state.board.build_city(action.color, node_id)
        apply_city(state, action.color, node_id)
        state.resource_deck += ResourceDeck.city_cost()  # replenish bank
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

        apply_buy_dev(state, action.color, card.value)
        state.resource_deck += ResourceDeck.development_card_cost()

        action = Action(action.color, action.action_type, card)
    elif action.action_type == ActionType.ROLL:
        key = player_key(state, action.color)
        state.player_state[f"{key}_HAS_ROLLED"] = True

        dices = action.value or roll_dice()
        number = dices[0] + dices[1]

        if number == 7:
            seatings_to_discard = [
                seating
                for seating, player in enumerate(state.players)
                if player_num_resource_cards(state, player.color) > 7
            ]
            state.tick_queue.extend(
                [(seating, ActionPrompt.DISCARD) for seating in seatings_to_discard]
            )
            state.tick_queue.append(
                (state.current_player_index, ActionPrompt.MOVE_ROBBER)
            )
        else:
            payout, _ = yield_resources(state.board, state.resource_deck, number)
            for color, resource_deck in payout.items():
                # Atomically add to player's hand and remove from bank
                player_deck_add(state, color, resource_deck)
                state.resource_deck -= resource_deck

        action = Action(action.color, action.action_type, dices)
        state.tick_queue.append((state.current_player_index, ActionPrompt.PLAY_TURN))
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
    elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
        if not player_deck_can_play(state, action.color, "KNIGHT"):
            raise ValueError("Player cant play knight card now")
        (coordinate, robbed_color, robbed_resource) = action.value
        state.board.robber_coordinate = coordinate
        if robbed_color is not None:
            if robbed_resource is None:
                robbed_resource = player_deck_random_draw(state, robbed_color)
                action = Action(
                    action.color,
                    action.action_type,
                    (coordinate, robbed_color, resource),
                )
            else:  # for replay functionality
                player_deck_draw(state, robbed_color, robbed_resource.value)
            player_deck_replenish(state, action.color, robbed_resource.value)

        previous_army_color, previous_army_size = get_larget_army_color(state)
        apply_play_dev_card(state, action.color, "KNIGHT")
        mantain_largets_army(
            state, action.color, previous_army_color, previous_army_size
        )

    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        cards_selected = ResourceDeck.from_array(action.value)
        if not player_deck_can_play(state, action.color, "YEAR_OF_PLENTY"):
            raise ValueError("Player cant play year of plenty now")
        if not state.resource_deck.includes(cards_selected):
            raise ValueError("Not enough resources of this type (these types?) in bank")
        player_deck_add(state, action.color, cards_selected)
        state.resource_deck -= cards_selected
        apply_play_dev_card(state, action.color, "YEAR_OF_PLENTY")
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        mono_resource = action.value
        cards_stolen = ResourceDeck()
        if not player_deck_can_play(state, action.color, "MONOPOLY"):
            raise ValueError("Player cant play monopoly now")
        for player in state.players:
            if not player.color == action.color:
                key = player_key(state, player.color)
                number_of_cards_to_steal = state.player_state[
                    f"{key}_{mono_resource.value}_IN_HAND"
                ]
                cards_stolen.replenish(number_of_cards_to_steal, mono_resource)
                player_deck_draw(
                    state, player.color, mono_resource.value, number_of_cards_to_steal
                )
        player_deck_add(state, action.color, cards_stolen)
        apply_play_dev_card(state, action.color, MONOPOLY)
    elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
        (first_edge, second_edge) = action.value
        if not player_deck_can_play(state, action.color, "ROAD_BUILDING"):
            raise ValueError("Player cant play road building now")

        state.board.build_road(action.color, first_edge)
        previous_road_color, road_color, road_lengths = state.board.build_road(
            action.color, second_edge
        )
        apply_road(state, action.color, first_edge, True)
        apply_road(state, action.color, second_edge, True)
        apply_play_dev_card(state, action.color, "ROAD_BUILDING")
        mantain_longest_road(state, previous_road_color, road_color, road_lengths)
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
    else:
        raise RuntimeError("Unknown ActionType " + str(action.action_type))

    # TODO: Think about possible-action/idea vs finalized-action design
    state.actions.append(action)
    return action


def mantain_longest_road(state, previous_road_color, road_color, road_lengths):
    for color, length in road_lengths.items():
        key = player_key(state, color)
        state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = length
    if road_color is None:
        return  # do nothing

    if previous_road_color != road_color:
        winner_key = player_key(state, color)
        state.player_state[f"{winner_key}_HAS_ROAD"] = True
        state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
        state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2
        if previous_road_color is not None:
            loser_key = player_key(state, previous_road_color)
            state.player_state[f"{loser_key}_HAS_ROAD"] = False
            state.player_state[f"{loser_key}_VICTORY_POINTS"] -= 2
            state.player_state[f"{loser_key}_ACTUAL_VICTORY_POINTS"] -= 2


def mantain_largets_army(state, color, previous_army_color, previous_army_size):
    candidate_size = get_played_dev_cards(state, color, "KNIGHT")
    if candidate_size >= 3:
        if previous_army_color is None:
            winner_key = player_key(state, color)
            state.player_state[f"{winner_key}_HAS_ARMY"] = True
            state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
            state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2
        elif previous_army_size < candidate_size:
            # switch, remove previous points and award to new king
            winner_key = player_key(state, color)
            state.player_state[f"{winner_key}_HAS_ARMY"] = True
            state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
            state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2
            if previous_army_color is not None:
                loser_key = player_key(state, previous_army_color)
                state.player_state[f"{loser_key}_HAS_ARMY"] = False
                state.player_state[f"{loser_key}_VICTORY_POINTS"] -= 2
                state.player_state[f"{loser_key}_ACTUAL_VICTORY_POINTS"] -= 2
        # else: someone else has army and we dont compete


# TODO: Deprecated, just mantain army...
def compute_largest_army(state, actions):
    """Returns (color, count) for max army"""
    num_knights_to_players = defaultdict(set)
    for player in state.players:
        num_knight_played = get_played_dev_cards(state, player.color, "KNIGHT")
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


# ===== State Getters
def player_key(state, color):
    return f"P{state.color_to_index[color]}"


def player_has_rolled(state, color):
    key = player_key(state, color)
    return state.player_state[f"{key}_HAS_ROLLED"]


def get_longest_road_length(state, color):
    key = player_key(state, color)
    return state.player_state[key + "_LONGEST_ROAD_LENGTH"]


def get_played_dev_cards(state, color, dev_card=None):
    key = player_key(state, color)
    if dev_card is None:
        return (
            state.player_state[f"{key}_PLAYED_KNIGHT"]
            + state.player_state[f"{key}_PLAYED_MONOPOLY"]
            + state.player_state[f"{key}_PLAYED_ROAD_BUILDING"]
            + state.player_state[f"{key}_PLAYED_YEAR_OF_PLENTY"]
        )
    else:
        return state.player_state[f"{key}_PLAYED_{dev_card}"]


def get_dev_cards_in_hand(state, color, dev_card=None):
    key = player_key(state, color)
    if dev_card is None:
        return (
            state.player_state[f"{key}_KNIGHT_IN_HAND"]
            + state.player_state[f"{key}_MONOPOLY_IN_HAND"]
            + state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
            + state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
            + state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        )
    else:
        return state.player_state[f"{key}_{dev_card}_IN_HAND"]


def get_player_buildings(state, color_param, building_type_param):
    return state.buildings_by_color[color_param][building_type_param]


# ===== State Mutators
def apply_settlement(state, color, node_id, is_free):
    state.buildings_by_color[color][BuildingType.SETTLEMENT].append(node_id)

    key = player_key(state, color)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] -= 1

    state.player_state[f"{key}_VICTORY_POINTS"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        state.player_state[f"{key}_SHEEP_IN_HAND"] -= 1
        state.player_state[f"{key}_WHEAT_IN_HAND"] -= 1


def apply_road(state, color, edge, is_free):
    state.buildings_by_color[color][BuildingType.ROAD].append(edge)

    key = player_key(state, color)
    state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1


def apply_city(state, color, node_id):
    state.buildings_by_color[color][BuildingType.SETTLEMENT].remove(node_id)
    state.buildings_by_color[color][BuildingType.CITY].append(node_id)

    key = player_key(state, color)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] += 1
    state.player_state[f"{key}_CITIES_AVAILABLE"] -= 1

    state.player_state[f"{key}_VICTORY_POINTS"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    state.player_state[f"{key}_WHEAT_IN_HAND"] -= 2
    state.player_state[f"{key}_ORE_IN_HAND"] -= 3


# ===== Deck Functions
def player_can_afford_dev_card(state, color):
    key = player_key(state, color)
    return (
        state.player_state[f"{key}_SHEEP_IN_HAND"] >= 1
        and state.player_state[f"{key}_WHEAT_IN_HAND"] >= 1
        and state.player_state[f"{key}_ORE_IN_HAND"] >= 1
    )


def player_resource_deck_contains(state, color, deck):
    key = player_key(state, color)
    return (
        state.player_state[f"{key}_WOOD_IN_HAND"] >= deck.array[0]
        and state.player_state[f"{key}_BRICK_IN_HAND"] >= deck.array[1]
        and state.player_state[f"{key}_SHEEP_IN_HAND"] >= deck.array[2]
        and state.player_state[f"{key}_WHEAT_IN_HAND"] >= deck.array[3]
        and state.player_state[f"{key}_ORE_IN_HAND"] >= deck.array[4]
    )


def player_resource_deck_reset(state, color):
    key = player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] = 0
    state.player_state[f"{key}_BRICK_IN_HAND"] = 0
    state.player_state[f"{key}_SHEEP_IN_HAND"] = 0
    state.player_state[f"{key}_WHEAT_IN_HAND"] = 0
    state.player_state[f"{key}_ORE_IN_HAND"] = 0


def player_deck_can_play(state, color, dev_card):
    key = player_key(state, color)
    return (
        not state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
        and state.player_state[f"{key}_{dev_card}_IN_HAND"] >= 1
    )


def player_deck_add(state, color, deck):
    key = player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] += deck.array[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] += deck.array[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] += deck.array[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] += deck.array[3]
    state.player_state[f"{key}_ORE_IN_HAND"] += deck.array[4]


def apply_buy_dev(state, color, dev_card):
    key = player_key(state, color)
    state.player_state[f"{key}_{dev_card}_IN_HAND"] += 1
    state.player_state[f"{key}_SHEEP_IN_HAND"] -= 1
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= 1
    state.player_state[f"{key}_ORE_IN_HAND"] -= 1


def player_num_resource_cards(state, color, card=None):
    key = player_key(state, color)
    if card is None:
        return (
            state.player_state[f"{key}_WOOD_IN_HAND"]
            + state.player_state[f"{key}_BRICK_IN_HAND"]
            + state.player_state[f"{key}_SHEEP_IN_HAND"]
            + state.player_state[f"{key}_WHEAT_IN_HAND"]
            + state.player_state[f"{key}_ORE_IN_HAND"]
        )
    else:
        return state.player_state[f"{key}_{card}_IN_HAND"]


def player_num_dev_cards(state, color):
    key = player_key(state, color)
    return (
        state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
        + state.player_state[f"{key}_MONOPOLY_IN_HAND"]
        + state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        + state.player_state[f"{key}_KNIGHT_IN_HAND"]
        + state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
    )


def player_deck_to_array(state, color):
    key = player_key(state, color)
    return (
        state.player_state[f"{key}_WOOD_IN_HAND"] * [Resource.WOOD]
        + state.player_state[f"{key}_BRICK_IN_HAND"] * [Resource.BRICK]
        + state.player_state[f"{key}_SHEEP_IN_HAND"] * [Resource.SHEEP]
        + state.player_state[f"{key}_WHEAT_IN_HAND"] * [Resource.WHEAT]
        + state.player_state[f"{key}_ORE_IN_HAND"] * [Resource.ORE]
    )


def player_deck_subtract(state, color, to_discard):
    key = player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] -= to_discard.array[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] -= to_discard.array[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] -= to_discard.array[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= to_discard.array[3]
    state.player_state[f"{key}_ORE_IN_HAND"] -= to_discard.array[4]


def player_deck_draw(state, color, card, amount=1):
    key = player_key(state, color)
    state.player_state[f"{key}_{card}_IN_HAND"] -= amount


def player_deck_replenish(state, color, resource, amount=1):
    key = player_key(state, color)
    state.player_state[f"{key}_{resource}_IN_HAND"] += amount


def player_deck_random_draw(state, color):
    deck_array = player_deck_to_array(state, color)
    resource = random.choice(deck_array)
    player_deck_draw(state, color, resource.value)
    return resource


def apply_play_dev_card(state, color, dev_card):
    key = player_key(state, color)
    player_deck_draw(state, color, dev_card)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state.player_state[f"{key}_PLAYED_{dev_card}"] += 1


def player_clean_turn(state, color):
    key = player_key(state, color)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    state.player_state[f"{key}_HAS_ROLLED"] = False
