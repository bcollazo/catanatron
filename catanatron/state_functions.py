"""
Functions that mutate the given state accordingly. Core of game logic.
Some are helpers to _read_ information from state and keep the rest
of the code decoupled from state representation.
"""
import random

from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import BuildingType, Resource


def mantain_longest_road(state, previous_road_color, road_color, road_lengths):
    for color, length in road_lengths.items():
        key = player_key(state, color)
        state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = length
    if road_color is None:
        return  # do nothing

    if previous_road_color != road_color:
        winner_key = player_key(state, road_color)
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


# ===== State Getters
def player_key(state, color):
    return f"P{state.color_to_index[color]}"


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
def build_settlement(state, color, node_id, is_free):
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


def build_road(state, color, edge, is_free):
    state.buildings_by_color[color][BuildingType.ROAD].append(edge)

    key = player_key(state, color)
    state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        state.resource_deck += ResourceDeck.road_cost()  # replenish bank


def build_city(state, color, node_id):
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


def player_can_play_dev(state, color, dev_card):
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


def buy_dev_card(state, color, dev_card):
    key = player_key(state, color)

    assert state.player_state[f"{key}_SHEEP_IN_HAND"] >= 1
    assert state.player_state[f"{key}_WHEAT_IN_HAND"] >= 1
    assert state.player_state[f"{key}_ORE_IN_HAND"] >= 1

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
    assert state.player_state[f"{key}_{card}_IN_HAND"] >= amount
    state.player_state[f"{key}_{card}_IN_HAND"] -= amount


def player_deck_replenish(state, color, resource, amount=1):
    key = player_key(state, color)
    state.player_state[f"{key}_{resource}_IN_HAND"] += amount


def player_deck_random_draw(state, color):
    deck_array = player_deck_to_array(state, color)
    resource = random.choice(deck_array)
    player_deck_draw(state, color, resource.value)
    return resource


def play_dev_card(state, color, dev_card):
    if dev_card == "KNIGHT":
        previous_army_color, previous_army_size = get_larget_army_color(state)
    key = player_key(state, color)
    player_deck_draw(state, color, dev_card)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state.player_state[f"{key}_PLAYED_{dev_card}"] += 1
    if dev_card == "KNIGHT":
        mantain_largets_army(state, color, previous_army_color, previous_army_size)


def player_clean_turn(state, color):
    key = player_key(state, color)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    state.player_state[f"{key}_HAS_ROLLED"] = False
