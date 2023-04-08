"""
Functions that mutate the given state accordingly. Core of game logic.
Some are helpers to _read_ information from state and keep the rest
of the code decoupled from state representation.
"""
import random
from typing import Optional

from catanatron.models.decks import ROAD_COST_FREQDECK, freqdeck_add
from catanatron.models.enums import (
    VICTORY_POINT,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    SETTLEMENT,
    CITY,
    ROAD,
    FastResource,
)


def maintain_longest_road(state, previous_road_color, road_color, road_lengths):
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


def maintain_largest_army(state, color, previous_army_color, previous_army_size):
    candidate_size = get_played_dev_cards(state, color, "KNIGHT")
    if candidate_size >= 3:
        if previous_army_color is None:
            winner_key = player_key(state, color)
            state.player_state[f"{winner_key}_HAS_ARMY"] = True
            state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
            state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2
        elif previous_army_size < candidate_size and previous_army_color != color:
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


def get_enemy_colors(colors, player_color):
    return filter(lambda c: c != player_color, colors)


def get_actual_victory_points(state, color):
    key = player_key(state, color)
    return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]


def get_visible_victory_points(state, color):
    key = player_key(state, color)
    return state.player_state[f"{key}_VICTORY_POINTS"]


def get_longest_road_color(state):
    for index in range(len(state.colors)):
        if state.player_state[f"P{index}_HAS_ROAD"]:
            return state.colors[index]
    return None


def get_largest_army(state):
    for index in range(len(state.colors)):
        if state.player_state[f"P{index}_HAS_ARMY"]:
            return (
                state.colors[index],
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


def get_player_freqdeck(state, color):
    """Returns a 'freqdeck' of a player's resource hand."""
    key = player_key(state, color)
    return [
        state.player_state[f"{key}_WOOD_IN_HAND"],
        state.player_state[f"{key}_BRICK_IN_HAND"],
        state.player_state[f"{key}_SHEEP_IN_HAND"],
        state.player_state[f"{key}_WHEAT_IN_HAND"],
        state.player_state[f"{key}_ORE_IN_HAND"],
    ]


# ===== State Mutators
def build_settlement(state, color, node_id, is_free):
    state.buildings_by_color[color][SETTLEMENT].append(node_id)

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
    state.buildings_by_color[color][ROAD].append(edge)

    key = player_key(state, color)
    state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if not is_free:
        state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, ROAD_COST_FREQDECK
        )  # replenish bank


def build_city(state, color, node_id):
    state.buildings_by_color[color][SETTLEMENT].remove(node_id)
    state.buildings_by_color[color][CITY].append(node_id)

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


def player_resource_freqdeck_contains(state, color, freqdeck):
    key = player_key(state, color)
    return (
        state.player_state[f"{key}_WOOD_IN_HAND"] >= freqdeck[0]
        and state.player_state[f"{key}_BRICK_IN_HAND"] >= freqdeck[1]
        and state.player_state[f"{key}_SHEEP_IN_HAND"] >= freqdeck[2]
        and state.player_state[f"{key}_WHEAT_IN_HAND"] >= freqdeck[3]
        and state.player_state[f"{key}_ORE_IN_HAND"] >= freqdeck[4]
    )


def player_can_play_dev(state, color, dev_card):
    key = player_key(state, color)
    return (
        not state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
        and state.player_state[f"{key}_{dev_card}_IN_HAND"] >= 1
    )


def player_freqdeck_add(state, color, freqdeck):
    key = player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] += freqdeck[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] += freqdeck[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] += freqdeck[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] += freqdeck[3]
    state.player_state[f"{key}_ORE_IN_HAND"] += freqdeck[4]


def player_freqdeck_subtract(state, color, freqdeck):
    key = player_key(state, color)
    state.player_state[f"{key}_WOOD_IN_HAND"] -= freqdeck[0]
    state.player_state[f"{key}_BRICK_IN_HAND"] -= freqdeck[1]
    state.player_state[f"{key}_SHEEP_IN_HAND"] -= freqdeck[2]
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= freqdeck[3]
    state.player_state[f"{key}_ORE_IN_HAND"] -= freqdeck[4]


def buy_dev_card(state, color, dev_card):
    key = player_key(state, color)

    assert state.player_state[f"{key}_SHEEP_IN_HAND"] >= 1
    assert state.player_state[f"{key}_WHEAT_IN_HAND"] >= 1
    assert state.player_state[f"{key}_ORE_IN_HAND"] >= 1

    state.player_state[f"{key}_{dev_card}_IN_HAND"] += 1
    if dev_card == VICTORY_POINT:
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1

    state.player_state[f"{key}_SHEEP_IN_HAND"] -= 1
    state.player_state[f"{key}_WHEAT_IN_HAND"] -= 1
    state.player_state[f"{key}_ORE_IN_HAND"] -= 1


def player_num_resource_cards(state, color, card: Optional[FastResource] = None):
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
        state.player_state[f"{key}_WOOD_IN_HAND"] * [WOOD]
        + state.player_state[f"{key}_BRICK_IN_HAND"] * [BRICK]
        + state.player_state[f"{key}_SHEEP_IN_HAND"] * [SHEEP]
        + state.player_state[f"{key}_WHEAT_IN_HAND"] * [WHEAT]
        + state.player_state[f"{key}_ORE_IN_HAND"] * [ORE]
    )


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
    player_deck_draw(state, color, resource)
    return resource


def play_dev_card(state, color, dev_card):
    if dev_card == "KNIGHT":
        previous_army_color, previous_army_size = get_largest_army(state)
    key = player_key(state, color)
    player_deck_draw(state, color, dev_card)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state.player_state[f"{key}_PLAYED_{dev_card}"] += 1
    if dev_card == "KNIGHT":
        maintain_largest_army(state, color, previous_army_color, previous_army_size)  # type: ignore


def player_clean_turn(state, color):
    key = player_key(state, color)
    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    state.player_state[f"{key}_HAS_ROLLED"] = False
