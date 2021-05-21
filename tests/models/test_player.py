from catanatron.models.map import BaseMap
from catanatron.state import (
    State,
    player_clean_turn,
    player_deck_can_play,
    player_deck_replenish,
)
from catanatron.models.player import Color, SimplePlayer


def test_playable_cards():
    player = SimplePlayer(Color.RED)

    state = State([player], BaseMap())
    player_deck_replenish(state, Color.RED, "KNIGHT")
    player_clean_turn(state, Color.RED)

    assert player_deck_can_play(state, Color.RED, "KNIGHT")
