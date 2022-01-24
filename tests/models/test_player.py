from catanatron.state import (
    State,
    player_clean_turn,
    player_can_play_dev,
    player_deck_replenish,
)
from catanatron.models.player import Color, SimplePlayer


def test_playable_cards():
    player = SimplePlayer(Color.RED)

    state = State([player])
    player_deck_replenish(state, Color.RED, "KNIGHT")
    player_clean_turn(state, Color.RED)

    assert player_can_play_dev(state, Color.RED, "KNIGHT")
