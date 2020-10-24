from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import Resource, DevelopmentCard


def test_playable_cards():
    player = SimplePlayer(Color.RED)
    player.development_deck.replenish(1, DevelopmentCard.KNIGHT)
    player.start_turn()

    assert player.can_play_knight()
