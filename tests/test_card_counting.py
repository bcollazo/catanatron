import pytest

from catanatron.game import Game
from catanatron.models.enums import (
    RESOURCES,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.card_counting import CardCounter
from catanatron.state_functions import player_key


def true_hand(state, color):
    key = player_key(state, color)
    return tuple(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


def make_counter():
    colors = (Color.RED, Color.BLUE, Color.WHITE)
    return CardCounter(colors), colors


def test_disabled_by_default():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    assert game.state.card_counter is None
    assert game.copy().state.card_counter is None


def test_public_gain_and_spend_are_known_to_everyone():
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [2, 1, 0, 0, 0])
    counter.spend(0, [1, 1, 0, 0, 0])
    for observer in (red, blue, white):
        assert counter.known_hand(observer, red) == (1, 0, 0, 0, 0)
        assert counter.num_unknown_cards(observer, red) == 0


def test_steal_visible_only_to_thief_and_victim():
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [1, 1, 0, 0, 0])  # RED holds wood + brick
    counter.steal(1, 0, "BRICK")  # BLUE steals brick from RED

    # BLUE (thief) and RED (victim) know exactly.
    for observer in (red, blue):
        assert counter.known_hand(observer, blue) == (0, 1, 0, 0, 0)
        assert counter.known_hand(observer, red) == (1, 0, 0, 0, 0)
    # WHITE only knows a card moved: 50/50 wood or brick.
    assert counter.num_cards(white, blue) == 1
    assert counter.num_unknown_cards(white, blue) == 1
    assert counter.steal_probabilities(white, blue) == pytest.approx((0.5, 0.5, 0, 0, 0))
    assert counter.steal_probabilities(white, red) == pytest.approx((0.5, 0.5, 0, 0, 0))


def test_later_spend_resolves_linked_uncertainty():
    """The scenario from the issue: a spend reveals what an earlier,
    hidden steal must have been, for both thief and victim."""
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [1, 1, 0, 0, 0])
    counter.steal(1, 0, "BRICK")  # WHITE can't tell which
    # RED is then seen paying a wood: so RED had a wood left, so the
    # stolen card must have been the brick.
    counter.spend(0, [1, 0, 0, 0, 0])
    assert counter.known_hand(white, blue) == (0, 1, 0, 0, 0)
    assert counter.num_unknown_cards(white, blue) == 0
    assert counter.num_cards(white, red) == 0


def test_copy_is_independent():
    counter, (red, blue, white) = make_counter()
    other = counter.copy()
    other.gain(0, [1, 0, 0, 0, 0])
    assert counter.num_cards(red, red) == 0
    assert other.num_cards(red, red) == 1


@pytest.mark.parametrize("seed", range(15))
def test_beliefs_always_contain_the_truth(seed):
    """Play whole games and check, after every action, that every
    observer's belief (a) includes the real hands and (b) never claims
    certainty about a card the target doesn't have. Own hand is exact."""
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)]
    game = Game(players, seed=seed, card_counting=True)
    counter = game.state.card_counter
    checked = 0
    while game.winning_color() is None and game.state.num_turns < 300:
        game.play_tick()
        for observer in game.state.colors:
            for target in game.state.colors:
                truth = true_hand(game.state, target)
                dist = counter.hand_distribution(observer, target)
                assert truth in dist, (observer, target)
                assert counter.num_cards(observer, target) == sum(truth)
                if observer == target:
                    assert dist == {truth: pytest.approx(1.0)}
        checked += 1
    assert not counter.degraded
    assert checked > 50


def test_state_copy_keeps_counting_independently():
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    game = Game(players, seed=1, card_counting=True)
    for _ in range(60):
        game.play_tick()
    copy = game.copy()
    for _ in range(20):
        copy.play_tick()
    for observer in game.state.colors:
        for target in game.state.colors:
            assert true_hand(game.state, target) in game.state.card_counter.hand_distribution(
                observer, target
            )
            assert true_hand(copy.state, target) in copy.state.card_counter.hand_distribution(
                observer, target
            )


def test_discards_are_public():
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    game = Game(players, seed=3, card_counting=True)
    counter = game.state.card_counter
    red = game.state.colors[0]
    counter.gain(0, [3, 0, 0, 0, 1])
    counter.spend(0, [0, 0, 0, 0, 1])
    assert counter.known_hand(game.state.colors[1], red)[4] == 0
