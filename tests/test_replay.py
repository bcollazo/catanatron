"""Replaying a game from its action log.

Every random outcome the engine produces is recorded in the ActionRecord --
the dice, the stolen resource, the drawn development card -- so a game can be
rebuilt by re-applying its log rather than by re-running the players. That is
what makes it safe to store a game as a document plus a log.
"""

import copy
import random

import pytest

from catanatron import Color, Game, Player
from catanatron.models.enums import ActionRecord
from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.serialization import action_from_json, state_from_json, state_to_json


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


class UnpredictablePlayer(Player):
    """Decides from a source the game knows nothing about, so replaying the
    game is impossible and replaying the log is the only way back."""

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self.private = random.Random()

    def decide(self, game, playable_actions):
        return playable_actions[self.private.randrange(len(playable_actions))]


def seats():
    return [RandomPlayer(color) for color in Color]


def played_game(players=None, ticks=200, seed=7):
    random.seed(seed)
    game = Game(players or seats())
    opening = copy.deepcopy(state_to_json(game))
    while game.winning_color() is None and game.state.num_turns < ticks:
        game.play_tick()
    return game, opening


def replay(opening, log, players):
    """Rebuild a game from a document and a log, touching no rng at all."""
    game = state_from_json(copy.deepcopy(opening), players)
    game.state.random = None  # any draw from the stream is a bug
    for payload, result in log:
        action = action_from_json(payload)
        game.execute(
            action, validate_action=False, action_record=ActionRecord(action, result)
        )
    return game


def documents_of(game, replayed):
    """Both as documents, with the rng put back so they can be serialized."""
    replayed.state.random = game.state.random
    return state_to_json(game), state_to_json(replayed)


def test_a_replayed_game_is_the_same_game():
    game, opening = played_game()
    replayed = replay(opening, state_to_json(game)["action_records"], seats())

    original, rebuilt = documents_of(game, replayed)
    assert rebuilt == original


def test_the_deck_keeps_its_order():
    """The order is hidden information, but it is still information: draw the
    way the original drew rather than plucking the named card out."""
    game, opening = played_game()
    log = state_to_json(game)["action_records"]
    assert any(a[1] == "BUY_DEVELOPMENT_CARD" for a, _ in log), "fixture buys none"

    replayed = replay(opening, log, seats())
    assert replayed.state.development_listdeck == game.state.development_listdeck


def test_a_deck_in_another_order_still_draws_the_right_cards():
    """Replaying onto a deck we do not have the true order for -- the cards
    drawn are still right, only the residue cannot be recovered."""
    game, opening = played_game()
    log = state_to_json(game)["action_records"]

    scrambled = copy.deepcopy(opening)
    random.Random(99).shuffle(scrambled["development_listdeck"])
    replayed = replay(scrambled, log, seats())

    assert sorted(replayed.state.development_listdeck) == sorted(
        game.state.development_listdeck
    )


def test_replay_does_not_depend_on_the_players():
    """The log records decisions, so who made them is irrelevant on the way
    back -- including players the game cannot reproduce."""
    game, opening = played_game([UnpredictablePlayer(c) for c in Color], ticks=60)
    log = state_to_json(game)["action_records"]

    for players in (
        [UnpredictablePlayer(c) for c in Color],
        [RandomPlayer(c) for c in Color],
        [AlphaBetaPlayer(c) for c in Color],
    ):
        original, rebuilt = documents_of(game, replay(opening, log, players))
        assert rebuilt == original


def test_the_same_seed_does_not_reproduce_an_unpredictable_game():
    """The control for the test above: without the log there is no way back."""
    first, _ = played_game([UnpredictablePlayer(c) for c in Color], ticks=60)
    second, _ = played_game([UnpredictablePlayer(c) for c in Color], ticks=60)
    assert [str(r.action) for r in first.state.action_records] != [
        str(r.action) for r in second.state.action_records
    ]
