"""Players and accumulators share one lifecycle."""

import random

import pytest

from catanatron import Color, Game, GameAccumulator
from catanatron.models.player import Player, RandomPlayer
from catanatron.observer import GameObserver
from catanatron.players.minimax import AlphaBetaPlayer


class Watcher(RandomPlayer):
    """A player that also observes."""

    def before(self, game):
        self.tiles_at_start = len(game.state.board.map.tiles)
        self.seen = []
        self.finished_as = None

    def step(self, game_before_action, action):
        self.seen.append(action)

    def after(self, game):
        self.finished_as = game.winning_color()


class Quiet(RandomPlayer):
    """A player that does not observe."""


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def test_player_is_a_game_observer():
    assert issubclass(Player, GameObserver)
    assert issubclass(GameAccumulator, GameObserver)


def test_before_fires_at_construction_with_the_board_visible():
    """Replaces reset_state(), which never got to see the game."""
    players = [Watcher(color) for color in Color]
    Game(players)
    for player in players:
        assert player.tiles_at_start > 0
        assert player.seen == []


def test_before_fires_once_per_game():
    players = [Watcher(color) for color in Color]
    Game(players)
    first = players[0].tiles_at_start
    Game(players)  # a second game re-fires before(), resetting per-game state
    assert players[0].seen == []
    assert players[0].tiles_at_start == first


def test_step_sees_every_action_including_opponents():
    random.seed(3)
    players = [Watcher(color) for color in Color]
    game = Game(players)
    game.play()
    for player in players:
        assert len(player.seen) == len(game.state.action_records)
    # ...and not just its own
    assert len({action.color for action in players[0].seen}) > 1


def test_after_fires_with_the_result():
    random.seed(3)
    players = [Watcher(color) for color in Color]
    game = Game(players)
    winner = game.play()
    assert all(player.finished_as == winner for player in players)


def test_players_that_do_not_override_step_are_skipped():
    """Keeps the per-tick cost off bots that don't observe."""
    assert Game([Quiet(color) for color in Color])._steppers == []
    game = Game([Watcher(Color.RED), Quiet(Color.BLUE)])
    assert [type(p).__name__ for p in game._steppers] == ["Watcher"]


def test_every_player_is_an_observer():
    """after() and before() reach all of them; only step() is filtered."""
    game = Game([Watcher(Color.RED), Quiet(Color.BLUE)])
    assert len(game.observers) == 2
    assert len(game._steppers) == 1


def test_accumulators_join_the_same_list():
    class Counter(GameAccumulator):
        def step(self, game_before_action, action):
            pass

    random.seed(5)
    game = Game([Quiet(Color.RED), Quiet(Color.BLUE)])
    counter = Counter()
    game.play(accumulators=[counter])
    assert counter in game.observers and counter in game._steppers


def test_search_does_not_fire_observers():
    """AlphaBeta copies and executes millions of times; hooks must not fire."""

    class CountingBot(AlphaBetaPlayer):
        def before(self, game):
            self.steps = 0

        def step(self, game_before_action, action):
            self.steps += 1

    random.seed(1)
    bot = CountingBot(Color.RED, AlphaBetaPlayer.Params(depth=2))
    game = Game([bot, RandomPlayer(Color.BLUE)])
    for _ in range(12):
        game.play_tick()

    # one step per action actually played, not per node explored
    assert bot.steps == len(game.state.action_records)


def test_accumulators_and_observing_players_both_fire():
    class Counter(GameAccumulator):
        def before(self, game):
            self.actions = 0

        def step(self, game_before_action, action):
            self.actions += 1

    random.seed(5)
    players = [Watcher(Color.RED), Quiet(Color.BLUE)]
    game = Game(players)
    counter = Counter()
    game.play(accumulators=[counter])
    assert counter.actions == len(players[0].seen) == len(game.state.action_records)


def test_observers_tolerate_construction_kwargs():
    """play_batch builds CLI accumulators as cls(players=..., game_config=...)."""

    class Bare(GameAccumulator):
        pass

    Bare(players=[], game_config=None)
    Bare()
