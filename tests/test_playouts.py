import pickle

from catanatron import Color, Game, RandomPlayer
from catanatron.players import playouts
from catanatron.players.playouts import _derive_playout_seeds


def test_playout_seeds_are_distinct_deterministic_and_do_not_advance_game():
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=7)
    original_rng_state = game.state.random.getstate()

    first = _derive_playout_seeds(game, 8)
    second = _derive_playout_seeds(game, 8)

    assert game.state.random.getstate() == original_rng_state
    assert first == second
    assert len(set(first)) == len(first)


class _SerializedPool:
    """Run inline, but deserialize each input as a separate worker would."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, params):
        return [fn(pickle.loads(pickle.dumps(param))) for param in params]


def test_multiprocessing_playouts_do_not_repeat_one_rng_stream(monkeypatch):
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=7, vps_to_win=3)
    original_rng_state = game.state.random.getstate()
    monkeypatch.setattr(playouts.multiprocessing, "Pool", _SerializedPool)

    winners = playouts.run_playouts(game, 16)

    assert sum(winners.values()) == 16
    assert len(winners) > 1
    assert game.state.random.getstate() == original_rng_state
