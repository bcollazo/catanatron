from catanatron import Color, Game, RandomPlayer
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
