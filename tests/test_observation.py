import re

import pytest

from catanatron.game import Game
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.models.observation import Observation
from catanatron.models.perspective_player import PerspectivePlayer
from catanatron.models.player import Color, RandomPlayer


class RecorderPlayer(PerspectivePlayer):
    """Trivial fair bot that records every Observation it receives."""

    def __init__(self, color):
        super().__init__(color, is_bot=True)
        self.observations = []
        self.calls_since_reset = 0

    def decide_observation(self, observation, playable_actions):
        self.observations.append(observation)
        self.calls_since_reset += 1
        return playable_actions[0]

    def reset_state(self):
        super().reset_state()
        self.calls_since_reset = 0


HIDDEN_PATTERNS = [
    re.compile(rf"^P[1-9]_{resource}_IN_HAND$") for resource in RESOURCES
] + [
    re.compile(rf"^P[1-9]_{card}_IN_HAND$") for card in DEVELOPMENT_CARDS
] + [
    re.compile(r"^P[1-9]_ACTUAL_VICTORY_POINTS$"),
]


def scan(value, path, violations):
    if isinstance(value, dict):
        for k, v in value.items():
            if any(p.match(str(k)) for p in HIDDEN_PATTERNS):
                violations.append(f"{path}.{k}")
            scan(v, f"{path}.{k}", violations)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            scan(v, f"{path}[{i}]", violations)


def _make_game(*players, seed=0):
    return Game(list(players), seed=seed)


def _game_with_records(records):
    game = _make_game(RandomPlayer(Color.RED), RandomPlayer(Color.BLUE))
    game.state.action_records = records
    return game


def _move_robber(color, coordinate, robbed_color, stolen):
    return ActionRecord(
        Action(color, ActionType.MOVE_ROBBER, (coordinate, robbed_color)), stolen
    )


# ===== Sanitizer unit tests (per row of the §3.3 table) =====
def test_sanitizer_redacts_opponent_buy_dev_card():
    record = ActionRecord(
        Action(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"), "KNIGHT"
    )
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
    assert seen.action.value is None
    assert seen.result is None


def test_sanitizer_keeps_own_buy_dev_card():
    record = ActionRecord(
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"), "KNIGHT"
    )
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.value == "KNIGHT"
    assert seen.result == "KNIGHT"


def test_sanitizer_spectator_does_not_see_stolen_card():
    record = _move_robber(Color.BLUE, (0, 0, 0), Color.ORANGE, "WOOD")
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.value == ((0, 0, 0), Color.ORANGE)
    assert seen.result is None


def test_sanitizer_victim_sees_stolen_card():
    record = _move_robber(Color.BLUE, (0, 0, 0), Color.RED, "WOOD")
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.value == ((0, 0, 0), Color.RED)
    assert seen.result == "WOOD"


def test_sanitizer_keeps_own_move_robber():
    record = _move_robber(Color.RED, (0, 0, 0), Color.BLUE, "WOOD")
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.value == ((0, 0, 0), Color.BLUE)
    assert seen.result == "WOOD"


def test_sanitizer_discards_are_public():
    record = ActionRecord(
        Action(Color.BLUE, ActionType.DISCARD_RESOURCE, "WOOD"), "WOOD"
    )
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen.action.value == "WOOD"
    assert seen.result == "WOOD"


def test_sanitizer_passes_through_public_actions():
    record = ActionRecord(Action(Color.BLUE, ActionType.ROLL, None), (3, 4))
    obs = Observation(_game_with_records([record]), Color.RED)
    (seen,) = obs.public_history
    assert seen == record


# ===== Fairness invariants over full games =====
@pytest.mark.parametrize("seed", range(5))
def test_no_opponent_private_info_leaks(seed):
    recorder = RecorderPlayer(Color.RED)
    game = _make_game(
        recorder,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=seed,
    )
    game.play()
    assert len(recorder.observations) > 0

    for obs in recorder.observations:
        violations = []
        scan(obs.features, "features", violations)
        assert not violations, violations

        for record in obs.public_history:
            action = record.action
            if action.color == obs.color:
                continue  # own records retain full detail
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                assert action.value is None
                assert record.result is None
            elif action.action_type == ActionType.MOVE_ROBBER:
                if action.value is not None and action.value[1] == obs.color:
                    assert record.result is not None  # victim knows the card
                else:
                    assert record.result is None


@pytest.mark.parametrize("seed", range(3))
def test_public_hand_counts_reachable(seed):
    recorder = RecorderPlayer(Color.RED)
    game = _make_game(
        recorder,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=seed,
    )
    game.play()

    for obs in recorder.observations:
        features = obs.features
        for opponent_index in range(1, len(game.state.colors)):
            assert f"P{opponent_index}_NUM_RESOURCES_IN_HAND" in features
            assert f"P{opponent_index}_NUM_DEVS_IN_HAND" in features
            assert f"P{opponent_index}_PUBLIC_VPS" in features


# ===== Seam tests =====
def test_perspective_player_plays_game_to_completion():
    recorder = RecorderPlayer(Color.RED)
    game = _make_game(
        recorder,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=42,
    )
    winner = game.play()
    assert winner is None or winner in game.state.colors
    assert len(recorder.observations) > 0


def test_decide_fn_seam():
    players = [
        RecorderPlayer(Color.RED),
        RecorderPlayer(Color.BLUE),
        RecorderPlayer(Color.ORANGE),
    ]
    game = Game(players, seed=7)

    def decide_fn(player, g, actions):
        return player.decide_observation(Observation(g, player.color), actions)

    game.play(decide_fn=decide_fn)
    assert all(len(p.observations) > 0 for p in players)


def test_reset_state_via_inheritance():
    player = PerspectivePlayer(Color.RED)
    assert "reset_state" not in PerspectivePlayer.__dict__  # inherited, not redefined
    player.reset_state()  # must not raise


def test_reset_state_between_games():
    recorder = RecorderPlayer(Color.RED)
    for _ in range(2):
        game = _make_game(
            recorder,
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.ORANGE),
            seed=11,
        )
        game.play()
        assert recorder.calls_since_reset > 0
        recorder.reset_state()
        assert recorder.calls_since_reset == 0
    assert len(recorder.observations) > 0
