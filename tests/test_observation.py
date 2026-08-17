import re

import pytest

from catanatron.features import create_sample
from catanatron.game import Game
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.models.observation import Observation
from catanatron.models.observation_agent import ObservationAgent
from catanatron.models.perspective_player import (
    PerspectivePlayer,
    _build_public_state,
    _sanitize_history,
    _sanitize_record,
)
from catanatron.models.player import Color, Player, RandomPlayer


class RecorderAgent(ObservationAgent):
    """Trivial fair bot that records every Observation it receives."""

    def __init__(self, color):
        super().__init__(color)
        self.observations = []
        self.calls_since_reset = 0

    def decide_observation(self, observation, playable_actions):
        self.observations.append(observation)
        self.calls_since_reset += 1
        return playable_actions[0]

    def reset_state(self):
        super().reset_state()
        self.calls_since_reset = 0


HIDDEN_PATTERNS = (
    [re.compile(rf"^P[1-9]_{resource}_IN_HAND$") for resource in RESOURCES]
    + [re.compile(rf"^P[1-9]_{card}_IN_HAND$") for card in DEVELOPMENT_CARDS]
    + [
        re.compile(r"^P[1-9]_ACTUAL_VICTORY_POINTS$"),
    ]
)


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


def _move_robber(color, coordinate, robbed_color, stolen):
    return ActionRecord(
        Action(color, ActionType.MOVE_ROBBER, (coordinate, robbed_color)), stolen
    )


# ===== Structural separation tests =====
def test_observation_agent_is_not_a_player():
    agent = ObservationAgent(Color.RED)
    assert not isinstance(agent, Player)
    assert not hasattr(agent, "decide")
    assert not hasattr(agent, "is_bot")


def test_observation_is_pure_data_snapshot():
    obs = Observation(
        color=Color.RED,
        features={"P0_WOOD_IN_HAND": 3},
        public_history=[],
        current_prompt=None,
        current_trade=None,
        acceptees=(),
    )
    assert obs.color == Color.RED
    assert obs.features["P0_WOOD_IN_HAND"] == 3
    assert obs.public_state == {}
    assert not hasattr(obs, "_game")
    assert not hasattr(obs, "state")


# ===== Sanitizer unit tests (per row of the §3.3 table) =====
def test_sanitize_redacts_opponent_buy_dev_card():
    record = ActionRecord(
        Action(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"), "KNIGHT"
    )
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
    assert seen.action.value is None
    assert seen.result is None


def test_sanitize_keeps_own_buy_dev_card():
    record = ActionRecord(
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"), "KNIGHT"
    )
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.value == "KNIGHT"
    assert seen.result == "KNIGHT"


def test_sanitize_spectator_does_not_see_stolen_card():
    record = _move_robber(Color.BLUE, (0, 0, 0), Color.ORANGE, "WOOD")
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.value == ((0, 0, 0), Color.ORANGE)
    assert seen.result is None


def test_sanitize_victim_sees_stolen_card():
    record = _move_robber(Color.BLUE, (0, 0, 0), Color.RED, "WOOD")
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.value == ((0, 0, 0), Color.RED)
    assert seen.result == "WOOD"


def test_sanitize_keeps_own_move_robber():
    record = _move_robber(Color.RED, (0, 0, 0), Color.BLUE, "WOOD")
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.value == ((0, 0, 0), Color.BLUE)
    assert seen.result == "WOOD"


def test_sanitize_discards_are_public():
    record = ActionRecord(
        Action(Color.BLUE, ActionType.DISCARD_RESOURCE, "WOOD"), "WOOD"
    )
    seen = _sanitize_record(record, Color.RED)
    assert seen.action.value == "WOOD"
    assert seen.result == "WOOD"


def test_sanitize_passes_through_public_actions():
    record = ActionRecord(Action(Color.BLUE, ActionType.ROLL, None), (3, 4))
    assert _sanitize_record(record, Color.RED) == record


# ===== public_state (structured public surface) =====
def _play_recorded_game(seed=0):
    """Plays a game through the decide_fn seam, recording for the RED agent
    each Observation plus a snapshot of the engine state it was built from."""
    recorder = RecorderAgent(Color.RED)
    game = _make_game(
        recorder,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=seed,
    )
    observations = []
    snapshots = []

    def decide_fn(player, g, actions):
        if player is not recorder:
            return player.decide(g, actions)
        observation = Observation(
            color=player.color,
            features=create_sample(g, player.color),
            public_history=_sanitize_history(g, player.color),
            current_prompt=g.state.current_prompt,
            current_trade=g.state.current_trade,
            acceptees=g.state.acceptees,
            public_state=_build_public_state(g),
        )
        observations.append(observation)
        snapshots.append(
            {
                "colors": tuple(g.state.colors),
                "buildings": dict(g.state.board.buildings),
                "roads": dict(g.state.board.roads),
                "robber_coordinate": g.state.board.robber_coordinate,
                "road_color": g.state.board.road_color,
                "road_length": g.state.board.road_length,
                "player_state": dict(g.state.player_state),
            }
        )
        return player.decide_observation(observation, actions)

    game.play(decide_fn=decide_fn)
    return observations, snapshots


def test_public_state_matches_engine_board():
    observations, snapshots = _play_recorded_game(seed=2)
    assert observations
    for obs, snap in zip(observations, snapshots):
        public_board = obs.public_state["board"]
        assert public_board["robber_coordinate"] == snap["robber_coordinate"]
        assert public_board["longest_road_color"] == snap["road_color"]
        assert public_board["longest_road_length"] == snap["road_length"]
        assert public_board["buildings"] == snap["buildings"]
        expected_roads = {
            edge: color for edge, color in snap["roads"].items() if edge[0] < edge[1]
        }
        assert public_board["roads"] == expected_roads


def test_public_state_per_player_public_counts():
    observations, snapshots = _play_recorded_game(seed=3)
    for obs, snap in zip(observations, snapshots):
        players = obs.public_state["players"]
        assert set(players.keys()) == set(snap["colors"])
        player_state = snap["player_state"]
        for color, public in players.items():
            key = f"P{snap['colors'].index(color)}"
            assert public["public_vps"] == player_state[f"{key}_VICTORY_POINTS"]
            assert public["has_army"] == player_state[f"{key}_HAS_ARMY"]
            assert public["has_road"] == player_state[f"{key}_HAS_ROAD"]
            assert (
                public["longest_road_length"]
                == player_state[f"{key}_LONGEST_ROAD_LENGTH"]
            )
            assert public["roads_left"] == player_state[f"{key}_ROADS_AVAILABLE"]
            assert (
                public["settlements_left"]
                == player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
            )
            assert public["cities_left"] == player_state[f"{key}_CITIES_AVAILABLE"]
            assert public["has_rolled"] == player_state[f"{key}_HAS_ROLLED"]
            assert public["hand_resource_count"] == sum(
                player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES
            )
            assert public["hand_dev_count"] == sum(
                player_state[f"{key}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS
            )


def test_public_state_leaks_no_opponent_private_info():
    observations, _ = _play_recorded_game(seed=4)
    for obs in observations:
        violations = []
        scan(obs.public_state, "public_state", violations)
        assert not violations, violations
        for public in obs.public_state["players"].values():
            assert "actual_vps" not in public
            assert not any(key.endswith("_IN_HAND") for key in public)


def test_build_public_state_is_standalone():
    game = _make_game(RandomPlayer(Color.RED), seed=5)
    state = _build_public_state(game)
    assert state["board"]["buildings"] == {}
    assert state["board"]["robber_coordinate"] == game.state.board.robber_coordinate
    assert set(state["players"].keys()) == set(game.state.colors)


# ===== Fairness invariants over full games =====
@pytest.mark.parametrize("seed", range(5))
def test_no_opponent_private_info_leaks(seed):
    recorder = RecorderAgent(Color.RED)
    game = _make_game(
        PerspectivePlayer(recorder),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=seed,
    )
    game.play()
    assert len(recorder.observations) > 0

    for obs in recorder.observations:
        assert isinstance(obs, Observation)
        assert not hasattr(obs, "_game")
        assert not hasattr(obs, "state")

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
    recorder = RecorderAgent(Color.RED)
    game = _make_game(
        PerspectivePlayer(recorder),
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
    recorder = RecorderAgent(Color.RED)
    game = _make_game(
        PerspectivePlayer(recorder),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=42,
    )
    winner = game.play()
    assert winner is None or winner in game.state.colors
    assert len(recorder.observations) > 0


def test_decide_fn_seam_drives_bare_agents():
    agents = [
        RecorderAgent(Color.RED),
        RecorderAgent(Color.BLUE),
        RecorderAgent(Color.ORANGE),
    ]
    game = Game(agents, seed=7)

    def decide_fn(player, g, actions):
        return player.decide_observation(
            Observation(
                color=player.color,
                features=create_sample(g, player.color),
                public_history=_sanitize_history(g, player.color),
                current_prompt=g.state.current_prompt,
                current_trade=g.state.current_trade,
                acceptees=g.state.acceptees,
                public_state=_build_public_state(g),
            ),
            actions,
        )

    game.play(decide_fn=decide_fn)
    assert all(len(agent.observations) > 0 for agent in agents)


def test_reset_state_hook_on_agent():
    agent = ObservationAgent(Color.RED)
    agent.reset_state()  # must not raise


def test_reset_state_delegates_to_agent():
    recorder = RecorderAgent(Color.RED)
    player = PerspectivePlayer(recorder)
    game = _make_game(
        player,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        seed=11,
    )
    game.play()
    assert recorder.calls_since_reset > 0
    player.reset_state()
    assert recorder.calls_since_reset == 0
