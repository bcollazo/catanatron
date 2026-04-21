import numpy as np

from capstone_agent.Placement.benchmark_placement import HybridPlayer
from capstone_agent.Placement.placement_action_space import (
    PlacementPrompt,
    capstone_action_to_local,
    capstone_mask_to_local_mask,
    prompt_from_game,
)
from capstone_agent.Placement.placement_features import (
    get_compact_placement_observation,
    validate_static_node_feature_order,
)
from capstone_agent.Placement.placement_supervised_dataset import (
    OPENING_STEP_COUNT,
    CompactPlacementAccumulator,
    decode_opening_action_onehot,
    encode_static_node_features,
    encode_opening_action_onehot,
    extract_local_target,
    load_chunk_records,
    opening_step_actor_seat,
    reconstruct_compact_x,
    reconstruct_local_mask,
    save_chunk_records,
)
from capstone_agent.Placement.router_search_player import (
    EnginePlayerMainAgentAdapter,
    RouterCapstonePlayer,
    get_capstone_action_mask,
)
from catanatron.game import Game
from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index
from catanatron.models.enums import ActionPrompt
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.minimax import AlphaBetaPlayer


class FirstValidPlacementAgent:
    def select_action(self, state, mask, **_kwargs):
        del state
        action = int(np.flatnonzero(np.asarray(mask) > 0.5)[0])
        return action, 0.0, 0.0


def _other_color(game, self_color):
    return next(color for color in game.state.colors if color != self_color)


def _collect_opening_trace(game):
    prompts = []
    actor_colors = []
    while game.state.is_initial_build_phase:
        actor_colors.append(game.state.current_color())
        prompts.append(game.state.current_prompt)
        game.play_tick()
    return actor_colors, prompts


def test_opening_schedule_matches_two_player_engine():
    game = Game(
        players=[SimplePlayer(Color.BLUE), SimplePlayer(Color.RED)],
        seed=17,
    )
    actor_colors, prompts = _collect_opening_trace(game)

    assert len(actor_colors) == OPENING_STEP_COUNT
    assert prompts == [
        ActionPrompt.BUILD_INITIAL_SETTLEMENT,
        ActionPrompt.BUILD_INITIAL_ROAD,
        ActionPrompt.BUILD_INITIAL_SETTLEMENT,
        ActionPrompt.BUILD_INITIAL_ROAD,
        ActionPrompt.BUILD_INITIAL_SETTLEMENT,
        ActionPrompt.BUILD_INITIAL_ROAD,
        ActionPrompt.BUILD_INITIAL_SETTLEMENT,
        ActionPrompt.BUILD_INITIAL_ROAD,
    ]

    first_actor = actor_colors[0]
    second_actor = next(color for color in game.state.colors if color != first_actor)
    assert actor_colors == [
        first_actor,
        first_actor,
        second_actor,
        second_actor,
        second_actor,
        second_actor,
        first_actor,
        first_actor,
    ]

    seat_by_color = {color: idx for idx, color in enumerate(game.state.colors)}
    actor_seats = [seat_by_color[color] for color in actor_colors]
    assert actor_seats == [opening_step_actor_seat(step_idx) for step_idx in range(OPENING_STEP_COUNT)]


def test_reconstruction_matches_live_engine_all_eight_plies():
    game = Game(
        players=[SimplePlayer(Color.BLUE), SimplePlayer(Color.RED)],
        seed=23,
    )
    static_node_features = encode_static_node_features(game)
    prior_actions = []

    for step_idx in range(OPENING_STEP_COUNT):
        current_color = game.state.current_color()
        opp_color = _other_color(game, current_color)
        prompt = prompt_from_game(game)

        live_x = get_compact_placement_observation(game, current_color, opp_color)
        reconstructed_x = reconstruct_compact_x(static_node_features, prior_actions, step_idx)
        np.testing.assert_allclose(live_x, reconstructed_x, atol=1e-6)

        live_mask = capstone_mask_to_local_mask(
            get_capstone_action_mask(game.playable_actions),
            prompt,
        )
        reconstructed_mask = reconstruct_local_mask(prior_actions, step_idx)
        np.testing.assert_array_equal(live_mask, reconstructed_mask)

        action_record = game.play_tick()
        onehot = encode_opening_action_onehot(action_record.action)
        capstone_idx = catanatron_action_to_capstone_index(action_record.action)
        assert decode_opening_action_onehot(onehot) == capstone_idx
        assert extract_local_target(onehot, step_idx) == capstone_action_to_local(
            prompt,
            capstone_idx,
        )
        prior_actions.append(onehot)


def test_static_node_feature_order_matches_live_projection_across_seeds():
    for seed in (0, 1, 7, 23, 41):
        game = Game(
            players=[SimplePlayer(Color.BLUE), SimplePlayer(Color.RED)],
            seed=seed,
        )

        for _ in range(OPENING_STEP_COUNT):
            current_color = game.state.current_color()
            validate_static_node_feature_order(
                game,
                current_color,
                _other_color(game, current_color),
            )
            game.play_tick()


def test_compact_chunk_schema_requires_metadata_fields(tmp_path):
    chunk_path = tmp_path / 'bad_chunk.npz'
    np.savez_compressed(
        chunk_path,
        schema_version=np.asarray([1], dtype=np.int64),
        static_node_features=np.zeros((1, 54, 11), dtype=np.float32),
        opening_actions_onehot=np.zeros((1, 8, 126), dtype=np.float32),
        winner_is_first_actor=np.asarray([True], dtype=np.bool_),
    )

    try:
        load_chunk_records(str(chunk_path))
    except ValueError as exc:
        message = str(exc)
        assert 'missing required fields' in message
        assert 'game_id' in message
    else:  # pragma: no cover - direct-call test guard
        raise AssertionError('Expected compact chunk load to fail without metadata')


def test_router_capstone_player_matches_hybrid_opening_path():
    placement_agent = FirstValidPlacementAgent()
    router_blue = RouterCapstonePlayer(
        Color.BLUE,
        placement_agent,
        EnginePlayerMainAgentAdapter(SimplePlayer(Color.BLUE)),
    )
    hybrid_blue = HybridPlayer(Color.BLUE, placement_agent)

    router_game = Game(players=[router_blue, SimplePlayer(Color.RED)], seed=31)
    hybrid_game = Game(players=[hybrid_blue, SimplePlayer(Color.RED)], seed=31)

    for _ in range(OPENING_STEP_COUNT):
        router_record = router_game.play_tick()
        hybrid_record = hybrid_game.play_tick()
        assert router_record.action.action_type == hybrid_record.action.action_type
        assert router_record.action.value == hybrid_record.action.value


def test_compact_chunk_round_trip(tmp_path):
    game = Game(
        players=[AlphaBetaPlayer(Color.BLUE), AlphaBetaPlayer(Color.RED)],
        seed=41,
    )
    accumulator = CompactPlacementAccumulator()
    winner = game.play(accumulators=[accumulator])

    assert winner is not None
    assert accumulator.record is not None

    chunk_path = tmp_path / "placement_chunk.npz"
    save_chunk_records(str(chunk_path), [accumulator.record])
    loaded_records = load_chunk_records(str(chunk_path))

    assert len(loaded_records) == 1
    loaded = loaded_records[0]
    np.testing.assert_array_equal(
        loaded["static_node_features"],
        accumulator.record["static_node_features"],
    )
    np.testing.assert_array_equal(
        loaded["opening_actions_onehot"],
        accumulator.record["opening_actions_onehot"],
    )
    assert loaded["winner_is_first_actor"] == accumulator.record["winner_is_first_actor"]
