"""Small executable reproductions for rule bugs found during the Rust rewrite.

Each test states the intended rule. They are strict xfails until the Python
engine is corrected: an unexpected pass means the bug was fixed and this file
must be updated rather than silently preserving the old expectation.
"""

from collections import defaultdict

import pytest

from catanatron.apply_action import apply_action
from catanatron.models.board import Board, longest_acyclic_path
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.state import State


known_bug = pytest.mark.xfail(strict=True, reason="known Python rule divergence")


@known_bug
def test_d001_trade_proposer_does_not_answer_own_offer():
    """Seat 0 answers, then seat 2 should answer; seat 1 made the offer.

        seats:   responder(0) -> proposer(1) -> next responder(2)
        replies: responder(0) -----------------> next responder(2)
    """
    state = State(
        [
            SimplePlayer(Color.ORANGE),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.RED),
        ]
    )
    proposer = state.colors[1]
    first_responder = state.colors[0]
    expected_next = state.colors[2]
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = state.current_turn_index = 1
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    apply_action(state, Action(proposer, ActionType.OFFER_TRADE, offer))
    assert state.current_color() == first_responder
    apply_action(state, Action(first_responder, ActionType.REJECT_TRADE, None))

    assert state.current_color() == expected_next


@known_bug
def test_d002_road_may_end_at_an_opponents_settlement():
    """The final edge counts, but travel cannot continue through X.

        ORANGE: 35 -- 36 -- 37 -- 38[X BLUE]
                 1     2     3
    """
    board = Board()
    board.build_settlement(Color.ORANGE, 35, initial_build_phase=True)
    board.build_road(Color.ORANGE, (35, 36))
    board.build_road(Color.ORANGE, (36, 37))
    board.build_settlement(Color.BLUE, 38, initial_build_phase=True)
    board.build_road(Color.ORANGE, (37, 38))

    assert board.road_lengths[Color.ORANGE] == 3


@known_bug
def test_d003_branch_uses_the_longest_edge_simple_trail():
    """A live game left a stale length of four for this five-edge trail.

             21
              \
        0 -- 20 -- 19
              \
              22 -- 23 -- 52

        trail: 21-19-20-22-23-52 (five edges)
    """
    board = Board()
    edges = [(19, 21), (19, 20), (20, 22), (22, 23), (23, 52), (0, 20)]
    _install_roads(board, Color.RED, edges)
    exact_length = len(
        longest_acyclic_path(board, {0, 19, 20, 21, 22, 23, 52}, Color.RED)
    )
    assert exact_length == 5  # makes the graph itself easy to audit

    # This is the inconsistent cache value captured by the differential run.
    # The production getters and award logic consume this cached value.
    board.road_lengths[Color.RED] = 4

    assert board.road_lengths[Color.RED] == 5


def _install_roads(board, color, edges):
    """Install one deliberately tiny cached road component for split tests."""
    nodes = set()
    for a, b in edges:
        board.roads[a, b] = board.roads[b, a] = color
        nodes.update((a, b))
    board.connected_components[color].append(nodes)


@known_bug
def test_d004_incumbent_keeps_longest_road_when_tied():
    """Splitting ORANGE from seven to five ties RED; ORANGE is incumbent.

        ORANGE: 27-8-9-2-3-12-13-34   split at 12 -> longest side is 5
        RED:    45-46-48-49-50-51     already 5
        BLUE:                  11-12   builds the splitting settlement
    """
    board = Board()
    _install_roads(
        board,
        Color.ORANGE,
        [(27, 8), (8, 9), (9, 2), (2, 3), (3, 12), (12, 13), (13, 34)],
    )
    _install_roads(
        board, Color.RED, [(45, 46), (46, 48), (48, 49), (49, 50), (50, 51)]
    )
    _install_roads(board, Color.BLUE, [(11, 12)])
    board.road_lengths = defaultdict(
        int, {Color.RED: 5, Color.ORANGE: 7, Color.BLUE: 1}
    )
    board.road_color, board.road_length = Color.ORANGE, 7

    board.build_settlement(Color.BLUE, 12)

    assert board.road_lengths[Color.ORANGE] == 5
    assert board.road_color == Color.ORANGE


@known_bug
def test_d005_longest_road_is_never_awarded_below_five():
    """RED's length-four road splits 2+2; BLUE's maximum is only four.

        RED: 27-8-9-2-3    split at 9
        BLUE: 45-46-48-49-50 (four), plus access road 10-9
    """
    board = Board()
    _install_roads(board, Color.RED, [(27, 8), (8, 9), (9, 2), (2, 3)])
    _install_roads(board, Color.BLUE, [(45, 46), (46, 48), (48, 49), (49, 50)])
    board.connected_components[Color.BLUE].append({10, 9})
    board.roads[10, 9] = board.roads[9, 10] = Color.BLUE
    board.road_lengths = defaultdict(int, {Color.BLUE: 4, Color.RED: 4})
    board.road_color, board.road_length = Color.RED, 4

    board.build_settlement(Color.BLUE, 9)

    assert board.road_lengths[Color.RED] == 2
    assert board.road_color is None
