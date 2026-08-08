from catanatron.models.board import Board
from catanatron.state import (
    State,
)
from catanatron.state_functions import (
    buy_dev_card,
    get_largest_army,
    play_dev_card,
    player_deck_replenish,
)
from catanatron.models.player import SimplePlayer, Color
from catanatron.models.enums import KNIGHT, ORE, SHEEP, WHEAT


def test_longest_road_simple():
    board = Board()

    # Place initial settlements.
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    board.build_settlement(Color.BLUE, 24, initial_build_phase=True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_settlement(Color.BLUE, 26, initial_build_phase=True)
    board.build_road(Color.BLUE, (25, 26))
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))
    assert board.road_color is None
    assert board.road_lengths == {Color.RED: 2, Color.BLUE: 2}

    board.build_road(Color.RED, (2, 3))
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))
    assert board.road_color is Color.RED
    assert board.road_length == 5
    assert board.road_lengths == {Color.RED: 5, Color.BLUE: 2}


def test_longest_road_tie():
    board = Board()
    # Place initial settlements.
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    board.build_settlement(Color.BLUE, 24, initial_build_phase=True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_settlement(Color.BLUE, 26, initial_build_phase=True)
    board.build_road(Color.BLUE, (25, 26))
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))
    assert board.road_color is None
    assert board.road_lengths == {Color.RED: 2, Color.BLUE: 2}

    board.build_road(Color.RED, (2, 3))
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))

    board.build_road(Color.BLUE, (26, 27))
    board.build_road(Color.BLUE, (27, 28))
    board.build_road(Color.BLUE, (28, 29))
    assert (
        board.road_color is Color.RED
    )  # even if blue also has 5-road. red had it first
    assert board.road_length == 5
    assert board.road_lengths == {Color.RED: 5, Color.BLUE: 5}

    board.build_road(Color.BLUE, (29, 30))
    assert board.road_color is Color.BLUE
    assert board.road_length == 6
    assert board.road_lengths == {Color.RED: 5, Color.BLUE: 6}


# test: complicated circle around
def test_complicated_road():  # classic 8-like roads
    board = Board()

    # Place initial settlements.
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    board.build_settlement(Color.BLUE, 24, initial_build_phase=True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_settlement(Color.BLUE, 26, initial_build_phase=True)
    board.build_road(Color.BLUE, (25, 26))
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))

    board.build_road(Color.RED, (2, 3))
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))
    board.build_road(Color.RED, (0, 5))

    board.build_road(Color.RED, (1, 6))
    board.build_road(Color.RED, (6, 7))
    board.build_road(Color.RED, (7, 8))
    board.build_road(Color.RED, (8, 9))
    board.build_road(Color.RED, (2, 9))

    assert board.road_color is Color.RED
    assert board.road_length == 11
    assert board.road_lengths == {Color.RED: 11, Color.BLUE: 2}

    board.build_road(Color.RED, (8, 27))
    assert board.road_color is Color.RED
    assert board.road_length == 11
    assert board.road_lengths == {Color.RED: 11, Color.BLUE: 2}


def test_triple_longest_road_tie():
    board = Board()

    board.build_settlement(Color.RED, 3, True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))
    board.build_road(Color.RED, (0, 5))
    board.build_road(Color.RED, (5, 4))
    board.build_road(Color.RED, (3, 4))

    board.build_settlement(Color.BLUE, 24, True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_road(Color.BLUE, (25, 26))
    board.build_road(Color.BLUE, (26, 27))
    board.build_road(Color.BLUE, (27, 8))
    board.build_road(Color.BLUE, (8, 7))
    board.build_road(Color.BLUE, (7, 24))

    board.build_settlement(Color.WHITE, 17, True)
    board.build_road(Color.WHITE, (18, 17))
    board.build_road(Color.WHITE, (17, 39))
    board.build_road(Color.WHITE, (39, 41))
    board.build_road(Color.WHITE, (41, 42))
    board.build_road(Color.WHITE, (42, 40))
    board.build_road(Color.WHITE, (40, 18))

    assert board.road_color is Color.RED
    assert board.road_length == 6
    assert board.road_lengths == {Color.RED: 6, Color.BLUE: 6, Color.WHITE: 6}


def test_largest_army_calculation_when_no_one_has_three():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)
    state = State([red, blue, white])

    player_deck_replenish(state, Color.RED, WHEAT, 2)
    player_deck_replenish(state, Color.RED, SHEEP, 2)
    player_deck_replenish(state, Color.RED, ORE, 2)
    player_deck_replenish(state, Color.BLUE, WHEAT, 1)
    player_deck_replenish(state, Color.BLUE, SHEEP, 1)
    player_deck_replenish(state, Color.BLUE, ORE, 1)
    buy_dev_card(state, Color.RED, KNIGHT)
    buy_dev_card(state, Color.RED, KNIGHT)
    buy_dev_card(state, Color.BLUE, KNIGHT)

    play_dev_card(state, Color.RED, KNIGHT)

    color, count = get_largest_army(state)
    assert color is None and count is None


def test_largest_army_calculation_on_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)
    state = State([red, blue, white])

    player_deck_replenish(state, red.color, KNIGHT, 3)
    player_deck_replenish(state, blue.color, KNIGHT, 4)
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.BLUE, KNIGHT)
    play_dev_card(state, Color.BLUE, KNIGHT)
    play_dev_card(state, Color.BLUE, KNIGHT)

    color, count = get_largest_army(state)
    assert color is Color.RED and count == 3

    play_dev_card(state, Color.BLUE, KNIGHT)

    color, count = get_largest_army(state)
    assert color is Color.BLUE and count == 4


def test_cut_but_not_disconnected():
    board = Board()

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    board.build_road(Color.RED, (1, 2))
    board.build_road(Color.RED, (2, 3))
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (3, 12))
    assert (
        max(map(lambda path: len(path), board.continuous_roads_by_player(Color.RED)))
        == 7
    )
    assert len(board.find_connected_components(Color.RED)) == 1

    board.build_settlement(Color.BLUE, 2, initial_build_phase=True)
    assert len(board.find_connected_components(Color.RED)) == 1
    assert (
        max(map(lambda path: len(path), board.continuous_roads_by_player(Color.RED)))
        == 6
    )


def longest_route(board, color):
    """Recompute from scratch.

    `board.road_lengths` is a monotone cache: it is not refreshed when an
    endpoint gets blocked, so asserting on it would make these tests pass on
    unpatched code for the wrong reason.
    """
    return max(len(path) for path in board.continuous_roads_by_player(color))


def test_longest_road_counts_road_ending_at_enemy_settlement():
    """A road leading into an opponent's building still counts.

    The route may not continue past it, but the segment itself is part of it.
    """
    board = Board()
    board.build_settlement(Color.RED, 7, initial_build_phase=True)
    for edge in [(6, 7), (1, 6), (0, 1), (7, 8), (8, 9), (2, 9)]:
        board.build_road(Color.RED, edge)
    assert longest_route(board, Color.RED) == 6

    # BLUE blocks one end of RED's road at node 0.
    board.build_settlement(Color.BLUE, 16, initial_build_phase=True)
    board.build_road(Color.BLUE, (5, 16))
    board.build_road(Color.BLUE, (0, 5))
    board.build_settlement(Color.BLUE, 0)
    assert longest_route(board, Color.RED) == 6, "segment (0, 1) still counts"

    # WHITE blocks the other end at node 2.
    board.build_settlement(Color.WHITE, 12, initial_build_phase=True)
    board.build_road(Color.WHITE, (3, 12))
    board.build_road(Color.WHITE, (2, 3))
    board.build_settlement(Color.WHITE, 2)
    assert longest_route(board, Color.RED) == 6, "segment (2, 9) still counts"


def test_longest_road_returns_to_bank_when_cut_below_five():
    """Nobody holds the card when no player reaches five roads anymore."""
    board = Board()
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    for edge in [(0, 1), (1, 6), (6, 7), (7, 8), (8, 9), (2, 9)]:
        board.build_road(Color.RED, edge)
    assert board.road_color is Color.RED
    assert board.road_length == 6

    # BLUE cuts RED's road in half at node 7 (3 roads on each side).
    board.build_settlement(Color.BLUE, 25, initial_build_phase=True)
    board.build_road(Color.BLUE, (24, 25))
    board.build_road(Color.BLUE, (7, 24))
    board.build_settlement(Color.BLUE, 7)

    assert board.road_lengths[Color.RED] == 3
    assert board.road_color is None, "card goes back to the bank"
    assert board.road_length == 0


def test_longest_road_stays_in_bank_while_challengers_are_tied():
    """When the holder is cut and two challengers tie for the lead, the card
    stays in the bank until a single player leads again."""
    board = Board()
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    for edge in [(0, 1), (1, 6), (6, 7), (7, 8), (8, 9), (2, 9)]:
        board.build_road(Color.RED, edge)

    board.build_settlement(Color.BLUE, 4, initial_build_phase=True)
    for edge in [(4, 15), (15, 17), (17, 18), (18, 16), (16, 21)]:
        board.build_road(Color.BLUE, edge)

    board.build_settlement(Color.WHITE, 11, initial_build_phase=True)
    for edge in [(11, 32), (32, 33), (33, 34), (34, 13), (13, 12)]:
        board.build_road(Color.WHITE, edge)

    # RED leads with 6; BLUE and WHITE are tied behind with 5 each.
    assert board.road_color is Color.RED
    assert board.road_lengths == {Color.RED: 6, Color.BLUE: 5, Color.WHITE: 5}

    # ORANGE cuts RED down to 3, leaving BLUE and WHITE tied for the lead.
    board.build_settlement(Color.ORANGE, 25, initial_build_phase=True)
    board.build_road(Color.ORANGE, (25, 24))
    board.build_road(Color.ORANGE, (24, 7))
    board.build_settlement(Color.ORANGE, 7)

    assert board.road_lengths[Color.RED] == 3
    assert board.road_color is None, "no single leader, so nobody holds the card"

    # WHITE breaks the tie and takes the card.
    board.build_road(Color.WHITE, (12, 3))
    assert board.road_color is Color.WHITE
    assert board.road_length == 6


def test_loop_cut_during_play_does_not_duplicate_component():
    """Cutting one node of a loop disconnects nothing: still one component.

    Unlike test_cut_but_not_disconnected, the settlement here is built during
    play rather than in the initial phase, which is what exercises the split.
    """
    board = Board()
    cycle = [0, 5, 4, 3, 2, 1]
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    for i in range(6):
        board.build_road(Color.RED, (cycle[i], cycle[(i + 1) % 6]))
    assert len(board.connected_components[Color.RED]) == 1
    assert board.road_lengths[Color.RED] == 6

    board.build_settlement(Color.BLUE, 11, initial_build_phase=True)
    board.build_road(Color.BLUE, (11, 12))
    board.build_road(Color.BLUE, (3, 12))
    board.build_settlement(Color.BLUE, 3)

    assert len(board.connected_components[Color.RED]) == 1, "no duplicate component"
    assert board.road_lengths[Color.RED] == 5, "route may not pass through node 3"
