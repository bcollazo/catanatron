from catanatron_experimental.machine_learning.players.minimax import (
    generate_desired_domestic_trades,
)


def test_sheep_for_wheat_generation():
    """Test that if you have 1-1-2-0-0 its interesting
    to trade that sheep for the missing wheat"""
    hand_freqdeck = [1, 1, 2, 0, 0]

    result = generate_desired_domestic_trades(hand_freqdeck)
    assert result == {(0, 0, 1, 0, 0, 0, 0, 0, 1, 0)}


def test_wood_for_brick_generation():
    """Test that if you have 2-0-0-0-0 its interesting
    to trade that wood for the missing brick"""
    hand_freqdeck = [2, 0, 0, 0, 0]

    result = generate_desired_domestic_trades(hand_freqdeck)
    assert result == {(1, 0, 0, 0, 0, 0, 1, 0, 0, 0)}


def test_can_give_multiple_generation():
    """Test that if missing a brick, but have extra wood,
    and sheep, will consider giving them away"""
    hand_freqdeck = [2, 0, 1, 0, 0]

    result = generate_desired_domestic_trades(hand_freqdeck)
    assert result == {
        (1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
    }


def test_only_if_completes_generation():
    """Test that if 2-0-1-1-0 will consider giving trades
    to find road, settlement or dev"""
    hand_freqdeck = [2, 0, 1, 1, 0]

    result = generate_desired_domestic_trades(hand_freqdeck)
    assert result == {
        (1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    }
