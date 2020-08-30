from catanatron.models import Color


def longest_road(board):
    """
    For each connected subgraph (made by single-colored roads) find
    the longest path. Take max of all these candidates.

    Returns (path, color) tuple where
    longest -- list of edges (all from a single color)
    color -- color of player whose longest path belongs.
    """
    raise NotImplemented


def largest_army(board):
    """
    Count the number of robbers activated by each player.

    Return (largest, color) tuple where
    largest -- number of knights activated
    color -- color of player who owns largest army
    """
    raise NotImplemented

