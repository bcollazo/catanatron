from typing import List
from catanatron import Game, RandomPlayer, Color
from catanatron.models.player import Player
from catanatron.players.mcts import StateNode


def test_root_node_initial_properties():
    """
    Tests the initial properties of a root StateNode.
    """
    # 1. Create a real Game object
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game_instance = Game(players)

    player_color = Color.BLUE

    # 2. Create a StateNode instance for the root of a search tree
    root_node = StateNode(
        color=player_color,
        game=game_instance,
        parent=None,  # A root node has no parent
        prunning=False,  # Default or explicit
    )

    # 3. Assert initial properties
    assert root_node.wins == 0, "Initial wins should be 0"
    assert root_node.visits == 0, "Initial visits should be 0"
    assert root_node.is_leaf(), "A new node should be a leaf"
    assert root_node.parent is None, "Root node's parent should be None"
    assert root_node.color == player_color, "Node color should be set correctly"
    assert root_node.game is game_instance, "Node should hold the correct game instance"
    assert root_node.level == 0, "Root node's level should be 0"
    assert not root_node.prunning, "Pruning should be False by default or as set"
    assert (
        root_node.children == []
    ), "Initial children should be an empty list"  # children is initialized as [] then turned into defaultdict in expand


def test_child_node_initial_properties():
    """
    Tests the initial properties of a child StateNode.
    """
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game_instance = game_instance = Game(players)
    parent_color = Color.RED
    child_color = (
        Color.RED
    )  # Typically the MCTS player\'s color remains the same for nodes it creates

    parent_node = StateNode(color=parent_color, game=game_instance, parent=None)
    parent_node.level = 5  # Manually set for testing child\'s level calculation

    # Create a new game state for the child, perhaps by copying or a new instance
    child_game_instance = Game(players)

    child_node = StateNode(
        color=child_color, game=child_game_instance, parent=parent_node, prunning=True
    )

    assert child_node.wins == 0, "Initial wins for child should be 0"
    assert child_node.visits == 0, "Initial visits for child should be 0"
    assert child_node.is_leaf(), "New child node should be a leaf"
    assert (
        child_node.parent is parent_node
    ), "Child node's parent should be set correctly"
    assert child_node.color == child_color, "Child node's color should be set correctly"
    assert (
        child_node.game is child_game_instance
    ), "Child node should hold its own game instance"
    assert (
        child_node.level == parent_node.level + 1
    ), "Child node's level should be parent's level + 1"
    assert child_node.prunning, "Child node's pruning status should be set correctly"
    assert (
        child_node.children == []
    ), "Initial children for child node should be an empty list"
