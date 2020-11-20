from catanatron.game import Game
from catanatron.algorithms import longest_road, largest_army
from catanatron.models.board_initializer import NodeRef, EdgeRef
from catanatron.models.actions import Action, ActionType
from catanatron.models.player import SimplePlayer, Color
from catanatron.models.decks import ResourceDeck
from catanatron.models.enums import DevelopmentCard


def road(player, node_id):
    """Helper function to create BUILD_ROAD actions"""
    return Action(player, ActionType.BUILD_ROAD, node_id)


def test_longest_road_simple():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red, ActionType.BUILD_FIRST_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)].id
        )
    )
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color is None

    game.execute(road(red, edges[((0, 0, 0), EdgeRef.EAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.WEST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 5


def test_longest_road_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((0, 0, 0), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(
        Action(
            blue,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((0, 2, -2), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.SOUTHEAST)].id))

    game.execute(road(red, edges[((0, 0, 0), EdgeRef.EAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.WEST)].id))

    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.EAST)].id))
    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.NORTHEAST)].id))
    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.NORTHWEST)].id))
    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.WEST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED  # even if blue also has 5-road. red had it first
    assert len(path) == 5

    game.execute(road(blue, edges[((0, 2, -2), EdgeRef.SOUTHWEST)].id))
    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.BLUE
    assert len(path) == 6


# test: complicated circle around
def test_complicated_road():  # classic 8-like roads
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red, ActionType.BUILD_FIRST_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)].id
        ),
    )
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.EAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.WEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHWEST)].id))

    game.execute(road(red, edges[((1, -1, 0), EdgeRef.SOUTHWEST)].id))
    game.execute(road(red, edges[((1, -1, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(road(red, edges[((1, -1, 0), EdgeRef.EAST)].id))
    game.execute(road(red, edges[((1, -1, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(red, edges[((1, -1, 0), EdgeRef.NORTHWEST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11

    game.execute(road(red, edges[((2, -2, 0), EdgeRef.SOUTHWEST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 11


def test_triple_longest_road_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)
    red.resource_deck += ResourceDeck.starting_bank()
    blue.resource_deck += ResourceDeck.starting_bank()
    white.resource_deck += ResourceDeck.starting_bank()

    game = Game(players=[red, blue, white])
    nodes = game.board.nodes
    edges = game.board.edges

    game.execute(
        Action(
            red, ActionType.BUILD_FIRST_SETTLEMENT, nodes[((0, 0, 0), NodeRef.SOUTH)].id
        ),
    )
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.EAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.WEST)].id))
    game.execute(road(red, edges[((0, 0, 0), EdgeRef.SOUTHWEST)].id))

    game.execute(
        Action(
            blue,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((-2, 2, 0), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.EAST)].id))
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.WEST)].id))
    game.execute(road(blue, edges[((-2, 2, 0), EdgeRef.SOUTHWEST)].id))

    game.execute(
        Action(
            white,
            ActionType.BUILD_FIRST_SETTLEMENT,
            nodes[((2, -2, 0), NodeRef.SOUTH)].id,
        ),
    )
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.SOUTHEAST)].id))
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.EAST)].id))
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.NORTHEAST)].id))
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.NORTHWEST)].id))
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.WEST)].id))
    game.execute(road(white, edges[((2, -2, 0), EdgeRef.SOUTHWEST)].id))

    color, path = longest_road(game.board, game.players, game.actions)
    assert color == Color.RED
    assert len(path) == 6


def test_largest_army_calculation_when_no_one_has_three():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)

    red.development_deck.replenish(2, DevelopmentCard.KNIGHT)
    blue.development_deck.replenish(1, DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions = [
        Action(red, ActionType.PLAY_KNIGHT_CARD, None),
        Action(red, ActionType.PLAY_KNIGHT_CARD, None),
        Action(blue, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    color, count = largest_army([red, blue, white], actions)
    assert color is None and count is None


def test_largest_army_calculation_on_tie():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    white = SimplePlayer(Color.WHITE)

    red.development_deck.replenish(3, DevelopmentCard.KNIGHT)
    blue.development_deck.replenish(4, DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    red.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions = [
        Action(red, ActionType.PLAY_KNIGHT_CARD, None),
        Action(red, ActionType.PLAY_KNIGHT_CARD, None),
        Action(red, ActionType.PLAY_KNIGHT_CARD, None),
        Action(blue, ActionType.PLAY_KNIGHT_CARD, None),
        Action(blue, ActionType.PLAY_KNIGHT_CARD, None),
        Action(blue, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    color, count = largest_army([red, blue, white], actions)
    assert color is Color.RED and count == 3

    blue.mark_played_dev_card(DevelopmentCard.KNIGHT)
    actions.append(Action(blue, ActionType.PLAY_KNIGHT_CARD, None))

    color, count = largest_army([red, blue, white], actions)
    assert color is Color.BLUE and count == 4
