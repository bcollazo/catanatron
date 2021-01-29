from catanatron.models.actions import (
    monopoly_possible_actions,
    year_of_plenty_possible_actions,
    road_possible_actions,
    settlement_possible_actions,
    city_possible_actions,
    robber_possibilities,
    initial_settlement_possibilites,
    discard_possibilities,
    maritime_trade_possibilities,
    road_building_possibilities,
    ActionType,
    ActionPrompt,
)
from catanatron.models.board import Board
from catanatron.models.enums import Resource
from catanatron.models.player import Color, SimplePlayer
from catanatron.game import Game
from catanatron.models.decks import ResourceDeck


def test_playable_actions():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    actions = game.playable_actions(players[0], ActionPrompt.ROLL)
    assert len(actions) == 1
    assert actions[0].action_type == ActionType.ROLL


def test_year_of_plenty_possible_actions_full_resource_bank():
    player = SimplePlayer(Color.RED)
    bank_resource_deck = ResourceDeck.starting_bank()
    actions = year_of_plenty_possible_actions(player, bank_resource_deck)
    assert len(actions) == 15


def test_year_of_plenty_possible_actions_not_enough_cards():
    player = SimplePlayer(Color.RED)
    bank_resource_deck = ResourceDeck()
    bank_resource_deck.replenish(2, Resource.ORE)
    actions = year_of_plenty_possible_actions(player, bank_resource_deck)
    assert len(actions) == 2  # one ORE, or 2 OREs.


def test_monopoly_possible_actions():
    player = SimplePlayer(Color.RED)
    assert len(monopoly_possible_actions(player)) == len(Resource)


def test_road_possible_actions():
    board = Board()
    player = SimplePlayer(Color.RED)

    assert len(road_possible_actions(player, board)) == 0  # no money or place

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    assert len(road_possible_actions(player, board)) == 0  # no money

    player.resource_deck.replenish(1, Resource.WOOD)
    player.resource_deck.replenish(1, Resource.BRICK)
    assert len(road_possible_actions(player, board)) == 3

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    assert len(road_possible_actions(player, board)) == 6


def test_settlement_possible_actions():
    board = Board()
    player = SimplePlayer(Color.RED)

    assert len(settlement_possible_actions(player, board)) == 0  # no money or place

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 5))
    assert len(settlement_possible_actions(player, board)) == 0  # no money

    player.resource_deck += ResourceDeck.settlement_cost()
    assert len(settlement_possible_actions(player, board)) == 1

    board.build_road(Color.RED, (5, 0))
    assert len(settlement_possible_actions(player, board)) == 2


def test_city_playable_actions():
    board = Board()
    player = SimplePlayer(Color.RED)

    assert len(city_possible_actions(player)) == 0  # no money or place

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    player.build_settlement(3, True)
    assert len(city_possible_actions(player)) == 0  # no money

    player.resource_deck.replenish(2, Resource.WHEAT)
    player.resource_deck.replenish(3, Resource.ORE)
    assert len(city_possible_actions(player)) == 1

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    player.build_settlement(3, True)
    assert len(city_possible_actions(player)) == 2


def test_robber_possibilities():
    board = Board()
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    orange = SimplePlayer(Color.ORANGE)
    players = [red, blue, orange]

    # one for each resource tile (excluding desert)
    assert len(robber_possibilities(red, board, players, True)) == 18

    # assert same number of possibilities, b.c. players have no cards.
    board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    board.build_settlement(Color.ORANGE, 0, initial_build_phase=True)
    assert len(robber_possibilities(red, board, players, True)) == 18

    # assert same number of possibilities, b.c. only one player to rob in this tile
    orange.resource_deck.replenish(1, Resource.WHEAT)
    assert len(robber_possibilities(red, board, players, False)) == 18

    # now possibilites increase by 1 b.c. we have to decide to steal from blue or orange
    # Unless desert is (0,0,0)... in which case still at 18...
    blue.resource_deck.replenish(1, Resource.WHEAT)
    possibilities = len(robber_possibilities(red, board, players, False))
    assert possibilities == 19 or (
        possibilities == 18 and board.tiles[(0, 0, 0)].resource is None
    )


def test_initial_placement_possibilities():
    board = Board()
    red = SimplePlayer(Color.RED)
    assert len(initial_settlement_possibilites(red, board, True)) == 54


# TODO: Forcing random selection to ease dimensionality.
# def test_discard_possibilities():
#     player = SimplePlayer(Color.RED)
#     player.resource_deck.replenish(8, Resource.WHEAT)
#     assert len(discard_possibilities(player)) == 70


def test_4to1_maritime_trade_possibilities():
    board = Board()
    player = SimplePlayer(Color.RED)

    bank = ResourceDeck.starting_bank()
    assert len(maritime_trade_possibilities(player, bank, board)) == 0

    player.resource_deck.replenish(4, Resource.WHEAT)
    assert len(maritime_trade_possibilities(player, bank, board)) == 4

    player.resource_deck.replenish(4, Resource.BRICK)
    assert len(maritime_trade_possibilities(player, bank, board)) == 8


def test_maritime_possibities_respect_bank_not_having_cards():
    board = Board()
    player = SimplePlayer(Color.RED)
    player.resource_deck.replenish(4, Resource.WHEAT)
    bank = ResourceDeck()
    assert len(maritime_trade_possibilities(player, bank, board)) == 0


def test_road_building_possibilities():
    board = Board()
    player = SimplePlayer(Color.RED)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    result = road_building_possibilities(player, board)

    # 6 length-2 paths, 3 * 2 combinations
    assert len(result) == 6 + 6


def test_road_building_two_houses():
    board = Board()
    player = SimplePlayer(Color.RED)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_settlement(Color.RED, 0, initial_build_phase=True)

    result = road_building_possibilities(player, board)
    # 6 length-2 paths in first house,
    # 6 length-2 paths in second house,
    # 6 * 5 combinations of length-1 paths
    assert len(result) == 6 + 6 + 6 * 5


def test_year_of_plenty_same_resource():
    bank = ResourceDeck()
    bank.replenish(1, Resource.WHEAT)

    player = SimplePlayer(Color.RED)
    actions = year_of_plenty_possible_actions(player, bank)

    assert len(actions) == 1
    assert actions[0].value == [Resource.WHEAT]
