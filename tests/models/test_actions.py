from catanatron.state import (
    State,
    build_city,
    build_settlement,
    player_deck_add,
    player_deck_replenish,
)
from catanatron.models.actions import (
    generate_playable_actions,
    monopoly_possible_actions,
    year_of_plenty_possibilities,
    road_possible_actions,
    settlement_possible_actions,
    city_possible_actions,
    robber_possibilities,
    maritime_trade_possibilities,
    road_building_possibilities,
)
from catanatron.models.board import Board
from catanatron.models.enums import (
    BRICK,
    ORE,
    Resource,
    ActionType,
    ActionPrompt,
    WHEAT,
    WOOD,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.game import Game
from catanatron.models.decks import ResourceDeck


def test_playable_actions():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    actions = generate_playable_actions(state)
    assert len(actions) == 54
    assert actions[0].action_type == ActionType.BUILD_SETTLEMENT


def test_year_of_plenty_possible_actions_full_resource_bank():
    player = SimplePlayer(Color.RED)
    bank_resource_deck = ResourceDeck.starting_bank()
    actions = year_of_plenty_possibilities(player, bank_resource_deck)
    assert len(actions) == 15


def test_year_of_plenty_possible_actions_not_enough_cards():
    player = SimplePlayer(Color.RED)
    bank_resource_deck = ResourceDeck()
    bank_resource_deck.replenish(2, Resource.ORE)
    actions = year_of_plenty_possibilities(player, bank_resource_deck)
    assert len(actions) == 2  # one ORE, or 2 OREs.


def test_monopoly_possible_actions():
    player = SimplePlayer(Color.RED)
    assert len(monopoly_possible_actions(player)) == len(Resource)


def test_road_possible_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(road_possible_actions(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    assert len(road_possible_actions(state, Color.RED)) == 0  # no money

    player_deck_replenish(state, player.color, WOOD)
    player_deck_replenish(state, player.color, BRICK)
    assert len(road_possible_actions(state, Color.RED)) == 3

    state.board.build_settlement(Color.RED, 1, initial_build_phase=True)
    assert len(road_possible_actions(state, Color.RED)) == 6


def test_settlement_possible_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(settlement_possible_actions(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    state.board.build_road(Color.RED, (3, 4))
    state.board.build_road(Color.RED, (4, 5))
    assert len(settlement_possible_actions(state, Color.RED)) == 0  # no money

    player_deck_add(state, player.color, ResourceDeck.settlement_cost())
    assert len(settlement_possible_actions(state, Color.RED)) == 1

    state.board.build_road(Color.RED, (5, 0))
    assert len(settlement_possible_actions(state, Color.RED)) == 2


def test_city_playable_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(city_possible_actions(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    build_settlement(state, player.color, 3, True)
    assert len(city_possible_actions(state, Color.RED)) == 0  # no money

    player_deck_replenish(state, Color.RED, WHEAT, 2)
    player_deck_replenish(state, Color.RED, ORE, 3)
    assert len(city_possible_actions(state, Color.RED)) == 1

    state.board.build_settlement(Color.RED, 0, initial_build_phase=True)
    build_settlement(state, player.color, 0, True)
    assert len(city_possible_actions(state, Color.RED)) == 2


def test_robber_possibilities():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    orange = SimplePlayer(Color.ORANGE)
    players = [red, blue, orange]
    state = State(players)

    # one for each resource tile (excluding desert)
    assert len(robber_possibilities(state, Color.RED)) == 18

    # assert same number of possibilities, b.c. players have no cards.
    state.board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    state.board.build_settlement(Color.ORANGE, 0, initial_build_phase=True)
    assert len(robber_possibilities(state, Color.RED)) == 18

    # assert same number of possibilities, b.c. only one player to rob in this tile
    player_deck_replenish(state, orange.color, WHEAT)
    assert len(robber_possibilities(state, Color.RED)) == 18

    # now possibilites increase by 1 b.c. we have to decide to steal from blue or orange
    # Unless desert is (0,0,0)... in which case still at 18...
    player_deck_replenish(state, blue.color, WHEAT)
    possibilities = len(robber_possibilities(state, Color.RED))
    assert possibilities == 19 or (
        possibilities == 18 and state.board.map.tiles[(0, 0, 0)].resource is None
    )


def test_building_settlement_gives_vp():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    build_settlement(state, state.players[0].color, 0, True)
    assert state.player_state["P0_VICTORY_POINTS"] == 1
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 1


def test_building_city_gives_vp():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    build_settlement(state, state.players[0].color, 0, True)
    player_deck_replenish(state, state.players[0].color, WHEAT, 2)
    player_deck_replenish(state, state.players[0].color, ORE, 2)
    build_city(state, state.players[0].color, 0)
    assert state.player_state["P0_VICTORY_POINTS"] == 2
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2


def test_robber_possibilities_simple():
    red = SimplePlayer(Color.RED)
    blue = SimplePlayer(Color.BLUE)
    orange = SimplePlayer(Color.ORANGE)
    players = [red, blue, orange]
    state = State(players)

    # one for each resource tile (excluding desert)
    assert len(robber_possibilities(state, Color.RED)) == 18


def test_initial_placement_possibilities():
    red = SimplePlayer(Color.RED)
    state = State([red])
    assert len(settlement_possible_actions(state, Color.RED, True)) == 54


# TODO: Forcing random selection to ease dimensionality.
# def test_discard_possibilities():
#     player = SimplePlayer(Color.RED)
#     player_deck_replenish(state, player.color, Resource.WHEAT)
#     assert len(discard_possibilities(player)) == 70


def test_4to1_maritime_trade_possibilities():
    player = SimplePlayer(Color.RED)
    state = State([player])

    possibilities = maritime_trade_possibilities(state, player.color)
    assert len(possibilities) == 0

    player_deck_replenish(state, player.color, WHEAT, 4)
    possibilities = maritime_trade_possibilities(state, player.color)
    print(possibilities)
    assert len(possibilities) == 4

    player_deck_replenish(state, player.color, BRICK, 4)
    possibilities = maritime_trade_possibilities(state, player.color)
    assert len(possibilities) == 8


def test_maritime_possibities_respect_bank_not_having_cards():
    player = SimplePlayer(Color.RED)
    state = State([player])
    player_deck_replenish(state, player.color, WHEAT)
    assert len(maritime_trade_possibilities(state, player.color)) == 0


def test_road_building_possibilities():
    board = Board()
    player = SimplePlayer(Color.RED)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    result = road_building_possibilities(player, board)

    # 6 length-2 paths, (3 * 2 combinations) / (2  b.c. of symmetry)
    assert len(result) == 6 + 6 / 2


def test_road_building_two_houses():
    board = Board()
    player = SimplePlayer(Color.RED)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_settlement(Color.RED, 0, initial_build_phase=True)

    result = road_building_possibilities(player, board)
    # 6 length-2 paths in first house
    # 6 length-2 paths in second house
    # 6 * 5 combinations of length-1 paths, divided by 2 b.c. of symmetry
    assert len(result) == 6 + 6 + (6 * 5) / 2


def test_year_of_plenty_same_resource():
    bank = ResourceDeck()
    bank.replenish(1, Resource.WHEAT)

    player = SimplePlayer(Color.RED)
    actions = year_of_plenty_possibilities(player, bank)

    assert len(actions) == 1
    assert actions[0].value[0] == Resource.WHEAT
