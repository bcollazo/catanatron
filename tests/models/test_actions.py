from catanatron.state import State
from catanatron.models.actions import (
    generate_playable_actions,
    monopoly_possibilities,
    year_of_plenty_possibilities,
    road_building_possibilities,
    settlement_possibilities,
    city_possibilities,
    robber_possibilities,
    maritime_trade_possibilities,
)
from catanatron.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    ActionType,
    WHEAT,
    WOOD,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import (
    SETTLEMENT_COST_FREQDECK,
    starting_resource_bank,
)
from catanatron.state_functions import (
    build_city,
    build_settlement,
    player_deck_replenish,
    player_freqdeck_add,
)


def test_playable_actions():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    actions = generate_playable_actions(state)
    assert len(actions) == 54
    assert actions[0].action_type == ActionType.BUILD_SETTLEMENT


def test_year_of_plenty_possible_actions_full_resource_bank():
    bank_resource_freqdeck = starting_resource_bank()
    actions = year_of_plenty_possibilities(Color.RED, bank_resource_freqdeck)
    assert len(actions) == 15


def test_year_of_plenty_possible_actions_not_enough_cards():
    bank_resource_freqdeck = [0, 0, 0, 0, 2]
    actions = year_of_plenty_possibilities(Color.RED, bank_resource_freqdeck)
    assert len(actions) == 2  # one ORE, or 2 OREs.


def test_monopoly_possible_actions():
    assert len(monopoly_possibilities(Color.RED)) == len(RESOURCES)


def test_road_possible_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(road_building_possibilities(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    assert len(road_building_possibilities(state, Color.RED)) == 0  # no money

    player_deck_replenish(state, player.color, WOOD)
    player_deck_replenish(state, player.color, BRICK)
    assert len(road_building_possibilities(state, Color.RED)) == 3

    state.board.build_settlement(Color.RED, 1, initial_build_phase=True)
    assert len(road_building_possibilities(state, Color.RED)) == 6


def test_settlement_possible_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(settlement_possibilities(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    state.board.build_road(Color.RED, (3, 4))
    state.board.build_road(Color.RED, (4, 5))
    assert len(settlement_possibilities(state, Color.RED)) == 0  # no money

    player_freqdeck_add(state, player.color, SETTLEMENT_COST_FREQDECK)
    assert len(settlement_possibilities(state, Color.RED)) == 1

    state.board.build_road(Color.RED, (5, 0))
    assert len(settlement_possibilities(state, Color.RED)) == 2


def test_city_playable_actions():
    player = SimplePlayer(Color.RED)
    state = State([player])

    assert len(city_possibilities(state, Color.RED)) == 0  # no money or place

    state.board.build_settlement(Color.RED, 3, initial_build_phase=True)
    build_settlement(state, player.color, 3, True)
    assert len(city_possibilities(state, Color.RED)) == 0  # no money

    player_deck_replenish(state, Color.RED, WHEAT, 2)
    player_deck_replenish(state, Color.RED, ORE, 3)
    assert len(city_possibilities(state, Color.RED)) == 1

    state.board.build_settlement(Color.RED, 0, initial_build_phase=True)
    build_settlement(state, player.color, 0, True)
    assert len(city_possibilities(state, Color.RED)) == 2


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
        possibilities == 18 and state.board.map.land_tiles[(0, 0, 0)].resource is None
    )


def test_building_settlement_gives_vp():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    build_settlement(state, state.colors[0], 0, True)
    assert state.player_state["P0_VICTORY_POINTS"] == 1
    assert state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 1


def test_building_city_gives_vp():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)

    build_settlement(state, state.colors[0], 0, True)
    player_deck_replenish(state, state.colors[0], WHEAT, 2)
    player_deck_replenish(state, state.colors[0], ORE, 2)
    build_city(state, state.colors[0], 0)
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
    assert len(settlement_possibilities(state, Color.RED, True)) == 54


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


def test_year_of_plenty_same_resource():
    bank = [0, 0, 0, 1, 0]

    actions = year_of_plenty_possibilities(Color.RED, bank)

    assert len(actions) == 1
    assert actions[0].value[0] == WHEAT


def test_can_trade_with_port():
    players = [SimplePlayer(Color.RED)]

    state = State(players)
    state.board.build_settlement(Color.RED, 26, initial_build_phase=True)

    port_tile = state.board.map.tiles[(3, -3, 0)]  # port with node_id=25,26
    resource_out = port_tile.resource or WHEAT  # type: ignore
    num_out = 3 if port_tile.resource is None else 2  # type: ignore
    player_deck_replenish(state, Color.RED, resource_out, num_out)

    possibilities = maritime_trade_possibilities(state, Color.RED)
    assert len(possibilities) == 4
