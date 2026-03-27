import pytest
from catanatron.state import State
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import (
    BRICK,
    ORE,
    SHEEP,
    WHEAT,
    WOOD,
    ActionType,
)
from catanatron.models.actions import maritime_trade_possibilities
from catanatron.state_functions import player_deck_replenish

def test_maritime_trade_no_ports_only_4_to_1():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    
    # Player has 4 woods
    player_deck_replenish(state, Color.RED, WOOD, 4)
    
    actions = maritime_trade_possibilities(state, Color.RED)
    
    # Extract trade values
    trades = [a.value for a in actions]
    
    # Should only have 4:1 trades (4 WOODs, 1 j_resource)
    # Trade offer is (r1, r2, r3, r4, r_asked)
    for trade in trades:
        # Check that it's 4:1
        assert trade[:4].count(WOOD) == 4
        assert trade[4] != WOOD

def test_maritime_trade_general_port_allows_3_to_1_and_4_to_1():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    
    # Give player a general port (e.g., at node 0 if it's a port node, 
    # but easier to just mock get_player_port_resources or state.board)
    # Let's use the actual board for realism. 
    # Node 0, 1 are ports in BASE_MAP_TEMPLATE (usually)
    # Actually, let's look at the map nodes or just trust the logic.
    
    # Instead of building, let's just force the port resources for the test
    # by mocking get_player_port_resources because we want to test actions logic.
    player_deck_replenish(state, Color.RED, WOOD, 4)
    
    # Mocking port resources
    state.board.get_player_port_resources = lambda color: {None}
    
    actions = maritime_trade_possibilities(state, Color.RED)
    trades = [a.value for a in actions]
    
    # Should have both 3:1 and 4:1 trades
    rates = set()
    for trade in trades:
        rate = trade[:4].count(WOOD)
        rates.add(rate)
    
    assert 3 in rates
    assert 4 in rates

def test_maritime_trade_specific_port_allows_2_to_1_3_to_1_and_4_to_1():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    
    player_deck_replenish(state, Color.RED, WOOD, 4)
    
    # Mocking port resources: 2:1 WOOD and General port
    state.board.get_player_port_resources = lambda color: {WOOD, None}
    
    actions = maritime_trade_possibilities(state, Color.RED)
    trades = [a.value for a in actions]
    
    rates = set()
    for trade in trades:
        if trade[0] == WOOD: # ensure we are looking at WOOD trades
           rate = sum(1 for r in trade[:4] if r == WOOD)
           rates.add(rate)
    
    assert 2 in rates
    assert 3 in rates
    assert 4 in rates
    
def test_maritime_trade_specific_port_for_other_resource_does_not_affect_wood():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    
    player_deck_replenish(state, Color.RED, WOOD, 4)
    
    # Mocking port resources: 2:1 BRICK
    state.board.get_player_port_resources = lambda color: {BRICK}
    
    actions = maritime_trade_possibilities(state, Color.RED)
    trades = [a.value for a in actions]
    
    rates = set()
    for trade in trades:
        if trade[0] == WOOD:
           rate = sum(1 for r in trade[:4] if r == WOOD)
           rates.add(rate)
    
    assert 2 not in rates
    assert 3 not in rates
    assert 4 in rates

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
