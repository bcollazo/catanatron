import pytest
import random
from catanatron.game import Game
from catanatron.models.player import SimplePlayer
from catanatron.models.player import Color
from catanatron.models.map import build_map
from catanatron.models.enums import Action, ActionType

def test_dice_restriction_mini_map():
    # MINI map has numbers [3, 4, 5, 6, 8, 9, 10]
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    catan_map = build_map("MINI")
    
    # Test case 1: Restrict to board, allow 7s
    game = Game(players, catan_map=catan_map, restrict_dice_to_board=True, allow_sevens=True)
    allowed = {3, 4, 5, 6, 7, 8, 9, 10}
    
    for _ in range(100):
        game.state.player_state["P0_HAS_ROLLED"] = False
        action = Action(Color.RED, ActionType.ROLL, None)
        game.execute(action)
        roll_sum = sum(game.state.last_roll)
        assert roll_sum in allowed

def test_dice_restriction_no_sevens():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    catan_map = build_map("BASE")
    
    # Test case 2: Allow all board numbers, but NO 7s
    game = Game(players, catan_map=catan_map, restrict_dice_to_board=False, allow_sevens=False)
    
    for _ in range(100):
        game.state.player_state["P0_HAS_ROLLED"] = False
        action = Action(Color.RED, ActionType.ROLL, None)
        game.execute(action)
        roll_sum = sum(game.state.last_roll)
        assert roll_sum != 7
        assert 2 <= roll_sum <= 12

def test_dice_restriction_combined():
    # MINI map: [3, 4, 5, 6, 8, 9, 10]
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    catan_map = build_map("MINI")
    
    # Test case 3: Restrict to board AND no 7s
    game = Game(players, catan_map=catan_map, restrict_dice_to_board=True, allow_sevens=False)
    allowed = {3, 4, 5, 6, 8, 9, 10}
    
    for _ in range(100):
        game.state.player_state["P0_HAS_ROLLED"] = False
        action = Action(Color.RED, ActionType.ROLL, None)
        game.execute(action)
        roll_sum = sum(game.state.last_roll)
        assert roll_sum in allowed
        assert roll_sum != 7

def test_bandit_map_restriction():
    # BANDIT map: [2, 3, 4, 5, 6, 8]
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    catan_map = build_map("BANDIT")
    
    game = Game(players, catan_map=catan_map, restrict_dice_to_board=True, allow_sevens=False)
    allowed = {2, 3, 4, 5, 6, 8}
    
    for _ in range(100):
        game.state.player_state["P0_HAS_ROLLED"] = False
        action = Action(Color.RED, ActionType.ROLL, None)
        game.execute(action)
        roll_sum = sum(game.state.last_roll)
        assert roll_sum in allowed
