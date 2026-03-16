
from catanatron.game import Game
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.state_functions import player_key

def test_immediate_win_logic():
    # 1. Initialize a game
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    vps_to_win = 10
    game = Game(players, vps_to_win=vps_to_win)
    
    p0 = game.state.current_player()
    p0_key = player_key(game.state, p0.color)
    
    # 2. Initially p0 should have some actions (ROLL or initial build)
    actions = generate_playable_actions(game.state)
    assert len(actions) > 0
    
    # 3. Buff p0 to 10 VPs manually
    game.state.player_state[f"{p0_key}_ACTUAL_VICTORY_POINTS"] = 10
    
    # 4. Verify generate_playable_actions returns empty list IMMEDIATELY
    actions = generate_playable_actions(game.state)
    assert actions == [], f"Expected 0 actions, got {len(actions)}"
    
    print("Test passed: generate_playable_actions returns empty list once VPs reached!")

if __name__ == "__main__":
    test_immediate_win_logic()
