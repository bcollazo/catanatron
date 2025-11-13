"""
Test script to run the trained Q-learning bot against random opponents
"""
from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from my_q_bot import MyQBot

def play_game():
    """Play a single game with the trained bot vs one random opponent."""
    # Create players: 1 trained bot + 1 random opponent (2-player game)
    # Colors are: RED, BLUE
    players = [
        MyQBot("RED"),  # Our trained bot
        RandomPlayer("BLUE"),  # One random opponent
    ]

    # Create and play game
    game = Game(players)
    game.play()

    return game


def main():
    """Run multiple games and show stats."""
    num_games = 20
    wins = {player_name: 0 for player_name in ["MyQBot", "RandomPlayer"]}

    print(f"\nPlaying {num_games} games...")
    print("MyQBot (RED) vs 3 RandomPlayers\n")

    for i in range(num_games):
        game = play_game()
        winner = game.winning_player()

        # Track wins
        if isinstance(winner, MyQBot):
            wins["MyQBot"] += 1
            winner_name = "MyQBot"
        else:
            wins["RandomPlayer"] += 1
            winner_name = f"RandomPlayer ({winner.color})"

        # Print game result
        vps = {p.color: game.state.player_state[f"P{i}_ACTUAL_VPS"]
               for i, p in enumerate(game.players)}
        print(f"Game {i+1:2d}: Winner = {winner_name:20s} | VPs: {vps}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"MyQBot wins:      {wins['MyQBot']:2d} / {num_games} ({wins['MyQBot']/num_games*100:.1f}%)")
    print(f"RandomPlayer wins: {wins['RandomPlayer']:2d} / {num_games} ({wins['RandomPlayer']/num_games*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
