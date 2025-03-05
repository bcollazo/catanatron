#!/usr/bin/env python
"""
Simple script to run a Catan game and display the winner.
"""
import time


def main():
    try:
        from catanatron_rust import Game

        # Create and play the game
        game = Game(4)
        print(f"Created game with {game.get_num_players()} players")
        print("Game configuration:")
        print("- Victory points to win: 10")
        print("- Maximum ticks: 10000")

        print("\nStarting game simulation...")
        start_time = time.time()
        game.play()
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get and display the winner
        winner = game.get_winner()

        print("\n" + "=" * 50)
        print(f"Game completed in {elapsed_time:.2f} seconds")

        if winner is not None:
            print(f"WINNER: Player {winner} reached 10 victory points!")
        else:
            print("NO WINNER: Game reached 10000 ticks without a winner")
            print("This happens when no player reaches 10 victory points")
            print("within the maximum number of turns.")
        print("=" * 50)

    except ImportError as e:
        print(f"Failed to import catanatron_rust: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
