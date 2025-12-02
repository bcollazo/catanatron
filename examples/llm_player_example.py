"""
Example of using LLMPlayer with a local Ollama instance.

This example demonstrates how to use the LLMPlayer that makes decisions
by calling a local Ollama LLM instance via network.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Install required dependencies:
   pip install langchain langchain-ollama
3. Pull a model (e.g., llama3.2):
   ollama pull llama3.2
4. Make sure Ollama is running:
   ollama serve
"""

from catanatron import Game, RandomPlayer, Color
from catanatron.players.llm import LLMPlayer


def main():
    """Run a game with an LLM player against random opponents."""

    # Create players - one LLM player and three random players
    players = [
        LLMPlayer(Color.RED, model_name="llama3.2"),  # Uses local Ollama
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]

    # Create and play the game
    print("Starting game with LLM player as RED...")
    print("This will make network calls to Ollama for each decision.\n")

    game = Game(players)
    winner = game.play()

    print(f"\nGame finished! Winner: {winner}")

    # Print final scores
    print("\nFinal Scores:")
    for color in game.state.colors:
        player_idx = game.state.colors.index(color)
        vp = game.state.player_state.get(f"P{player_idx}_VICTORY_POINTS", 0)
        print(f"  {color.value}: {vp} VP")


def main_custom_config():
    """Example with custom Ollama configuration."""

    # Create LLM player with custom settings
    llm_player = LLMPlayer(
        color=Color.RED,
        model_name="mistral",  # Use a different model
        ollama_base_url="http://localhost:11434",  # Custom Ollama URL
    )

    players = [
        llm_player,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]

    game = Game(players)
    winner = game.play()

    print(f"Winner: {winner}")


if __name__ == "__main__":
    # Run the basic example
    main()

    # Uncomment to run the custom config example:
    # main_custom_config()
