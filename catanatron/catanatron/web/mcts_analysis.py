from catanatron.players.mcts import StateNode


class GameAnalyzer:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def analyze_win_probabilities(self, game):
        """Uses MCTS to analyze win probabilities from current game state"""
        if game.winning_color() is not None:
            winner = game.winning_color()
            result = {
                winner.value: 100.0,
                **{c.value: 0.0 for c in game.state.colors if c != winner},
            }
            return result

        # Create root node and run simulations
        root = StateNode(game.state.current_color(), game.copy(), None, prunning=True)
        for _ in range(self.num_simulations):
            root.run_simulation()

        # Calculate probabilities using MCTS statistics
        probabilities = {}
        for color in game.state.colors:
            if color == root.color:
                win_ratio = root.wins / root.visits if root.visits > 0 else 0
            else:
                # Assume remaining wins distributed evenly among other players
                # StateNode does not track wins for other colors
                remaining_wins = root.visits - root.wins
                num_other_players = len(game.state.colors) - 1
                win_ratio = (
                    (remaining_wins / num_other_players) / root.visits
                    if root.visits > 0
                    else 0
                )

            probabilities[color.value] = round(win_ratio * 100, 1)

        return probabilities
