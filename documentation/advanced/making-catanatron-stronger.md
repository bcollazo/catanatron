---
icon: dumbbell
---

# Making Catanatron Stronger

### AI Leaderboard

Catanatron will always refer to the best bot in this leaderboard.

The best bot right now is `AlphaBetaPlayer` with n = 2. Here a list of bots strength. Experiments done by running 1000 (when possible) 1v1 games against previous in list.

| Player               | % of wins in 1v1 games                      | num games used for result |
| -------------------- | ------------------------------------------- | ------------------------- |
| AlphaBeta(n=2)       | 80% vs ValueFunction                        | 25                        |
| ValueFunction        | 90% vs GreedyPlayouts(n=25)                 | 25                        |
| GreedyPlayouts(n=25) | 100% vs MCTS(n=100)                         | 25                        |
| MCTS(n=100)          | 60% vs WeightedRandom                       | 15                        |
| WeightedRandom       | <p>60% vs Random<br>50% vs VictoryPoint</p> | 1000                      |
| VictoryPoint         | 60% vs Random                               | 1000                      |
| Random               | -                                           | -                         |

### Making Catanatron Bot Stronger

The best bot right now is Alpha Beta Search with a hand-crafted value function. One of the most promising ways of improving Catanatron is to have your custom player inhert from ([`AlphaBetaPlayer`](../../catanatron/catanatron/players/minimax.py)) and set a better set of weights for the value function. You can also edit the value function and come up with your own innovative features!

For more sophisticated approaches, see example player implementations in [catanatron/catanatron/players](../../catanatron/catanatron/players)

If you find a bot that consistently beats the best bot right now, please submit a Pull Request! :)
