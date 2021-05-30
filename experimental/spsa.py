# Implements https://www.chessprogramming.org/SPSA
import random
import numpy as np

from catanatron.models.player import Color
from experimental.machine_learning.players.minimax import ValueFunctionPlayer
from experimental.play import play_batch

# for (k=0; k < N; k++) {
#   ak = a / (k + 1 + A)^alpha;
#   ck = c / (k + 1)^γ;
#   for each p
#     Δp = 2 * round ( rand() / (RAND_MAX + 1.0) ) - 1.0;
#   Θ+ = Θ + ck*Δ;
#   Θ- = Θ - ck*Δ;
#   Θ +=  ak * match(Θ+, Θ-) / (ck*Δ);
# }

N = 1000
a = 1
c = 1
A = 100

p = 8
r = 10.0

alpha = 0.602
gamma = 0.101


def main():
    theta = np.array([1.0 for i in range(p)])
    for k in range(N):
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)

        delta = np.array([random.choice((-r, r)) for _ in range(p)])

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta
        theta += ak * match(theta_plus, theta_minus) / (ck * delta)

    print(theta)


def match(theta_plus, theta_minus):
    print(theta_plus, "vs", theta_minus)
    games_played = 10
    players = [
        ValueFunctionPlayer(Color.RED, "ThetaPlus", "C", params=theta_plus),
        ValueFunctionPlayer(Color.BLUE, "ThetaMinus", "C", params=theta_minus),
    ]
    wins, _ = play_batch(games_played, players, None, False, False, verbose=True)
    return (wins[str(players[0])] / games_played - 0.5) * 4


if __name__ == "__main__":
    main()
