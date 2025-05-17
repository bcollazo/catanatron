"""
Implements https://www.chessprogramming.org/SPSA

This seems to work!  - November 7, 2021
"""

import random
import numpy as np

from catanatron.models.player import Color
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
)
from catanatron.cli.play import play_batch

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

p = len(DEFAULT_WEIGHTS.copy())
r = 10.0

alpha = 0.602
gamma = 0.101


def main():
    theta = np.array([1.0 for i in range(p)])
    for k in range(N):
        print("Iteration", k)
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)

        delta = np.array([random.choice((-r, r)) for _ in range(p)])

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta
        theta += ak * match(theta_plus, theta_minus) / (ck * delta)

    print(theta)


def match(theta_plus, theta_minus):
    print(theta_plus, "vs", theta_minus)
    games_played = 200

    weights_plus = {
        k: v + theta_plus[i] for i, (k, v) in enumerate(DEFAULT_WEIGHTS.items())
    }
    weights_minus = {
        k: v + theta_minus[i] for i, (k, v) in enumerate(DEFAULT_WEIGHTS.items())
    }
    players = [
        ValueFunctionPlayer(Color.RED, "C", params=weights_plus),
        ValueFunctionPlayer(Color.BLUE, "C", params=weights_minus),
    ]
    wins, _ = play_batch(games_played, players)

    result = (wins[Color.RED] / games_played - 0.5) * 4  # normalized to [-2,+2] range
    print(result, wins)
    return result


if __name__ == "__main__":
    main()
