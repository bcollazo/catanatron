import optuna

from catanatron.models.player import Color
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
)

from catanatron.cli.play import play_batch


def objective(trial):
    weights = {
        # Where to place. Note winning is best at all costs
        "public_vps": trial.suggest_float("public_vps", 0.0, 100.0),
        "production": trial.suggest_float("production", 0.0, 100.0),
        "enemy_production": -trial.suggest_float("enemy_production", 0.0, 100.0),
        "num_tiles": trial.suggest_float("num_tiles", 0.0, 100.0),
        # Towards where to expand and when
        "reachable_production_0": trial.suggest_float(
            "reachable_production_0", 0.0, 100.0
        ),
        "reachable_production_1": trial.suggest_float(
            "reachable_production_1", 0.0, 100.0
        ),
        "buildable_nodes": trial.suggest_float("buildable_nodes", 0.0, 100.0),
        "longest_road": trial.suggest_float("longest_road", 0.0, 100.0),
        # Hand, when to hold and when to use.
        "hand_synergy": trial.suggest_float("hand_synergy", 0.0, 100.0),
        "hand_resources": trial.suggest_float("hand_resources", 0.0, 100.0),
        "discard_penalty": -trial.suggest_float("discard_penalty", 0.0, 100.0),
        "hand_devs": trial.suggest_float("hand_devs", 0.0, 100.0),
        "army_size": trial.suggest_float("army_size", 0.0, 100.0),
    }

    players = [
        # AlphaBetaPlayer(Color.RED, 2, True),
        # AlphaBetaPlayer(Color.BLUE, 2, True, "C", weights),
        ValueFunctionPlayer(Color.RED, "C", params=DEFAULT_WEIGHTS),
        ValueFunctionPlayer(Color.BLUE, "C", params=weights),
    ]
    wins, results_by_player = play_batch(200, players)
    vps = results_by_player[players[1].color]
    avg_vps = sum(vps) / len(vps)
    return 1000 * wins[players[1].color] + avg_vps


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="optunatan",
        direction="maximize",
        load_if_exists=True,
        storage="sqlite:///optunatan.db",
    )
    study.optimize(
        objective,
        n_trials=100,
    )  # Invoke optimization of the objective function.
