# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
![Discord](https://img.shields.io/discord/1385302652014825552)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/examples/Overview.ipynb)

Catanatron is a high-performance simulator and strong AI player for Settlers of Catan. You can run thousands of games in the order of seconds. The goal is to find the strongest Settlers of Catan bot possible. 

Get Started with the Full Documentation: https://docs.catanatron.com

Join our Discord: https://discord.gg/FgFmb75TWd!

## Capstone RL Quick Start (Top-Level)

Base training run (recommended default: step-buffered PPO updates):

```bash
python capstone_agent/run_simulation.py --games 1000 --train
```

This default uses:
- `--train-update-trigger steps`
- `--train-every-steps 2048`

### `run_simulation.py` Parameters

All flags for `python capstone_agent/run_simulation.py`:

| Parameter | Default | What it does |
| --- | --- | --- |
| `--games` | `1` | Number of full games to simulate. |
| `--train` | off | Enable PPO training during the run. |
| `--train-update-trigger` | `steps` | Update schedule mode: `steps` or `games`. |
| `--train-every-steps` | `2048` | If trigger is `steps`, run PPO every N buffered transitions. |
| `--train-every-games` | `20` | If trigger is `games`, run PPO every N completed games. |
| `--verbose` | off | Print selected per-step action logs during simulation. |
| `--load` | `None` | Path to load main-agent model weights before running. |
| `--save` | `capstone_agent/models/capstone_model.pt` | Path to save main-agent weights at the end (also used for auto-resume in train mode when present). |
| `--placement-strategy` | `model` | Placement policy: `model` or `random`. |
| `--placement-model` | `None` | Path to load placement-agent weights (ignored for `--placement-strategy random`). |
| `--save-placement-model` | `capstone_agent/models/placement_model.pt` | Path to save placement-agent weights at the end (model strategy only). |
| `--enemy` | `random` | Opponent bot in environment (`random`, `alphabeta`, `alphabeta-prune`, `same-turn-ab`, `value`, `vp`, `weighted`). |
| `--enemy-ab-depth` | `2` | AlphaBeta depth when using an AlphaBeta-type enemy. |
| `--enemy-ab-prunning` | off | Enable pruning for `--enemy alphabeta`. |
| `--fresh-start` | off | In train mode, ignore existing save paths and start from scratch. |
| `--run-name` | auto timestamp | Optional run label for benchmark CSV rows. |
| `--benchmark-csv` | `capstone_agent/benchmarks/training_metrics.csv` | Output CSV path for per-game benchmark logs. |
| `--no-benchmark` | off | Disable benchmark CSV logging. |
| `--save-games-json-dir` | `None` | Directory to export replay JSONs for GUI replay/import. |
| `--save-games-json-every` | `100` | Save replay JSON for first game in each N-game block. |

Common variants:

```bash
# Step-buffered training with larger PPO batches
python capstone_agent/run_simulation.py \
  --games 1000 \
  --train \
  --train-update-trigger steps \
  --train-every-steps 4096

# Game-buffered training
python capstone_agent/run_simulation.py \
  --games 1000 \
  --train \
  --train-update-trigger games \
  --train-every-games 20

# Training + replay export + explicit run label
python capstone_agent/run_simulation.py \
  --games 1000 \
  --train \
  --run-name iter_full \
  --save-games-json-dir capstone_agent/replays/iter_full

# Train/eval directly against AlphaBeta
python capstone_agent/run_simulation.py \
  --games 10 \
  --enemy alphabeta \
  --enemy-ab-depth 2 \
  --placement-strategy model \
  --placement-model capstone_agent/models/placement_model.pt \
  --save-games-json-dir capstone_agent/replays/ab_placement_eval \
  --save-games-json-every 1
```

## Command Line Interface
Catanatron provides a `catanatron-play` CLI tool to run large scale simulations.

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/cli.gif">
</p>

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:bcollazo/catanatron.git
    cd catanatron/
    ```
2. Create a virtual environment (requires Python 3.11 or higher) 

    ```bash
    python -m venv venv
    source ./venv/bin/activate
    ```
3. Install dependencies

    ```bash
    pip install -e .
    ```
4. (Optional) Install developer and advanced dependencies 

    ```bash
    pip install -e ".[web,gym,dev]"
    ```

### Usage

Run simulations and generate datasets via the CLI:

```bash
catanatron-play --players=R,R,R,W --num=100
```

Generate datasets from the games to analyze:
```bash
catanatron-play --num 100 --output my-data-path/ --output-format json
```

See more examples at https://docs.catanatron.com.

### Capstone RL Training (Resume + Sparse Replays)

Train the Capstone PPO agent while continuing from existing weights and saving
GUI-replay JSONs for the first game of each 100-game block:

```bash
python capstone_agent/run_simulation.py \
  --games 100 \
  --train \
  --save capstone_agent/capstone_model.pt \
  --run-name iter_full \
  --save-games-json-dir capstone_agent/replays/iter_full
```

Run this command repeatedly (or in a loop) to keep training on top of the
latest checkpoint. By default, replay JSON saving is sparse (`1/100`).

Import saved replay JSONs into the GUI database so they are step-through
clickable in the web app:

```bash
python capstone_agent/import_replays_to_gui.py \
  --input-dir capstone_agent/replays/iter_full
```

Then open `http://localhost:3000/replays/<game_id>` (printed by the importer).

If some replay files contain inconsistent actions, import what can be replayed
and skip bad actions while printing their indices:

```bash
python capstone_agent/import_replays_to_gui.py \
  --input-dir capstone_agent/replays/iter_full \
  --skip-invalid-actions
```

Unified analytics script (benchmarks, first-vs-second, placements):

```bash
python capstone_agent/analytics.py benchmarks \
  --csv capstone_agent/benchmarks/training_metrics.csv \
  --run-name iter_full \
  --mode train \
  --rolling-window 100 \
  --ema-span 100 \
  --chunk-size 100 \
  --out capstone_agent/benchmarks/benchmark_plot.png
```

```bash
python capstone_agent/analytics.py first-second \
  --csv capstone_agent/benchmarks/training_metrics.csv \
  --run-name iter_full \
  --mode train \
  --chunk-size 200 \
  --out capstone_agent/benchmarks/first_vs_second_200.png
```

```bash
python capstone_agent/analytics.py placements --data-dir my-data-path
```


## Graphical User Interface

We provide Docker images so that you can watch, inspect, and play games against Catanatron via a web UI!

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/CatanatronUI.png">
</p>


### Installation

1. Ensure you have Docker installed (https://docs.docker.com/engine/install/)
2. Run the `docker-compose.yaml` in the root folder of the repo:

    ```bash
    docker compose up
    ```
3. Visit http://localhost:3000 in your browser!

## Python Library

You can also use `catanatron` package directly which provides a core
implementation of the Settlers of Catan game logic.

```python
from catanatron import Game, RandomPlayer, Color

# Play a simple 4v4 game
players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color
```

See more at http://docs.catanatron.com

## Gymnasium Interface
For Reinforcement Learning, catanatron provides an Open AI / Gymnasium Environment.

Install it with:
```bash
pip install -e .[gym]
```

and use it like:
```python
import random
import gymnasium
import catanatron.gym

env = gymnasium.make("catanatron/Catanatron-v0")
observation, info = env.reset()
for _ in range(1000):
    # your agent here (this takes random actions)
    action = random.choice(info["valid_actions"])

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        observation, info = env.reset()
env.close()
```

See more at: https://docs.catanatron.com


## Documentation
Full documentation here: https://docs.catanatron.com

## Contributing

To develop for Catanatron core logic, install the dev dependencies and use the following test suite:

```bash
pip install .[web,gym,dev]
coverage run --source=catanatron -m pytest tests/ && coverage report
```

See more at: https://docs.catanatron.com

## Appendix
See the motivation of the project here: [5 Ways NOT to Build a Catan AI](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17).


