# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
![Discord](https://img.shields.io/discord/1385302652014825552)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/examples/Overview.ipynb)

Catanatron is a high-performance simulator and strong AI player for Settlers of Catan. You can run thousands of games in the order of seconds. The goal is to find the strongest Settlers of Catan bot possible. 

Get Started with the Full Documentation: https://docs.catanatron.com

Join our Discord: https://discord.gg/FgFmb75TWd!

## Capstone RL Quick Start (Top-Level)

### Current Agent Architecture (Post-Refactor)

- `capstone_agent/CapstoneAgent.py` is now the combined router agent.
- Split policy modules are:
  - `capstone_agent/PlacementAgent.py` (initial settlement/road phase)
  - `capstone_agent/MainPlayAgent.py` (main gameplay phase)
- Feature space is now `1259` (adds an `is_settlement_phase` binary feature).
- A Catanatron-compatible player exists at
  `catanatron/catanatron/players/rl_capstone_agent.py` for loading trained
  placement/main-play weights into a playable bot class.

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
| `--save` | `capstone_agent/models/capstone_model.pt` | Path to save main-play model weights at the end (also used for auto-resume in train mode when present). |
| `--save-every-games` | `0` | In train mode, periodically overwrite save paths every N games (`0` disables periodic checkpointing). |
| `--placement-strategy` | `model` | Placement policy: `model` or `random`. |
| `--placement-model` | `None` | Path to load placement-agent weights (ignored for `--placement-strategy random`). |
| `--save-placement-model` | `capstone_agent/models/placement_model.pt` | Path to save placement-agent weights at the end (model strategy only). |
| `--enemy` | `random` | Opponent bot in environment (`random`, `alphabeta`, `alphabeta-prune`, `same-turn-ab`, `value`, `vp`, `weighted`). |
| `--enemy-ab-depth` | `2` | AlphaBeta depth when using an AlphaBeta-type enemy. |
| `--enemy-ab-prunning` | off | Enable pruning for `--enemy alphabeta`. |
| `--enemy-fixed-schedule` | off | Enable fixed phase-based opponent curriculum controlled by `--enemy-schedule`. |
| `--enemy-schedule` | `weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:50000` | Fixed curriculum string: `<enemy>:<games>,<enemy>@<ab_depth>:<games>,...` |
| `--map-template` | `AUTO` | Board template (`AUTO`, `BASE`, `MINI`, `TOURNAMENT`). `AUTO` selects `TOURNAMENT` for fixed mode and `BASE` for random mode. |
| `--map-mode` | `fixed` | Map layout mode: `fixed` (deterministic) or `random` (reshuffled each game). |
| `--fixed-map-seed` | `0` | Seed used when `--map-mode fixed` to generate the deterministic map. |
| `--fresh-start` | off | In train mode, ignore existing save paths and start from scratch. |
| `--run-name` | auto timestamp | Optional run label for benchmark CSV rows. |
| `--benchmark-csv` | `capstone_agent/benchmarks/training_metrics.csv` | Output CSV path for per-game benchmark logs. |
| `--no-benchmark` | off | Disable benchmark CSV logging. |
| `--save-games-json-dir` | `None` | Directory to export replay JSONs for GUI replay/import. |
| `--save-games-json-every` | `100` | Save replay JSON for first game in each N-game block. |
| `--self-play-ladder` | off | Enable champion/challenger self-play training with promotion checks. |
| `--self-play-winner-only` | on | In self-play mode, only keep winner (challenger-win) rollouts for updates. |
| `--champion-main-model` | `capstone_agent/models/champion_capstone_model.pt` | Stable champion main-play weights used as self-play opponent. |
| `--champion-placement-model` | `capstone_agent/models/champion_placement_model.pt` | Stable champion placement weights used as self-play opponent. |
| `--self-play-eval-every-games` | `1000` | Run challenger-vs-champion evaluation every N training games. |
| `--self-play-eval-games` | `400` | Number of head-to-head games per promotion check (seat-balanced). |
| `--self-play-promotion-threshold` | `0.55` | Promote challenger to champion when eval win-rate reaches this threshold. |

Common variants:

```bash
# Step-buffered training with larger PPO batches
python -m capstone_agent.run_simulation \
  --games 1000 \
  --train \
  --train-update-trigger steps \
  --train-every-steps 4096

# Game-buffered training
python -m capstone_agent.run_simulation \
  --games 1000 \
  --train \
  --train-update-trigger games \
  --train-every-games 20

# Training + replay export + explicit run label
python -m capstone_agent.run_simulation \
  --games 1000 \
  --train \
  --run-name iter_full \
  --save-games-json-dir capstone_agent/replays/iter_full

# Train/eval directly against AlphaBeta
python -m capstone_agent.run_simulation \
  --games 10 \
  --enemy alphabeta \
  --enemy-ab-depth 2 \
  --map-mode fixed \
  --placement-strategy model \
  --placement-model capstone_agent/models/placement_model.pt \
  --save-games-json-dir capstone_agent/replays/ab_placement_eval \
  --save-games-json-every 1

# Fixed schedule curriculum (weighted -> value -> AB depth 1 -> AB depth 2)
python -m capstone_agent.run_simulation \
  --games 200000 \
  --train \
  --enemy-fixed-schedule \
  --enemy-schedule "weighted:50000,value:50000,alphabeta@1:50000,alphabeta@2:50000" \
  --save-every-games 1000 \
  --save capstone_agent/models/capstone_model.pt \
  --save-placement-model capstone_agent/models/placement_model.pt

# Randomize map each game (non-tournament templates)
python -m capstone_agent.run_simulation \
  --games 200 \
  --enemy random \
  --map-mode random
```

### Quick Ops (Copy/Paste)

```bash
# 1) Run 5 games, save every replay, then import into GUI
python -m capstone_agent.run_simulation \
  --games 5 \
  --enemy random \
  --save-games-json-dir capstone_agent/replays/quick5 \
  --save-games-json-every 1

python capstone_agent/import_replays_to_gui.py \
  --input-dir capstone_agent/replays/quick5

# 1b) Self-play ladder (challenger vs champion with auto-promotion)
python -m capstone_agent.run_simulation \
  --games 20000 \
  --train \
  --self-play-ladder \
  --self-play-winner-only \
  --self-play-eval-every-games 1000 \
  --self-play-eval-games 400 \
  --self-play-promotion-threshold 0.55 \
  --save capstone_agent/models/capstone_model.pt \
  --save-placement-model capstone_agent/models/placement_model.pt \
  --champion-main-model capstone_agent/models/champion_capstone_model.pt \
  --champion-placement-model capstone_agent/models/champion_placement_model.pt \
  --run-name ladder_run \
  --benchmark-csv capstone_agent/benchmarks/training_metrics.csv

# 2) Long DCC run: 1,000,000 games as 100 x 10,000 chunks in tmux
tmux new -s dcc_train
for i in $(seq 1 100); do
  python -u -m capstone_agent.run_simulation \
    --games 10000 \
    --train \
    --save-every-games 1000 \
    --save capstone_agent/models/capstone_model.pt \
    --save-placement-model capstone_agent/models/placement_model.pt \
    --run-name dcc_1m \
    --benchmark-csv capstone_agent/benchmarks/training_metrics.csv \
    --save-games-json-dir capstone_agent/replays/dcc_1m \
    --save-games-json-every 10000 \
    --enemy random 2>&1 | tee -a dcc_1m_pretty.log
done

# Detach from tmux (job keeps running): Ctrl+b then d
# Re-attach later:
tmux attach -t dcc_train
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

### Capstone Smoke Tests (Paths + Commands)

Use these from repo root to verify the current refactor wiring:

```bash
# 1) Syntax sanity for key files
python -m py_compile \
  capstone_agent/run_simulation.py \
  capstone_agent/CapstoneAgent.py \
  capstone_agent/MainPlayAgent.py \
  capstone_agent/PlacementAgent.py \
  catanatron/catanatron/players/rl_capstone_agent.py

# 2) Short run vs random (fixed map default)
python capstone_agent/run_simulation.py --games 10 --enemy random

# 3) Short run vs AlphaBeta with learned placement
python capstone_agent/run_simulation.py \
  --games 10 \
  --enemy alphabeta \
  --enemy-ab-depth 2 \
  --placement-strategy model \
  --placement-model capstone_agent/models/placement_model.pt
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


