# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/bcollazo-catanatron/community](https://badges.gitter.im/bcollazo-catanatron/community.svg)](https://gitter.im/bcollazo-catanatron/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/examples/Overview.ipynb)

Catanatron is a high-performance simulator and strong AI player for Settlers of Catan. You can run thousands of games in the order of seconds. The goal is to find the strongest Settlers of Catan bot possible.

Get Started with the Full Documentation: https://docs.catanatron.com

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
    pip install -e .[web,gym,dev]
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

You should now be able to visit http://localhost:3000 and play!

You can also (in a new terminal window) install the `[web]` subpackage and use the `--db` flag
to make the catanatron-play simulator save the game in the database for inspection via the web server.

```bash
pip install .[web]
catanatron-play --players=W,W,W,W --db --num=1
```

The link should be printed in the console.

NOTE: A great contribution would be to make the Web UI allow to step forwards and backwards in a game to inspect it (ala chess.com).

### Accumulators

The `Accumulator` class allows you to hook into important events during simulations.

For example, write a file like `mycode.py` and have:

```python
from catanatron import ActionType
from catanatron.cli import SimulationAccumulator, register_cli_accumulator

class PortTradeCounter(SimulationAccumulator):
  def before_all(self):
    self.num_trades = 0

  def step(self, game_before_action, action):
    if action.action_type == ActionType.MARITIME_TRADE:
      self.num_trades += 1

  def after_all(self):
    print(f'There were {self.num_trades} trades with the bank!')

register_cli_accumulator(PortTradeCounter)
```

Then `catanatron-play --code=mycode.py` will count the number of trades in all simulations.

### As a Package / Library

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
from catanatron.web.utils import open_link
open_link(game)  # opens game in browser
```

## Architecture

The code is divided in three main components (folders):

- **catanatron**: The pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see [pyproject.toml](pyproject.toml)) and can be used as a Python package. See the documentation for the package here: https://catanatron.readthedocs.io/.

  - **catanatron.web**: An extension package (optionally installed) that contains a Flask web server in order to serve
    game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process.
    It defaults to using an ephemeral in-memory sqlite database. Also pip-installable with `pip install catanatron[web]`.

  - **catanatron.gym**: Gymnasium interface to Catan. Includes a 1v1 environment against a Random Bot and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron[gym]`, for more information see [catanatron/gym/README.md](catanatron/catanatron/gym/README.md).

- **catantron_experimental**: A collection of unorganized scripts with contain many failed attempts at finding the best possible bot. Its ok to break these scripts. Its pip-installable. Exposes a `catanatron-play` command-line script that can be used to play games in bulk, create machine learning datasets of games, and more!

- **ui**: A React web UI to render games. This is helpful for debugging the core implementation. We decided to use the browser as a randering engine (as opposed to the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness and the ability to use modern animation libraries in the future (https://www.framer.com/motion/ or https://www.react-spring.io/).

## AI Bots Leaderboard

Catanatron will always be the best bot in this leaderboard.

The best bot right now is `AlphaBetaPlayer` with n = 2. Here a list of bots strength. Experiments
done by running 1000 (when possible) 1v1 games against previous in list.

| Player               | % of wins in 1v1 games      | num games used for result |
| -------------------- | --------------------------- | ------------------------- |
| AlphaBeta(n=2)       | 80% vs ValueFunction        | 25                        |
| ValueFunction        | 90% vs GreedyPlayouts(n=25) | 25                        |
| GreedyPlayouts(n=25) | 100% vs MCTS(n=100)         | 25                        |
| MCTS(n=100)          | 60% vs WeightedRandom       | 15                        |
| WeightedRandom       | 53% vs WeightedRandom       | 1000                      |
| VictoryPoint         | 60% vs Random               | 1000                      |
| Random               | -                           | -                         |

## Developing for Catanatron

To develop for Catanatron core logic, install the dev dependencies and use the following test suite:

```bash
pip install .[web,gym,dev]
coverage run --source=catanatron -m pytest tests/ && coverage report
```

See more at: https://docs.catanatron.com

## Appendix
See the motivation of the project here: [5 Ways NOT to Build a Catan AI](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17).


