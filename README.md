# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/bcollazo/catanatron/blob/master/experimental/notebooks/Overview.ipynb?clone=true&runtime=bcollazo/paperspace-rl)

Fast Settlers of Catan Python implementation and strong AI player.

The goal of this project is to find the strongest Settlers of Catan bot possible.

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/sample-board.png" height="300">
</p>

## Usage

Install with pip:

```
pip install catanatron
```

Make your own bot by implementing the following API (see examples in `catanatron/players` and `experimental/machine_learning/players`):

```python
from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.players import Player

class MyPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        raise NotImplementedError
```

Then run a game (or many) like:

```python
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color

players = [
    MyPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
game.play()  # returns winning color
```

You can then inspect the game state any way you want
(e.g. `game.state.player_state`, `game.state.actions`, `game.state.board.buildings`, etc...). See [documentation](#documentation) for more.

For watching these games in a UI see
[watching games](#watching-games).

## Advanced Usage

Cloning the repo and using directly will allow you to access additional tools not included in the core package. In particular, a web UI for watching games and a `experimental/play.py` script that provides a blueprint to run many games, collect summary statistics (avg vps, avg game length, etc...),
save game for viewing in browser, and/or generate machine learning datasets.

Create a virtualenv with Python 3.8 and install requirements:

```
python3.8 -m venv venv
source ./venv/bin/activate
pip install -r dev-requirements.txt
```

Run games with the `play.py` script. It provides extra options you can explore with `--help`:

```
python experimental/play.py --num=100
```

Currently, we can execute one game in ~76 milliseconds.

## Watching Games

We provide a `docker-compose.yml` with everything needed to watch games (useful for debugging). It contains all the web-server infrastructure needed to render a game in a browser.

Ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run:

```
docker-compose up
```

To open a game from another command line process in the browser, set the following environment variable:

```
export DATABASE_URL=postgresql://catanatron:victorypoint@localhost:5432/catanatron_db
```

and use the `open_link` helper function:

```python
from catanatron_server.utils import open_link
open_link(game)  # opens game in browser
```

## Documentation

See https://catanatron.readthedocs.io for more details on how we represent the [state](https://catanatron.readthedocs.io/en/latest/catanatron.html#catanatron.state.State) and [actions](https://catanatron.readthedocs.io/en/latest/catanatron.models.html#catanatron.models.enums.Action).

In summary, Actions are tuples of enums like: `(ActionType.PLAY_MONOPOLY, Resource.WHEAT)` or `(ActionType.BUILD_SETTLEMENT, 3)` (i.e. build settlement on node 3).

State is currently represented by a simple data container class and is mutated by the functions in the `state_functions` module. This functional style allows us to create state copies (for bots that search through state space) faster. The closer we make this State class to an array of immutable primitives, the faster it will be to copy.

## Architecture

For debugging and entertainment purposes, we wanted to provide a
UI with which to inspect games.

We decided to use the browser as a rendering engine (as opposed to
the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness
and the ability to use modern animation libraries in the future (https://www.framer.com/motion/ or https://www.react-spring.io/).

To achieve this, we separated the code into three components:

- **catanatron**: A pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see `setup.py`) and can be used as a Python package.

- **catanatron_server**: Contains a Flask web server in order to serve
  game states from a database to a Web UI. The idea of using a database, is to ease watching games from different processes (you can play a game in a standalone Python script and save it for viewing). It defaults to using an ephemeral in-memory sqlite database.

- **React Web UI**: A web UI to render games. The `ui` folder.

### Experimental Folder

The experimental folder contains unorganized code with many failed attempts at finding the best possible bot.

### AI Bots Leaderboard

The best bot is `AlphaBetaPlayer` with n = 2. Here a list of bots strength. Experiments
done by running 1000 (when possible) 1v1 games against previous in list.

| Player               | % of wins in 1v1 games      | num games used for result |
| -------------------- | --------------------------- | ------------------------- |
| AlphaBeta(n=2)       | 80% vs ValueFunction        | 25                        |
| ValueFunction        | 90% vs GreedyPlayouts(n=25) | 25                        |
| GreedyPlayouts(n=25) | 100% vs MCTS(n=100)         | 25                        |
| MCTS(n=100)          | 60% vs WeightedRandom       | 15                        |
| WeightedRandom       | 53% vs WeightedRandom       | 1000                      |
| VictoryPoint         | 60% vs Random               | 1000                      |
| Random               | -                           | 1000                      |

## Developing for Catanatron

To develop for Catanatron core logic you can use the following test suite:

```
coverage run --source=catanatron -m pytest tests/ && coverage report
```

Or you can run the suite in watch-mode with:

```
ptw --ignore=tests/integration_tests/ --nobeep
```

## Machine Learning

Generate data (GZIP CSVs of features and PGSQL rows) by running:

```
python experimental/play.py --num=100 --outpath=my-data-path/
```

You can then use this data to build a machine learning model, and then
implement a `Player` subclass that implements the corresponding "predict"
step of your model. There are some examples of these type of
players in `experimental/machine_learning/players/reinforcement.py`.

# Appendix

## Running Components Individually

As an alternative to running the project with Docker, you can run the following 3 components: a React UI, a Flask Web Server, and a PostgreSQL database in three separate Terminal tabs.

### React UI

Make sure you have `yarn` installed (https://classic.yarnpkg.com/en/docs/install/).

```
cd ui/
yarn install
yarn start
```

This can also be run via Docker independetly like (after building):

```
docker build -t bcollazo/catanatron-react-ui:latest ui/
docker run -it -p 3000:3000 bcollazo/catanatron-react-ui
```

### Flask Web Server

Ensure you are inside a virtual environment with all dependencies installed and
use `flask run`.

```
python3.8 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
flask run
```

This can also be run via Docker independetly like (after building):

```
docker build -t bcollazo/catanatron-server:latest .
docker run -it -p 5000:5000 bcollazo/catanatron-server
```

### PostgreSQL Database

Make sure you have `docker-compose` installed (https://docs.docker.com/compose/install/).

```
docker-compose up
```

Or run any other database deployment (locally or in the cloud).

## Other Useful Commands

### TensorBoard

For watching training progress, use `keras.callbacks.TensorBoard` and open TensorBoard:

```
tensorboard --logdir logs
```

### Docker GPU TensorFlow

```
docker run -it tensorflow/tensorflow:latest-gpu-jupyter bash
docker run -it --rm -v $(realpath ./notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

### Testing Performance

```
python -m cProfile -o profile.pstats experimental/play.py --num=5
snakeviz profile.pstats
```

```
pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:10% --benchmark-columns=min,max,mean,stddev
```

### Head Large Datasets with Pandas

```
In [1]: import pandas as pd
In [2]: x = pd.read_csv("data/mcts-playouts-labeling-2/labels.csv.gzip", compression="gzip", iterator=True)
In [3]: x.get_chunk(10)
```

### Publishing to PyPi

```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*
```

### Building Docs

```
sphinx-quickstart docs
sphinx-apidoc -o docs/source catanatron
sphinx-build -b html docs/source/ docs/build/html
```

# Contributing

I am new to Open Source Development, so open to suggestions on this section. The best contributions would be to make the core bot stronger by tinkering with the weights of each of the hand-crafted features in `experimental/machine_learning/players/minimax.py`, or coming up with
new hand-crafted features!

Here is also a list of ideas:

- Improve `catanatron` package running time performance.

  - Continue refactoring the State to be more and more like a primitive `dict` or `array`.
    (Copies are much faster if State is just a native python object).
  - Move RESOURCE to be ints. Python `enums` turned out to be slow for hashing and using.
  - Move .actions to a Game concept. (to avoid copying when copying State)
  - Remove .current_prompt. It seems its redundant with (is_moving_knight, etc...) and not needed.

- Improve AlphaBetaPlayer:

  - Explore and improve prunning
  - Use Bayesian Methods or SPSA to tune weights and find better ones.

- Experiment ideas:

  - DQN Render Method. Use models/mbs=64\_\_1619973412.model. Try to understand it.
  - DQN Two Layer Algo. With Simple Action Space.
  - Simple Alpha Go
  - Try Tensorforce with simple action space.
  - Try simple flat CSV approach but with AlphaBeta-generated games.
  - Visualize tree with graphviz. With colors per maximizing/minimizing.

- Bugs:

  - Shouldn't be able to use dev card just bought.

- Features:

  - Continue implementing actions from the UI (not all implemented).
  - Chess.com-like UI for watching game replays (with Play/Pause and Back/Forward).
  - A terminal UI? (for ease of debugging)
