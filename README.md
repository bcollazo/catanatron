![logo](https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/logo.png)

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/bcollazo/catanatron/blob/master/experimental/notebooks/Overview.ipynb?clone=true&runtime=bcollazo/paperspace-rl)
[![Netlify Status](https://api.netlify.com/api/v1/badges/ccd61293-6735-4eb1-a6f0-bce11d6b91fa/deploy-status)](https://app.netlify.com/sites/catanatron/deploys)

Fast Settlers of Catan Python implementation and AI research environment.

This project's goal is to create the strongest
Settlers of Catan AI possible.

## Usage

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

Currently, we can execute ~13 games per second.

### Make your own bot

You can create your own AI bots/players by implementing the following API:

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

See examples in `players.py`. You can then modify `play.py` to import your bot and test it against others.

### Library API

You can use catanatron as a stand-alone module.

```python
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
game.play()
```

You can then inspect the game state any way you want
(e.g. `game.state.actions`, `game.state.board`, `game.state.players[0].buildings`, etc...).

### Watching Games

We provide a complete `docker-compose.yml` with everything needed to
watch games (watching random bots is very enteraining!). Ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run:

```
docker-compose up
```

You can also use a handy `ensure_link` function to create a link for a particular game:

```python
from catanatron_server.utils import ensure_link
ensure_link(game)  # prints a link to open in browser
```

For `ensure_link` to work correctly, you must run `docker-compose up` in another tab. The docker-compose contains the web-server infrastructure needed to render the game in a browser.

## Architecture

For debugging and entertainment purposes, we wanted to provide a
UI with which to inspect games.

We decided to use the browser as a rendering engine (as opposed to
the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness
and the ability to use modern animation libraries in the future (https://www.framer.com/motion/ or https://www.react-spring.io/).

To achieve this, we separated the code into three components:

- **catanatron**: A pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see `setup.py`) and can be used as a Python package.

- **catanatron_server**: Contains a Flask web server in order to serve
  game states from a database to a Web UI. The idea of using a database, is to ease watching games from different processes (you can play a game in a standalone Python script and save it for viewing).

- **React Web UI**: A web UI to render games. The `ui` folder.

### Experimental Folder

The experimental folder contains unorganized code with many failed attempts at finding the best possible bot.

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

## TODO:

- Improve `catanatron` package performance.

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
  - A terminal UI? (for ease of debugging)

<!-- NOTES ====== -->
<!-- There is a catanatron PyPi package. Core implementation -->
<!-- catanatron (Python Package) -->
<!-- pip install catanatron -->

<!-- Catanatron Web. Uses React FE and Flask BE. Can run with docker.
    Can run independently; yarn start and flask run (pip install reqs)
    BE uses an ephemeral sqlite database in memory (optionally overrided with DATABASE_URL).
-->
<!-- There is a catanatron_server and React UI that host
    serve catanatron.com. You can run these locally with Docker -->

<!-- Experimental. Simulate many games. Install dev-requirements.txt. Use experimental/play.py. Games can be saved for viewing. Use --save-in-db, run Catanatron.com docker image and visit http://localhost/games/123/replay -->

<!-- Contributing. If you find contributions in any part of the project. Some ideas include: better bot. ui improvements. testing.
    a terminal UI (for ease of debugging). There exists Test Suite. -->
