# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)

The goal of this project is to build the strongest Settlers of Catan bot possible.

See the motivation of the project here: [5 Ways NOT to Build a Catan AI](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17).

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/sample-board.png" height="300">
</p>

## Getting Started

The main usage is to clone this repo, install dependencies, and run simulations/experiments
so as to find stronger bot implementations and strategies.

1. Clone the repository

```
git clone https://github.com/bcollazo/catanatron
cd catanatron/
```

2. Install dependencies (needs Python3.8)

```
pip install -r dev-requirements.txt
pip install -e catanatron_core
pip install -e catanatron_server
pip install -e catanatron_gym
pip install -e catanatron_experimental
```

3. Use `catanatron-play` to run simulations and generate datasets!

```
catanatron-play --players=R,W,F,AB:2 --num=1000
```

Currently, we can execute one game in ~76 milliseconds and the best bot is `AB:2` (basically an Alpha Beta Prunning algorithm with a hand-crafted value function).

See more information with `catanatron-play --help`.

## How to make Catanatron stronger?

There are two main ways of testing a potentially stronger bot.

- Use Contender Bot in `catanatron-play`. There is already some infrastructure in which you can change the weights of the hand-crafted features in the current evaluation function right now ([minimax.py](catanatron_experimental/catanatron_experimental/machine_learning/players/minimax.py)). You could also come up with new hand-crafted features! To do this, edit the `CONTENDER_WEIGHTS` and/or `contender_fn` function and run a command like: `catanatron-play --players=AB:2,AB:2:False:C --num=100` to see if your changes improve the main bot (it should consistently win more games).

- If your bot idea is considerably different than the tree-search structure `AlphaBetaPlayer` follows, implement your idea in the provided `MyPlayer` stub ([my_player.py](catanatron_experimental/catanatron_experimental/my_player.py)), and test it against the best bot: `catanatron-play --players=AB:2,M --num=100`. See other example player implementations in [catanatron_core/catanatron/players](catanatron_core/catanatron/players) and [minimax.py](catanatron_experimental/catanatron_experimental/machine_learning/players/minimax.py)

If you find a bot that consistently beats the best bot right now, please submit a Pull Request! :)

## Watching Games (Browser UI)

We provide a [docker-compose.yml](docker-compose.yml) with everything needed to watch games (useful for debugging). It contains all the web-server infrastructure needed to render a game in a browser.

To use, ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run (from repo root):

```
docker-compose up
```

Then alongside it, use the `open_link` helper function to open up any finished game you have:

```python
from catanatron_server.utils import open_link
open_link(game)  # opens game in browser
```

See [sample.py](sample.py) for an example of this (run `python sample.py`).

NOTE: A great contribution would be to make the Web UI allow to step forwards and backwards in a game to inspect it (ala chess.com).

## Architecture

The code is divided in the following 5 components (folders):

- **catanatron**: A pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see `setup.py`) and can be used as a Python package. See the documentation for the package here: https://catanatron.readthedocs.io/.

- **catanatron_server**: Contains a Flask web server in order to serve
  game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process. It defaults to using an ephemeral in-memory sqlite database. Also pip-installable (not publised in PyPi however).

- **catanatron_gym**: OpenAI Gym interface to Catan. Includes a 1v1 environment against a Random Bot and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron_gym`, for more information see [catanatron_gym/README.md](catanatron_gym/README.md).

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

To develop for Catanatron core logic you can use the following test suite:

```
coverage run --source=catanatron -m pytest tests/ && coverage report
```

Or you can run the suite in watch-mode with:

```
ptw --ignore=tests/integration_tests/ --nobeep
```

## Machine Learning

Generate data (several GZIP CSVs of features) by running:

```
catanatron-play --num=100 --outpath=my-data-path/
```

You can then use this data to build a machine learning model, and then
implement a `Player` subclass that implements the corresponding "predict"
step of your model. There are some attempts of these type of
players in [reinforcement.py](catanatron_experimental/catanatron_experimental/machine_learning/players/reinforcement.py).

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
python -m cProfile -o profile.pstats catanatron_experimental/catanatron_experimental/play.py --num=5
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

catanatron Package

```
make build PACKAGE=catanatron_core
make upload PACKAGE=catanatron_core
make upload-production PACKAGE=catanatron_core
```

catanatron_gym Package

```
make build PACKAGE=catanatron_gym
make upload PACKAGE=catanatron_gym
make upload-production PACKAGE=catanatron_gym
```

### Building Docs

```
sphinx-quickstart docs
sphinx-apidoc -o docs/source catanatron
sphinx-build -b html docs/source/ docs/build/html
```

# Contributing

I am new to Open Source Development, so open to suggestions on this section. The best contributions would be to make the core bot stronger.

Other than that here is also a list of ideas:

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
  - Create simple entry-point notebook for this project. Runnable via Paperspace. (might be hard because catanatron requires Python 3.8 and I haven't seen a GPU-enabled tensorflow+jupyter+pyhon3.8 Docker Image out there).

- Bugs:

  - Shouldn't be able to use dev card just bought.

- Features:

  - Continue implementing actions from the UI (not all implemented).
  - Chess.com-like UI for watching game replays (with Play/Pause and Back/Forward).
  - A terminal UI? (for ease of debugging)
