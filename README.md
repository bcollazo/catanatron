# catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/bcollazo/catanatron/blob/master/experimental/notebooks/Overview.ipynb?clone=true&runtime=bcollazo/paperspace-rl)

Settlers of Catan simluation environment in Python and Machine-Learning player.

## Usage

```python
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron_server.utils import open_game

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
game.play()

# You can now explore the game state any way you want.
# e.g. game.actions, game.board, game.players[0].buildings, etc...
open_game(game)  # or open the game in a browser
```

For `open_game` to work correctly, you must run `docker-compose up` in another tab. The docker-compose contains the web-server infrastructure needed to render the game in a browser.

### Implementing your own bot

You can create your own AI bots/players by implementing the following API:

```python
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

### Running Games in Bulk

After `docker-compose up`, you can run many games with the `play.py` script. It provides extra options you can explore with `--help`.

```
python experimental/play.py --num=1000
```

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

We provide a complete `docker-compose.yml` with everything needed to
watch games (watching random bots is very enteraining!). Ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run:

```
docker build -t catanatron-server:latest .
docker build -t catanatron-react-ui:latest ui/
docker-compose up
```

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
docker run -it -p 3000:3000 catanatron-react-ui
```

### Flask Web Server

Make sure you have `pipenv` installed (https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv).

```
pipenv install
pipenv shell
export FLASK_ENV=development
export FLASK_APP=server.py
flask run
```

This can also be run via Docker independetly like (after building):

```
docker run -it -p 5000:5000 catanatron-server
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
python -m cProfile experimental/play.py --num=5 -o profile.pstats
snakeviz profile.pstats
```
