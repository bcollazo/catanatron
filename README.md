# catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/bcollazo/catanatron/blob/master/experimental/notebooks/Overview.ipynb?clone=true&runtime=paperspace/fastai)

Settlers of Catan simluation environment in Python and Machine-Learning player.

## Usage

You can create your own AI bots/players implementing the following API:

```python
from catanatron.models.players import Player

class MyPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError
```

## Catanatron Core and Catanatron Server

The code is divided into two python packages for ease of using independently. One is the `catanatron` package which contains
the core game logic and is a pure-python implementation of Catan,
with no extra dependencies.

The other is the `catanatron_server` which requires a database and
is meant to provide the complete system in which one can view the
games via the browser.

We provide a complete `docker-compose.yml` with everything needed to
watch bots play. Ensure you have Docker Compose installed, and:

```
docker build -t catanatron-server:latest .
docker build -t catanatron-react-ui:latest ui/
docker-compose up
```

To run independently:

```
docker run -it -p 5000:5000 catanatron-server
docker run -it -p 3000:3000 catanatron-react-ui
```

## Running Components Individually

The above docker-compose.yml just runs the following 3 components: a React UI, a Flask Web Server, and a PostgreSQL database.

Each can also be run independently in three different Terminal tabs.

### React UI

Make sure you have `yarn` installed (https://classic.yarnpkg.com/en/docs/install/).

```
cd ui/
yarn install
yarn start
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

### PostgreSQL Database

Make sure you have `docker-compose` installed (https://docs.docker.com/compose/install/).

```
docker-compose up
```

### Experimental Folder

The experimental folder contains not-so-well code used to
run experiments in finding the best possible bot.

After bringing up the three components, you can run the `play.py` script which
will run a game and print out a link to watch the game.

```
python experimental/play.py
```

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

To play games and save to database (for experience):

```
python experimental/play.py --num=100
```

For watching training progress, use `keras.callbacks.TensorBoard` and open TensorBoard:

```
tensorboard --logdir logs
```

### Docker GPU TensorFlow

```
docker run -it tensorflow/tensorflow:latest-gpu-jupyter bash
docker run -it --rm -v $(realpath ./notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

## For Testing Performance:

```
python -m cProfile experimental/play.py --num=5 -o profile.pstats
snakeviz profile.pstats
```

## Future Work

- Player to player trading
- Handle Discard high branching factor
- Learn no matter what the board looks like (board-topology independent features)
- Better inference on dev-cards (e.g. holding dev likely vp, didnt used road building on opportunity, etc...)
