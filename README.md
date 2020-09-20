# catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)

Settlers of Catan simluation environment in Python and Machine-Learning player.

## Usage

### API

You can create your own AI bots/players implementing the following API:

```python
from catanatron.models.players import RandomPlayer

class MyPlayer(Player):
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def discard(self):
        """Must return n/2 cards to discard from self.resource_deck"""
        raise NotImplementedError
```

Then running games like:

```python
from catanatron.models.game import Game
from catanatron.models.players import RandomPlayer, MyPlayer

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.GREEN),
    MyPlayer(Color.WHITE)
]
game = Game(players)
game.play()
```

You can then watch the game in the debugger UI or save a serialized verion of the game state.

### Debugger UI

This brings up a React-powered web interface where you can inspect robots playing
the game.

On one tab (server):

```
pipenv install
pipenv shell
export FLASK_ENV=development
export FLASK_APP=server.py
flask run
```

On another tab (client):

```
cd ui/
yarn install
yarn start
```

## Tests

To develop for Catanatron core logic you can run the following test suite:

```
coverage run --source=catanatron -m pytest tests/ && coverage report
```
