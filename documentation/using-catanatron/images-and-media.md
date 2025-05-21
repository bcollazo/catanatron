---
icon: books
---

# Python Library

You can also use `catanatron` package directly which provides a core implementation of the Settlers of Catan game logic.

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

### Iterating Over Plys (ticks)

Instead of using `game.play()` to play a game until completion, you can iterate step-by-step on each ply (decision point) like so:

```python
game = Game(players)
while game.winning_color() is None:
    print(game.state)
    action = game.play_tick()
    print(action)
```

### Debugging Game State

Inspecting the game state mid-simulation in the example above can be challenging. Even though inspecting the following is helpful:

```python
print(game.state.board.map.tiles)
print(game.state.board.buildings)
print(game.state.board.roads)
print(game.state.player_state)
```

Its often best to see them in the GUI. For this, if you have the GUI Docker Services running alongside this process you can open the game at this state in the GUI with the `open_link` function:

```python
from catanatron.web.utils import open_link
open_link(game)  # opens game in browser
```

