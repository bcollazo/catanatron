import builtins
import dataclasses
import random

from enum import Enum
from typing import Any, Callable, Optional

from catanatron.observer import GameObserver
from catanatron.params import NoParams


class Color(Enum):
    """Enum to represent the colors in the game"""

    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"

    def __repr__(self):
        return f"C.{self.name}"


class Player(GameObserver):
    """Interface to represent a player's decision logic.

    A player has two very different kinds of attribute, and they are kept
    separate on purpose:

    - ``color`` is the player's identity in a game. It is assigned by the
      game, not chosen by the bot author.
    - ``params`` is the bot's own configuration, declared as a nested
      ``Params`` model. It is the single source of truth for what is tunable:
      the CLI parses it, and ``GET /api/players`` publishes it.

    Whether a player is a bot is a property of the class (``IS_BOT``), not
    something passed in per instance.

    A player is also a :class:`~catanatron.observer.GameObserver`: override
    ``before``/``step``/``after`` to watch the game, not just act in it.
    ``step`` in particular is the only way to see what opponents did between
    your own turns.
    """

    #: Model declaring this player's tunable configuration.
    Params: Any = NoParams
    #: False for players whose decisions come from a human.
    IS_BOT = True
    #: Display name, when the class name is not the one to show a user.
    LABEL = ""

    def __init__(self, color: Color, params: Optional[Any] = None):
        """Initialize the player

        Args:
            color(Color): the color of the player
            params(Params, optional): this player's configuration. Defaults to
                ``type(self).Params()``, i.e. all declared defaults.
        """
        self.color = color
        self.params = params if params is not None else type(self).Params()

    @property
    def is_bot(self):
        return type(self).IS_BOT

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions or
        an OFFER_TRADE action if its your turn and you have already rolled.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def __repr__(self):
        fields = vars(self.params) if dataclasses.is_dataclass(self.params) else {}
        scalars = {
            k: v for k, v in fields.items() if isinstance(v, (int, float, str, bool))
        }
        inner = ",".join(f"{k}={v}" for k, v in scalars.items())
        return f"{type(self).__name__}:{self.color.value}" + (
            f"({inner})" if inner else ""
        )


class SimplePlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def decide(self, game, playable_actions):
        return playable_actions[0]


class HumanPlayer(Player):
    """Human player that selects which action to take using standard input"""

    LABEL = "Human (terminal)"
    IS_BOT = False

    @dataclasses.dataclass(frozen=True)
    class Params:
        #: Not externally settable (non-scalar); exists as a testing seam.
        input_fn: Callable = builtins.input

    def decide(self, game, playable_actions):
        for i, action in enumerate(playable_actions):
            print(f"{i}: {action.action_type} {action.value}")
        i = None
        while i is None or (i < 0 or i >= len(playable_actions)):
            print("Please enter a valid index:")
            try:
                x = self.params.input_fn(">>> ")
                i = int(x)
            except ValueError:
                pass

        return playable_actions[i]


class RandomPlayer(Player):
    """Random AI player that selects an action randomly from the list of playable_actions"""

    LABEL = "Random"

    def decide(self, game, playable_actions):
        return game.state.random.choice(playable_actions)
