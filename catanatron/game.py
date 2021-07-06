"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import Iterable, Union

from catanatron.models.enums import Action
from catanatron.state import State, apply_action
from catanatron.state_functions import player_key
from catanatron.models.map import BaseMap
from catanatron.models.player import Color, Player

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).
    """

    def __init__(
        self,
        players: Iterable[Player],
        seed: int = None,
        catan_map: BaseMap = None,
        initialize: bool = True,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (Iterable[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            catan_map (BaseMap, optional): Map configuration to use. Defaults to None.
            initialize (bool, optional): Whether to initialize. Defaults to True.
        """
        if initialize:
            self.seed = seed or random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.state = State(players, catan_map or BaseMap())

    def play(self, action_callbacks=[], decide_fn=None):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            action_callbacks (list, optional): list of functions to run after state is changed.
                These should expect state as a parameter. Defaults to [].
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.
        """
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(action_callbacks=action_callbacks, decide_fn=decide_fn)

    def play_tick(self, action_callbacks=[], decide_fn=None):
        """Advances game by one ply (player decision).

        Args:
            action_callbacks (list, optional): list of functions to run after state is changed.
                These should expect state as a parameter. Defaults to [].
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.

        Returns:
            Action: Final action (modified to be used as Log)
        """
        player = self.state.current_player()
        actions = self.state.playable_actions

        action = (
            decide_fn(player, self, actions)
            if decide_fn is not None
            else player.decide(self, actions)
        )
        return self.execute(action, action_callbacks=action_callbacks)

    def execute(
        self, action: Action, action_callbacks=[], validate_action: bool = True
    ) -> Action:
        """Internal call that carries out decided action by player"""
        if validate_action and action not in self.state.playable_actions:
            raise ValueError(
                f"{action} not in playable actions: {self.state.playable_actions}"
            )

        action = apply_action(self.state, action)

        for callback in action_callbacks:
            callback(self)

        return action

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        winning_player = None
        for player in self.state.players:
            key = player_key(self.state, player.color)
            if self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= 10:
                winning_player = player

        return None if winning_player is None else winning_player.color

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game([], None, None, initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.state = self.state.copy()
        return game_copy
