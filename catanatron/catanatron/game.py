"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import Sequence, Union, Optional

from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.state import State
from catanatron.apply_action import apply_action
from catanatron.state_functions import player_key, player_has_rolled
from catanatron.models.map import CatanMap, NumberPlacement
from catanatron.models.player import Color, Player
from catanatron.observer import GameObserver

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def is_valid_action(playable_actions, state: State, action: Action) -> bool:
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE in the right time."""
    if action.action_type == ActionType.OFFER_TRADE:
        return (
            state.current_color() == action.color
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, action.color)
            and is_valid_trade(action.value)
        )

    return action in playable_actions


def is_valid_trade(action_value):
    """Checks the value of a OFFER_TRADE does not
    give away resources or trade matching resources.
    """
    offering = action_value[:5]
    asking = action_value[5:]
    if sum(offering) == 0 or sum(asking) == 0:
        return False  # cant give away cards

    for i, j in zip(offering, asking):
        if i > 0 and j > 0:
            return False  # cant trade same resources
    return True


class GameAccumulator(GameObserver):
    """A :class:`~catanatron.observer.GameObserver` that only watches.

    Kept as a distinct name because accumulators are passed to ``play()``
    explicitly, while players are observed by virtue of being seated.
    """


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).

    Attributes:
        state (State): Current game state.
        playable_actions (List[Action]): List of playable actions by current player.
    """

    def __init__(
        self,
        players: Sequence[Player],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        friendly_robber: bool = False,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        number_placement: NumberPlacement = "official_spiral",
        initialize: bool = True,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (List[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            discard_limit (int, optional): Discard limit to use. Defaults to 7.
            vps_to_win (int, optional): Victory Points needed to win. Defaults to 10.
            catan_map (CatanMap, optional): Map to use. Defaults to None.
            initialize (bool, optional): Whether to initialize. Defaults to True.
        """
        #: Everything watching this game: the seated players, plus any
        #: accumulators handed to play(). One list, so each hook is one loop.
        self.observers = []
        #: The subset that overrides step(), so the per-action loop skips
        #: players that only decide. Purely an optimization.
        self._steppers = []
        if initialize:
            self.seed = seed if seed is not None else random.randrange(sys.maxsize)
            self.random = random.Random(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.friendly_robber = friendly_robber
            self.state = State(
                players,
                catan_map,
                discard_limit=discard_limit,
                friendly_robber=friendly_robber,
                number_placement=number_placement,
                rng=self.random,
            )
            self.playable_actions = generate_playable_actions(self.state)

            # Seat the players as observers here rather than in play(), so
            # before() fires exactly once per game no matter who drives the
            # loop (play, play_tick, or the web server).
            for player in self.state.players:
                self.watch(player)

    def watch(self, observer):
        """Have ``observer`` follow this game from here on."""
        observer.before(self)
        self.observers.append(observer)
        if type(observer).step is not GameObserver.step:
            self._steppers.append(observer)

    def play(self, accumulators=[], decide_fn=None):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            accumulators (list[Accumulator], optional): list of Accumulator classes to use.
                Their .consume method will be called with every action, and
                their .finalize method will be called when the game ends (if it ends)
                Defaults to [].
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.
        Returns:
            Color: winning color or None if game exceeded TURNS_LIMIT
        """
        for accumulator in accumulators:
            self.watch(accumulator)
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(decide_fn=decide_fn)
        for observer in self.observers:
            observer.after(self)
        return self.winning_color()

    def play_tick(self, decide_fn=None):
        """Advances game by one ply (player decision).

        Args:
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.

        Returns:
            ActionRecord: representing the executed action
        """
        # Ask Player for action
        player = self.state.current_player()
        action = (
            decide_fn(player, self, self.playable_actions)
            if decide_fn is not None
            else player.decide(self, self.playable_actions)
        )

        # Call step here, because we want game_before_action, action
        for observer in self._steppers:
            observer.step(self, action)

        # Apply Action, and do Move Generation
        return self.execute(action)

    def execute(
        self,
        action: Action,
        validate_action: bool = True,
        action_record: ActionRecord = None,
    ) -> ActionRecord:
        """Internal call that carries out decided action by player"""
        if validate_action and not is_valid_action(
            self.playable_actions, self.state, action
        ):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.playable_actions}"
            )

        action_record = apply_action(self.state, action, action_record)
        self.playable_actions = generate_playable_actions(self.state)
        return action_record

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        result = None
        for color in self.state.colors:
            key = player_key(self.state, color)
            if (
                self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
                >= self.vps_to_win
            ):
                result = color

        return result

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game(players=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.random = self.random
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.friendly_robber = self.friendly_robber
        game_copy.state = self.state.copy()
        game_copy.playable_actions = self.playable_actions
        return game_copy
