import uuid
import pickle
import random
import sys
from typing import Iterable

from catanatron.state import (
    State,
    apply_action,
    player_can_afford_dev_card,
    player_deck_can_play,
    player_has_rolled,
    player_key,
)
from catanatron.models.map import BaseMap
from catanatron.models.enums import ActionPrompt, Action, ActionType
from catanatron.models.actions import (
    road_possible_actions,
    city_possible_actions,
    settlement_possible_actions,
    robber_possibilities,
    year_of_plenty_possible_actions,
    monopoly_possible_actions,
    initial_road_possibilities,
    initial_settlement_possibilites,
    discard_possibilities,
    maritime_trade_possibilities,
    road_building_possibilities,
)
from catanatron.models.player import Player

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


class Game:
    """
    This contains the complete state of the game (board + players) and the
    core turn-by-turn controlling logic.
    """

    def __init__(
        self, players: Iterable[Player], seed=None, catan_map=None, initialize=True
    ):
        if initialize:
            self.seed = seed or random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.state = State(players, catan_map or BaseMap())
            self.advance_tick()

    def play(self, action_callbacks=[], decide_fn=None):
        """Runs the game until the end"""
        while self.winning_player() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(action_callbacks=action_callbacks, decide_fn=decide_fn)

    def winning_player(self):
        for player in self.state.players:
            key = player_key(self.state, player.color)
            if self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= 10:
                return player
        return None

    def play_tick(self, action_callbacks=[], decide_fn=None):
        """
        Consume from queue (player, decision) to support special building phase,
            discarding, and other decisions out-of-turn.
        If nothing there, fall back to (current, playable()) for current-turn.
        Assumes self.player and self.action_prompt and self.playable_actions are ready.
        """
        player = self.state.current_player()
        actions = self.state.playable_actions

        action = (
            decide_fn(player, self, actions)
            if decide_fn is not None
            else player.decide(self, actions)
        )
        return self.execute(action, action_callbacks=action_callbacks)

    def playable_actions(self, player, action_prompt):
        if action_prompt == ActionPrompt.BUILD_FIRST_SETTLEMENT:
            return initial_settlement_possibilites(player, self.state.board, True)
        elif action_prompt == ActionPrompt.BUILD_SECOND_SETTLEMENT:
            return initial_settlement_possibilites(player, self.state.board, False)
        elif action_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            return initial_road_possibilities(
                player, self.state.board, self.state.actions
            )
        elif action_prompt == ActionPrompt.MOVE_ROBBER:
            return robber_possibilities(self.state, player.color, False)
        elif action_prompt == ActionPrompt.ROLL:
            actions = [Action(player.color, ActionType.ROLL, None)]
            if player_deck_can_play(self.state, player.color, "KNIGHT"):
                actions.extend(robber_possibilities(self.state, player.color, True))
            return actions
        elif action_prompt == ActionPrompt.DISCARD:
            return discard_possibilities(player)
        elif action_prompt == ActionPrompt.PLAY_TURN:
            # Buy / Build
            actions = [Action(player.color, ActionType.END_TURN, None)]
            actions.extend(road_possible_actions(self.state, player.color))
            actions.extend(settlement_possible_actions(self.state, player.color))
            actions.extend(city_possible_actions(self.state, player.color))
            can_buy_dev_card = (
                player_can_afford_dev_card(self.state, player.color)
                and self.state.development_deck.num_cards() > 0
            )
            if can_buy_dev_card:
                actions.append(
                    Action(player.color, ActionType.BUY_DEVELOPMENT_CARD, None)
                )

            # Play Dev Cards
            if player_deck_can_play(self.state, player.color, "YEAR_OF_PLENTY"):
                actions.extend(
                    year_of_plenty_possible_actions(player, self.state.resource_deck)
                )
            if player_deck_can_play(self.state, player.color, "MONOPOLY"):
                actions.extend(monopoly_possible_actions(player))
            if player_deck_can_play(self.state, player.color, "KNIGHT"):
                actions.extend(robber_possibilities(self.state, player.color, True))
            if player_deck_can_play(self.state, player.color, "ROAD_BUILDING"):
                actions.extend(road_building_possibilities(player, self.state.board))

            # Trade
            actions.extend(maritime_trade_possibilities(self.state, player.color))

            return actions
        else:
            raise RuntimeError("Unknown ActionPrompt")

    def execute(self, action, action_callbacks=[], validate_action=True):
        if validate_action and action not in self.state.playable_actions:
            raise ValueError(
                f"{action} not in playable actions: {self.state.playable_actions}"
            )

        action = apply_action(self.state, action)

        self.advance_tick()

        for callback in action_callbacks:
            callback(self)

        return action

    def advance_tick(self):
        if len(self.state.tick_queue) > 0:
            (seating, action_prompt) = self.state.tick_queue.pop(0)
            self.state.current_player_index = seating
            player = self.state.current_player()
        else:
            player = self.state.current_player()
            action_prompt = (
                ActionPrompt.PLAY_TURN
                if player_has_rolled(self.state, player.color)
                else ActionPrompt.ROLL
            )

        self.state.current_prompt = action_prompt
        self.state.playable_actions = self.playable_actions(player, action_prompt)

    def current_player(self):
        return self.state.players[self.state.current_player_index]

    def winning_color(self):
        player = self.winning_player()
        return None if player is None else player.color

    def copy(self) -> "Game":
        state_copy = State(None, None, initialize=False)
        state_copy.players = self.state.players
        state_copy.player_state = self.state.player_state.copy()
        state_copy.color_to_index = self.state.color_to_index
        state_copy.buildings_by_color = pickle.loads(
            pickle.dumps(self.state.buildings_by_color)
        )
        state_copy.board = self.state.board.copy()
        state_copy.actions = self.state.actions.copy()
        # TODO: Move Deck to functional code, so as to quick-copy arrays.
        state_copy.resource_deck = pickle.loads(pickle.dumps(self.state.resource_deck))
        state_copy.development_deck = pickle.loads(
            pickle.dumps(self.state.development_deck)
        )

        state_copy.tick_queue = self.state.tick_queue.copy()
        state_copy.current_player_index = self.state.current_player_index
        state_copy.num_turns = self.state.num_turns
        state_copy.current_prompt = self.state.current_prompt
        state_copy.playable_actions = self.state.playable_actions

        game_copy = Game([], None, None, initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.state = state_copy
        return game_copy
