import uuid
import pickle
import random
import sys
from typing import Iterable

from catanatron.state import State, apply_action
from catanatron.models.map import BaseMap
from catanatron.models.enums import Resource, DevelopmentCard, BuildingType
from catanatron.models.actions import (
    ActionPrompt,
    Action,
    ActionType,
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
from catanatron.models.decks import ResourceDeck, DevelopmentDeck

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def number_probability(number):
    return {
        2: 2.778,
        3: 5.556,
        4: 8.333,
        5: 11.111,
        6: 13.889,
        7: 16.667,
        8: 13.889,
        9: 11.111,
        10: 8.333,
        11: 5.556,
        12: 2.778,
    }[number] / 100


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
            if player.actual_victory_points >= 10:
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
            return robber_possibilities(
                player, self.state.board, self.state.players, False
            )
        elif action_prompt == ActionPrompt.ROLL:
            actions = [Action(player.color, ActionType.ROLL, None)]
            if player.can_play_knight():
                actions.extend(
                    robber_possibilities(
                        player, self.state.board, self.state.players, True
                    )
                )
            return actions
        elif action_prompt == ActionPrompt.DISCARD:
            return discard_possibilities(player)
        elif action_prompt == ActionPrompt.PLAY_TURN:
            # Buy / Build
            actions = [Action(player.color, ActionType.END_TURN, None)]
            actions.extend(road_possible_actions(player, self.state.board))
            actions.extend(settlement_possible_actions(player, self.state.board))
            actions.extend(city_possible_actions(player))
            can_buy_dev_card = (
                player.resource_deck.includes(ResourceDeck.development_card_cost())
                and self.state.development_deck.num_cards() > 0
            )
            if can_buy_dev_card:
                actions.append(
                    Action(player.color, ActionType.BUY_DEVELOPMENT_CARD, None)
                )

            # Play Dev Cards
            if player.can_play_year_of_plenty():
                actions.extend(
                    year_of_plenty_possible_actions(player, self.state.resource_deck)
                )
            if player.can_play_monopoly():
                actions.extend(monopoly_possible_actions(player))
            if player.can_play_knight():
                actions.extend(
                    robber_possibilities(
                        player, self.state.board, self.state.players, True
                    )
                )
            if player.can_play_road_building():
                actions.extend(road_building_possibilities(player, self.state.board))

            # Trade
            actions.extend(
                maritime_trade_possibilities(
                    player, self.state.resource_deck, self.state.board
                )
            )

            return actions
        else:
            raise RuntimeError("Unknown ActionPrompt")

    def execute(self, action, action_callbacks=[], validate_action=True):
        if validate_action and action not in self.state.playable_actions:
            raise ValueError(
                f"{action} not in playable actions: {self.state.playable_actions}"
            )

        action = apply_action(self.state, action)

        # TODO: Think about possible-action/idea vs finalized-action design
        self.state.actions.append(action)
        self.count_victory_points()
        self.advance_tick()

        for callback in action_callbacks:
            callback(self)

    def advance_tick(self):
        if len(self.state.tick_queue) > 0:
            (seating, action_prompt) = self.state.tick_queue.pop(0)
            self.state.current_player_index = seating
            player = self.state.current_player()
        else:
            player = self.state.current_player()
            action_prompt = (
                ActionPrompt.PLAY_TURN if player.has_rolled else ActionPrompt.ROLL
            )

        self.state.current_prompt = action_prompt
        self.state.playable_actions = self.playable_actions(player, action_prompt)

    def execute_spectrum(self, action, action_callbacks=[]):
        """Returns [(game_copy, proba), ...] tuples for result of given action.
        Result probas should add up to 1. Does not modify self"""
        deterministic_actions = set(
            [
                ActionType.END_TURN,
                ActionType.BUILD_FIRST_SETTLEMENT,
                ActionType.BUILD_SECOND_SETTLEMENT,
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_INITIAL_ROAD,
                ActionType.BUILD_ROAD,
                ActionType.BUILD_CITY,
                ActionType.PLAY_YEAR_OF_PLENTY,
                ActionType.PLAY_ROAD_BUILDING,
                ActionType.MARITIME_TRADE,
                ActionType.DISCARD,  # for simplicity... ok if reality is slightly different
                ActionType.PLAY_MONOPOLY,  # for simplicity... we assume good card-counting and bank is visible...
            ]
        )
        if action.action_type in deterministic_actions:
            copy = self.copy()
            copy.execute(action, validate_action=False)
            return [(copy, 1)]
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            results = []
            for card in DevelopmentCard:
                option_action = Action(action.color, action.action_type, card)
                option_game = self.copy()
                try:
                    option_game.execute(option_action, validate_action=False)
                except Exception:
                    # ignore exceptions, since player might imagine impossible outcomes.
                    # ignoring means the value function of this node will be flattened,
                    # to the one before.
                    pass
                results.append((option_game, DevelopmentDeck.starting_card_proba(card)))
            return results
        elif action.action_type == ActionType.ROLL:
            results = []
            for outcome_a in range(1, 7):
                for outcome_b in range(1, 7):
                    outcome = (outcome_a, outcome_b)
                    option_action = Action(action.color, action.action_type, outcome)
                    option_game = self.copy()
                    option_game.execute(option_action, validate_action=False)
                    results.append((option_game, 1 / 36.0))
            return results
        elif action.action_type in [
            ActionType.MOVE_ROBBER,
            ActionType.PLAY_KNIGHT_CARD,
        ]:
            (coordinate, robbed_color, _) = action.value
            if robbed_color is None:  # no one to steal, then deterministic
                copy = self.copy()
                copy.execute(action, validate_action=False)
                return [(copy, 1)]
            else:
                results = []
                for card in Resource:
                    option_action = Action(
                        action.color,
                        action.action_type,
                        (coordinate, robbed_color, card),
                    )
                    option_game = self.copy()
                    try:
                        option_game.execute(option_action, validate_action=False)
                    except Exception:
                        # ignore exceptions, since player might imagine impossible outcomes.
                        # ignoring means the value function of this node will be flattened,
                        # to the one before.
                        pass
                    results.append((option_game, 1 / 5.0))
                return results
        else:
            raise RuntimeError("Unknown ActionType " + str(action.action_type))

    def current_player(self):
        return self.state.players[self.state.current_player_index]

    def winning_color(self):
        player = self.winning_player()
        return None if player is None else player.color

    def count_victory_points(self):
        for player in self.state.players:
            player.has_road = False
            player.has_army = False
        for player in self.state.players:
            public_vps = 0
            public_vps += len(player.buildings[BuildingType.SETTLEMENT])
            public_vps += 2 * len(player.buildings[BuildingType.CITY])
            if (
                self.state.road_color != None
                and self.state.players_by_color[self.state.road_color] == player
            ):
                public_vps += 2  # road
                player.has_road = True
            if (
                self.state.army_color != None
                and self.state.players_by_color[self.state.army_color] == player
            ):
                public_vps += 2  # army
                player.has_army = True

            player.public_victory_points = public_vps
            player.actual_victory_points = public_vps + player.development_deck.count(
                DevelopmentCard.VICTORY_POINT
            )

    def copy(self) -> "Game":
        players = pickle.loads(pickle.dumps(self.state.players))
        board = pickle.loads(pickle.dumps(self.state.board))

        state_copy = State(None, None, initialize=False)
        state_copy.players = players
        state_copy.players_by_color = {p.color: p for p in players}
        state_copy.board = board
        state_copy.actions = self.state.actions.copy()
        state_copy.resource_deck = pickle.loads(pickle.dumps(self.state.resource_deck))
        state_copy.development_deck = pickle.loads(
            pickle.dumps(self.state.development_deck)
        )
        state_copy.tick_queue = self.state.tick_queue.copy()
        state_copy.current_player_index = self.state.current_player_index
        state_copy.num_turns = self.state.num_turns
        state_copy.road_color = self.state.road_color
        state_copy.army_color = self.state.army_color
        state_copy.current_prompt = self.state.current_prompt
        state_copy.playable_actions = self.state.playable_actions

        game_copy = Game([], None, None, initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.state = state_copy
        return game_copy
