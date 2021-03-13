import uuid
import pickle
import random
import sys
from typing import Iterable
from collections import defaultdict

from catanatron.algorithms import longest_road, largest_army
from catanatron.models.map import BaseMap
from catanatron.models.board import Board
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


def roll_dice():
    return (random.randint(1, 6), random.randint(1, 6))


def yield_resources(board, resource_deck, number):
    """
    Returns:
        (payouts, depleted): tuple where:
        payouts: dictionary of "resource_deck" keyed by player
                e.g. {Color.RED: ResourceDeck({Resource.WEAT: 3})}
            depleted: list of resources that couldn't yield
    """
    intented_payout = defaultdict(lambda: defaultdict(int))
    resource_totals = defaultdict(int)
    for coordinate, tile in board.map.resource_tiles:
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for _, node_id in tile.nodes.items():
            building = board.buildings.get(node_id, None)
            if building is None:
                continue
            elif building[1] == BuildingType.SETTLEMENT:
                intented_payout[building[0]][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building[1] == BuildingType.CITY:
                intented_payout[building[0]][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource in Resource:
        total = resource_totals[resource]
        if not resource_deck.can_draw(total, resource):
            depleted.append(resource)

    # build final data ResourceDeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = ResourceDeck()

        for resource, count in player_payout.items():
            if resource not in depleted:
                payout[player].replenish(count, resource)

    return payout, depleted


def initialize_tick_queue(players):
    """First player goes, settlement and road, ..."""
    tick_queue = []
    for seating in range(len(players)):
        tick_queue.append((seating, ActionPrompt.BUILD_FIRST_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    for seating in range(len(players) - 1, -1, -1):
        tick_queue.append((seating, ActionPrompt.BUILD_SECOND_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    tick_queue.append((0, ActionPrompt.ROLL))
    return tick_queue


class State:
    """Small container object to group dynamic variables in state"""

    def __init__(self, players, map, initialize=True):
        if initialize:
            self.players = random.sample(players, len(players))
            self.players_by_color = {p.color: p for p in players}
            self.board = Board(map)
            self.actions = []  # log of all action taken by players
            self.resource_deck = ResourceDeck.starting_bank()
            self.development_deck = DevelopmentDeck.starting_bank()
            self.tick_queue = initialize_tick_queue(self.players)
            self.current_player_index = 0
            self.num_turns = 0
            self.road_color = None
            self.army_color = None


class Game:
    """
    This contains the complete state of the game (board + players) and the
    core turn-by-turn controlling logic.
    """

    def __init__(self, players: Iterable[Player], seed=None, map=None, initialize=True):
        if initialize:
            self.seed = seed or random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.state = State(players, map or BaseMap())

    def play(self, action_callbacks=[], decide_fn=None):
        """Runs the game until the end"""
        while self.winning_player() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(action_callbacks=action_callbacks, decide_fn=decide_fn)

    def winning_player(self):
        for player in self.state.players:
            if player.actual_victory_points >= 10:
                return player
        return None

    def winning_color(self):
        player = self.winning_player()
        return None if player is None else player.color

    def pop_from_queue(self):
        """Important: dont call this without consuming results. O/W illegal state"""
        if len(self.state.tick_queue) > 0:
            (seating, action_prompt) = self.state.tick_queue.pop(0)
            player = self.state.players[seating]
        else:
            player = self.current_player()
            action_prompt = (
                ActionPrompt.PLAY_TURN if player.has_rolled else ActionPrompt.ROLL
            )
        return player, action_prompt

    def play_tick(self, action_callbacks=[], decide_fn=None):
        """
        Consume from queue (player, decision) to support special building phase,
            discarding, and other decisions out-of-turn.
        If nothing there, fall back to (current, playable()) for current-turn.
        """
        player, action_prompt = self.pop_from_queue()

        actions = self.playable_actions(player, action_prompt)
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

    def execute(self, action, action_callbacks=[]):
        outcome_proba = None
        if action.action_type == ActionType.END_TURN:
            next_player_index = (self.state.current_player_index + 1) % len(
                self.state.players
            )
            self.state.current_player_index = next_player_index
            self.state.players[next_player_index].clean_turn_state()
            self.state.tick_queue.append((next_player_index, ActionPrompt.ROLL))
            self.state.num_turns += 1
        elif action.action_type == ActionType.BUILD_FIRST_SETTLEMENT:
            player, node_id = self.state.players_by_color[action.color], action.value
            self.state.board.build_settlement(player.color, node_id, True)
            player.build_settlement(node_id, True)
        elif action.action_type == ActionType.BUILD_SECOND_SETTLEMENT:
            player, node_id = self.state.players_by_color[action.color], action.value
            self.state.board.build_settlement(player.color, node_id, True)
            player.build_settlement(node_id, True)
            # yield resources of second settlement
            for tile in self.state.board.map.get_adjacent_tiles(node_id):
                if tile.resource != None:
                    self.state.resource_deck.draw(1, tile.resource)
                    player.resource_deck.replenish(1, tile.resource)
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            player, node_id = self.state.players_by_color[action.color], action.value
            self.state.board.build_settlement(player.color, node_id, False)
            player.build_settlement(node_id, False)
            self.state.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
            self.state.road_color = longest_road(
                self.state.board, self.state.players, self.state.actions
            )[0]
        elif action.action_type == ActionType.BUILD_INITIAL_ROAD:
            player, edge = self.state.players_by_color[action.color], action.value
            self.state.board.build_road(player.color, edge)
            player.build_road(edge, True)
        elif action.action_type == ActionType.BUILD_ROAD:
            player, edge = self.state.players_by_color[action.color], action.value
            self.state.board.build_road(player.color, edge)
            player.build_road(edge, False)
            self.state.resource_deck += ResourceDeck.road_cost()  # replenish bank
            self.state.road_color = longest_road(
                self.state.board, self.state.players, self.state.actions
            )[0]
        elif action.action_type == ActionType.BUILD_CITY:
            player, node_id = self.state.players_by_color[action.color], action.value
            self.state.board.build_city(player.color, node_id)
            player.build_city(node_id)
            self.state.resource_deck += ResourceDeck.city_cost()  # replenish bank
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            player = self.state.players_by_color[action.color]
            if self.state.development_deck.num_cards() == 0:
                raise ValueError("No more development cards")
            if not player.resource_deck.includes(ResourceDeck.development_card_cost()):
                raise ValueError("No money to buy development card")

            if action.value is None:
                card = self.state.development_deck.random_draw()
            else:
                card = action.value
                self.state.development_deck.draw(1, card)

            player.development_deck.replenish(1, card)
            player.resource_deck -= ResourceDeck.development_card_cost()
            self.state.resource_deck += ResourceDeck.development_card_cost()

            action = Action(action.color, action.action_type, card)
        elif action.action_type == ActionType.ROLL:
            player = self.state.players_by_color[action.color]
            dices = action.value or roll_dice()
            number = dices[0] + dices[1]

            if number == 7:
                seatings_to_discard = [
                    seating
                    for seating, player in enumerate(self.state.players)
                    if player.resource_deck.num_cards() > 7
                ]
                self.state.tick_queue.extend(
                    [(seating, ActionPrompt.DISCARD) for seating in seatings_to_discard]
                )
                self.state.tick_queue.append(
                    (self.state.current_player_index, ActionPrompt.MOVE_ROBBER)
                )
            else:
                payout, _ = yield_resources(
                    self.state.board, self.state.resource_deck, number
                )
                for color, resource_deck in payout.items():
                    player = self.state.players_by_color[color]

                    # Atomically add to player's hand and remove from bank
                    player.resource_deck += resource_deck
                    self.state.resource_deck -= resource_deck

            action = Action(action.color, action.action_type, dices)
            self.state.tick_queue.append(
                (self.state.current_player_index, ActionPrompt.PLAY_TURN)
            )
            player.has_rolled = True
        elif action.action_type == ActionType.DISCARD:
            player = self.state.players_by_color[action.color]
            hand = player.resource_deck.to_array()
            num_to_discard = len(hand) // 2
            if action.value is None:
                # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
                discarded = random.sample(hand, k=num_to_discard)
            else:
                discarded = action.value  # for replay functionality
            to_discard = ResourceDeck.from_array(discarded)

            player.resource_deck -= to_discard
            self.state.resource_deck += to_discard
            action = Action(action.color, action.action_type, discarded)
        elif action.action_type == ActionType.MOVE_ROBBER:
            player = self.state.players_by_color[action.color]
            (coordinate, robbed_color, robbed_resource) = action.value
            self.state.board.robber_coordinate = coordinate
            if robbed_color is not None:
                player_to_steal_from = self.state.players_by_color[robbed_color]
                enemy_cards = player_to_steal_from.resource_deck.num_cards()
                if robbed_resource is None:
                    resource = player_to_steal_from.resource_deck.random_draw()
                    action = Action(
                        action.color,
                        action.action_type,
                        (coordinate, robbed_color, resource),
                    )
                else:  # for replay functionality
                    resource = robbed_resource
                    player_to_steal_from.resource_deck.draw(1, resource)
                player.resource_deck.replenish(1, resource)
        elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
            player = self.state.players_by_color[action.color]
            if not player.can_play_knight():
                raise ValueError("Player cant play knight card now")
            (coordinate, robbed_color, robbed_resource) = action.value
            self.state.board.robber_coordinate = coordinate
            if robbed_color is not None:
                player_to_steal_from = self.state.players_by_color[robbed_color]
                enemy_cards = player_to_steal_from.resource_deck.num_cards()
                if robbed_resource is None:
                    resource = player_to_steal_from.resource_deck.random_draw()
                    action = Action(
                        action.color,
                        action.action_type,
                        (coordinate, robbed_color, resource),
                    )
                else:  # for replay functionality
                    resource = robbed_resource
                    player_to_steal_from.resource_deck.draw(1, resource)
                player.resource_deck.replenish(1, resource)
            player.mark_played_dev_card(DevelopmentCard.KNIGHT)
            self.state.army_color = largest_army(
                self.state.players, self.state.actions
            )[0]
        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            player = self.state.players_by_color[action.color]
            cards_selected = ResourceDeck.from_array(action.value)
            if not player.can_play_year_of_plenty():
                raise ValueError("Player cant play year of plenty now")
            if not self.state.resource_deck.includes(cards_selected):
                raise ValueError(
                    "Not enough resources of this type (these types?) in bank"
                )
            player.resource_deck += cards_selected
            self.state.resource_deck -= cards_selected
            player.mark_played_dev_card(DevelopmentCard.YEAR_OF_PLENTY)
        elif action.action_type == ActionType.PLAY_MONOPOLY:
            player, mono_resource = (
                self.state.players_by_color[action.color],
                action.value,
            )
            cards_stolen = ResourceDeck()
            if not player.can_play_monopoly():
                raise ValueError("Player cant play monopoly now")
            total_enemy_cards = 0
            for p in self.state.players:
                if not p.color == action.color:
                    number_of_cards_to_steal = p.resource_deck.count(mono_resource)
                    cards_stolen.replenish(number_of_cards_to_steal, mono_resource)
                    p.resource_deck.draw(number_of_cards_to_steal, mono_resource)
                    total_enemy_cards += p.resource_deck.num_cards()
            player.resource_deck += cards_stolen
            player.mark_played_dev_card(DevelopmentCard.MONOPOLY)
        elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
            player, (first_edge, second_edge) = (
                self.state.players_by_color[action.color],
                action.value,
            )
            if not player.can_play_road_building():
                raise ValueError("Player cant play road building now")

            self.state.board.build_road(player.color, first_edge)
            self.state.board.build_road(player.color, second_edge)
            player.build_road(first_edge, True)
            player.build_road(second_edge, True)
            player.mark_played_dev_card(DevelopmentCard.ROAD_BUILDING)
            self.state.road_color = longest_road(
                self.state.board, self.state.players, self.state.actions
            )[0]
        elif action.action_type == ActionType.MARITIME_TRADE:
            player, trade_offer = (
                self.state.players_by_color[action.color],
                action.value,
            )
            offering = ResourceDeck.from_array(trade_offer.offering)
            asking = ResourceDeck.from_array(trade_offer.asking)
            tradee = trade_offer.tradee or self  # self means bank
            if not player.resource_deck.includes(offering):
                raise ValueError("Trying to trade without money")
            if not isinstance(tradee, Game):
                raise ValueError("Non-bank tradee not supported")
            if not tradee.state.resource_deck.includes(asking):
                raise ValueError("Tradee doenst have those cards")
            player.resource_deck -= offering
            tradee.state.resource_deck += offering
            player.resource_deck += asking
            tradee.state.resource_deck -= asking
        else:
            raise RuntimeError("Unknown ActionType " + str(action.action_type))

        # TODO: Think about possible-action/idea vs finalized-action design
        self.state.actions.append(action)
        self.count_victory_points()

        for callback in action_callbacks:
            callback(self)

        return outcome_proba

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
            copy.execute(action)
            return [(copy, 1)]
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            results = []
            for card in DevelopmentCard:
                option_action = Action(action.color, action.action_type, card)
                option_game = self.copy()
                try:
                    option_game.execute(option_action)
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
                    option_game.execute(option_action)
                    results.append((option_game, 1 / 36.0))
            return results
        elif action.action_type in [
            ActionType.MOVE_ROBBER,
            ActionType.PLAY_KNIGHT_CARD,
        ]:
            (coordinate, robbed_color, _) = action.value
            if robbed_color is None:  # no one to steal, then deterministic
                copy = self.copy()
                copy.execute(action)
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
                        option_game.execute(option_action)
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

        game_copy = Game([], None, None, initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.state = state_copy
        return game_copy


def replay_game(game):
    game_copy = game.copy()

    # reset game state re-using the board (map really)
    tmp_game = Game(
        game_copy.state.players, seed=game.seed, map=game_copy.state.board.map
    )
    tmp_game.id = game_copy.id  # TODO: needed?
    for player in tmp_game.state.players:
        player.restart_state()

    for action in game_copy.state.actions:
        tmp_game.execute(action)
        yield tmp_game, action
