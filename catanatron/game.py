import uuid
import random
import sys
import copy
from enum import Enum
from typing import Iterable
from collections import namedtuple, defaultdict

from catanatron.algorithms import longest_road, largest_army
from catanatron.models.map import BaseMap
from catanatron.models.board import Board, BuildingType
from catanatron.models.board_initializer import Node
from catanatron.models.enums import Resource, DevelopmentCard
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
    for coordinate, tile in board.resource_tiles():
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for node_ref, node in tile.nodes.items():
            building = board.buildings.get(node)
            if building == None:
                continue
            elif building.building_type == BuildingType.SETTLEMENT:
                intented_payout[building.color][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building.building_type == BuildingType.CITY:
                intented_payout[building.color][tile.resource] += 2
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
    for player in players:
        tick_queue.append((player, ActionPrompt.BUILD_FIRST_SETTLEMENT))
        tick_queue.append((player, ActionPrompt.BUILD_INITIAL_ROAD))
    for player in reversed(players):
        tick_queue.append((player, ActionPrompt.BUILD_SECOND_SETTLEMENT))
        tick_queue.append((player, ActionPrompt.BUILD_INITIAL_ROAD))
    tick_queue.append((players[0], ActionPrompt.ROLL))
    return tick_queue


class Game:
    """
    This contains the complete state of the game (board + players) and the
    core turn-by-turn controlling logic.
    """

    def __init__(self, players: Iterable[Player], seed=None):
        self.seed = seed or random.randrange(sys.maxsize)
        random.seed(self.seed)

        self.id = str(uuid.uuid4())
        self.players = random.sample(players, len(players))
        self.players_by_color = {p.color: p for p in players}
        self.map = BaseMap()
        self.board = Board(self.map)
        self.actions = []  # log of all action taken by players
        self.resource_deck = ResourceDeck.starting_bank()
        self.development_deck = DevelopmentDeck.starting_bank()

        self.tick_queue = initialize_tick_queue(self.players)
        self.current_player_index = 0
        self.num_turns = 0

    def play(self, action_callback=None):
        """Runs the game until the end"""
        while self.winning_player() == None and self.num_turns < TURNS_LIMIT:
            self.play_tick(action_callback)

    def winning_player(self):
        for player in self.players:
            if player.actual_victory_points >= 10:
                return player
        return None

    def play_tick(self, action_callback=None):
        """
        Consume from queue (player, decision) to support special building phase,
            discarding, and other decisions out-of-turn.
        If nothing there, fall back to (current, playable()) for current-turn.
        """
        if len(self.tick_queue) > 0:
            (player, action_prompt) = self.tick_queue.pop(0)
        else:
            player = self.current_player()
            action_prompt = (
                ActionPrompt.PLAY_TURN if player.has_rolled else ActionPrompt.ROLL
            )

        actions = self.playable_actions(player, action_prompt)
        action = player.decide(self, actions)
        self.execute(action, action_callback=action_callback)

    def playable_actions(self, player, action_prompt):
        if action_prompt == ActionPrompt.BUILD_FIRST_SETTLEMENT:
            return initial_settlement_possibilites(player, self.board, True)
        elif action_prompt == ActionPrompt.BUILD_SECOND_SETTLEMENT:
            return initial_settlement_possibilites(player, self.board, False)
        elif action_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            return initial_road_possibilities(player, self.board, self.actions)
        elif action_prompt == ActionPrompt.MOVE_ROBBER:
            return robber_possibilities(player, self.board, self.players, False)
        elif action_prompt == ActionPrompt.ROLL:
            actions = [Action(player, ActionType.ROLL, None)]
            if player.can_play_knight():
                actions.extend(
                    robber_possibilities(player, self.board, self.players, True)
                )
            return actions
        elif action_prompt == ActionPrompt.DISCARD:
            return discard_possibilities(player)
        elif action_prompt == ActionPrompt.PLAY_TURN:
            # Buy / Build
            actions = [Action(player, ActionType.END_TURN, None)]
            actions.extend(road_possible_actions(player, self.board))
            actions.extend(settlement_possible_actions(player, self.board))
            actions.extend(city_possible_actions(player, self.board))
            can_buy_dev_card = (
                player.resource_deck.includes(ResourceDeck.development_card_cost())
                and self.development_deck.num_cards() > 0
            )
            if can_buy_dev_card:
                actions.append(Action(player, ActionType.BUY_DEVELOPMENT_CARD, None))

            # Play Dev Cards
            if player.can_play_year_of_plenty():
                actions.extend(
                    year_of_plenty_possible_actions(player, self.resource_deck)
                )
            if player.can_play_monopoly():
                actions.extend(monopoly_possible_actions(player))
            if player.can_play_knight():
                actions.extend(
                    robber_possibilities(player, self.board, self.players, True)
                )
            if player.can_play_road_building():
                actions.extend(road_building_possibilities(player, self.board))

            # Trade
            actions.extend(
                maritime_trade_possibilities(player, self.resource_deck, self.board)
            )

            return actions
        else:
            raise RuntimeError("Unknown ActionPrompt")

    def execute(self, action, action_callback=None):
        if action.action_type == ActionType.END_TURN:
            next_player_index = (self.current_player_index + 1) % len(self.players)
            self.current_player_index = next_player_index
            self.players[next_player_index].clean_turn_state()
            self.tick_queue.append((self.players[next_player_index], ActionPrompt.ROLL))
            self.num_turns += 1
        elif action.action_type == ActionType.BUILD_FIRST_SETTLEMENT:
            node = self.board.get_node_by_id(action.value)
            self.board.build_settlement(
                action.player.color, node, initial_build_phase=True
            )
            action.player.settlements_available -= 1
        elif action.action_type == ActionType.BUILD_SECOND_SETTLEMENT:
            node = self.board.get_node_by_id(action.value)
            self.board.build_settlement(
                action.player.color, node, initial_build_phase=True
            )
            action.player.settlements_available -= 1
            # yield resources of second settlement
            for tile in self.board.get_adjacent_tiles(action.value):
                if tile.resource != None:
                    action.player.resource_deck.replenish(1, tile.resource)
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            node = self.board.get_node_by_id(action.value)
            self.board.build_settlement(
                action.player.color, node, initial_build_phase=False
            )
            action.player.resource_deck -= ResourceDeck.settlement_cost()
            action.player.settlements_available -= 1
            self.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
        elif action.action_type == ActionType.BUILD_INITIAL_ROAD:
            edge = self.board.get_edge_by_id(action.value)
            self.board.build_road(action.player.color, edge)
            action.player.roads_available -= 1
        elif action.action_type == ActionType.BUILD_ROAD:
            edge = self.board.get_edge_by_id(action.value)
            self.board.build_road(action.player.color, edge)
            action.player.roads_available -= 1
            action.player.resource_deck -= ResourceDeck.road_cost()
            self.resource_deck += ResourceDeck.road_cost()  # replenish bank
        elif action.action_type == ActionType.BUILD_CITY:
            node = self.board.get_node_by_id(action.value)
            self.board.build_city(action.player.color, node)
            action.player.settlements_available += 1
            action.player.cities_available -= 1
            action.player.resource_deck -= ResourceDeck.city_cost()
            self.resource_deck += ResourceDeck.city_cost()  # replenish bank
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            if self.development_deck.num_cards() == 0:
                raise ValueError("No more development cards")
            if not action.player.resource_deck.includes(
                ResourceDeck.development_card_cost()
            ):
                raise ValueError("No money to buy development card")

            development_card = self.development_deck.random_draw()
            action.player.development_deck.replenish(1, development_card)
            action.player.resource_deck -= ResourceDeck.development_card_cost()
            self.resource_deck += ResourceDeck.development_card_cost()

            action = Action(action.player, action.action_type, development_card)
        elif action.action_type == ActionType.ROLL:
            dices = roll_dice()
            number = dices[0] + dices[1]

            if number == 7:
                players_to_discard = [
                    p for p in self.players if p.resource_deck.num_cards() > 7
                ]
                self.tick_queue.extend(
                    [(p, ActionPrompt.DISCARD) for p in players_to_discard]
                )
                self.tick_queue.append((action.player, ActionPrompt.MOVE_ROBBER))
            else:
                payout, depleted = yield_resources(
                    self.board, self.resource_deck, number
                )
                for color, resource_deck in payout.items():
                    player = self.players_by_color[color]

                    # Atomically add to player's hand and remove from bank
                    player.resource_deck += resource_deck
                    self.resource_deck -= resource_deck

            action = Action(action.player, action.action_type, dices)
            self.tick_queue.append((action.player, ActionPrompt.PLAY_TURN))
            action.player.has_rolled = True
        elif action.action_type == ActionType.DISCARD:
            # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
            hand = action.player.resource_deck.to_array()
            num_to_discard = len(hand) // 2
            discarded = random.sample(hand, k=num_to_discard)

            to_discard = ResourceDeck.from_array(discarded)

            action.player.resource_deck -= to_discard
            self.resource_deck += to_discard
        elif action.action_type == ActionType.MOVE_ROBBER:
            (coordinate, color_to_steal_from) = action.value
            self.board.robber_coordinate = coordinate
            if color_to_steal_from is not None:
                player_to_steal_from = self.players_by_color[color_to_steal_from]
                resource = player_to_steal_from.resource_deck.random_draw()
                action.player.resource_deck.replenish(1, resource)
        elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
            if not action.player.can_play_knight():
                raise ValueError("Player cant play knight card now")
            (coordinate, color_to_steal_from) = action.value
            self.board.robber_coordinate = coordinate
            if color_to_steal_from is not None:
                player_to_steal_from = self.players_by_color[color_to_steal_from]
                resource = player_to_steal_from.resource_deck.random_draw()
                action.player.resource_deck.replenish(1, resource)
            action.player.mark_played_dev_card(DevelopmentCard.KNIGHT)
        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            cards_selected = ResourceDeck.from_array(action.value)
            if not action.player.can_play_year_of_plenty():
                raise ValueError("Player cant play year of plenty now")
            if not self.resource_deck.includes(cards_selected):
                raise ValueError(
                    "Not enough resources of this type (these types?) in bank"
                )
            action.player.resource_deck += cards_selected
            self.resource_deck -= cards_selected
            action.player.mark_played_dev_card(DevelopmentCard.YEAR_OF_PLENTY)
        elif action.action_type == ActionType.PLAY_MONOPOLY:
            card_type_to_steal = action.value
            cards_stolen = ResourceDeck()
            if not action.player.can_play_monopoly():
                raise ValueError("Player cant play monopoly now")
            for player in self.players:
                if not action.player.color == player.color:
                    number_of_cards_to_steal = player.resource_deck.count(
                        card_type_to_steal
                    )
                    cards_stolen.replenish(number_of_cards_to_steal, card_type_to_steal)
                    player.resource_deck.draw(
                        number_of_cards_to_steal, card_type_to_steal
                    )
            action.player.resource_deck += cards_stolen
            action.player.mark_played_dev_card(DevelopmentCard.MONOPOLY)
        elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
            if not action.player.can_play_road_building():
                raise ValueError("Player cant play road building now")
            first_edge, second_edge = action.value
            self.board.build_road(action.player.color, first_edge)
            self.board.build_road(action.player.color, second_edge)
            action.player.roads_available -= 2
            action.player.mark_played_dev_card(DevelopmentCard.ROAD_BUILDING)
        elif action.action_type == ActionType.MARITIME_TRADE:
            trade_offer = action.value
            offering = ResourceDeck.from_array(trade_offer.offering)
            asking = ResourceDeck.from_array(trade_offer.asking)
            tradee = trade_offer.tradee or self  # self means bank
            if not action.player.resource_deck.includes(offering):
                raise ValueError("Trying to trade without money")
            if not tradee.resource_deck.includes(asking):
                raise ValueError("Tradee doenst have those cards")
            action.player.resource_deck -= offering
            tradee.resource_deck += offering
            action.player.resource_deck += asking
            tradee.resource_deck -= asking
        else:
            raise RuntimeError("Unknown ActionType " + str(action.action_type))

        # TODO: Think about possible-action/idea vs finalized-action design
        self.actions.append(action)
        self.count_victory_points()

        if action_callback:
            action_callback(self)

    def current_player(self):
        return self.players[self.current_player_index]

    def count_victory_points(self):
        road_color = longest_road(self.board, self.players, self.actions)[0]
        army_color = largest_army(self.players, self.actions)[0]

        for player in self.players:
            player.has_road = False
            player.has_army = False
        for player in self.players:
            public_vps = 0
            public_vps += len(
                self.board.get_player_buildings(player.color, BuildingType.SETTLEMENT)
            )  # count settlements
            public_vps += 2 * len(
                self.board.get_player_buildings(player.color, BuildingType.CITY)
            )  # count cities
            if road_color != None and self.players_by_color[road_color] == player:
                public_vps += 2  # road
                player.has_road = True
            if army_color != None and self.players_by_color[army_color] == player:
                public_vps += 2  # army
                player.has_army = True

            player.public_victory_points = public_vps
            player.actual_victory_points = public_vps + player.development_deck.count(
                DevelopmentCard.VICTORY_POINT
            )


def replay_game(game):
    game_copy = copy.deepcopy(game)

    for player in game_copy.players:
        player.public_victory_points = 0
        player.actual_victory_points = 0
        player.resource_deck = ResourceDeck()
        player.development_deck = DevelopmentDeck()

    tmp_game = Game(game_copy.players, seed=game.seed)
    tmp_game.id = game_copy.id
    tmp_game.players = game_copy.players  # use same seating order
    tmp_game.map = game_copy.map
    tmp_game.board = game_copy.board
    tmp_game.board.buildings = {}
    tmp_game.board.robber_coordinate = filter(
        lambda coordinate: tmp_game.board.tiles[coordinate].resource == None,
        tmp_game.board.tiles.keys(),
    ).__next__()

    for action in game_copy.actions:
        tmp_game.execute(action)
        yield tmp_game
