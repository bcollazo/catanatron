import uuid
import random
from typing import Iterable
from collections import namedtuple, defaultdict

from catanatron.algorithms import longest_road
from catanatron.models.map import BaseMap
from catanatron.models.board import Board, BuildingType
from catanatron.models.board_initializer import Node
from catanatron.models.enums import Resource, DevelopmentCard
from catanatron.models.actions import (
    Action,
    ActionType,
    road_possible_actions,
    city_possible_actions,
    settlement_possible_actions,
    robber_possibilities,
    year_of_plenty_possible_actions,
    monopoly_possible_actions,
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


class Game:
    """
    This contains the complete state of the game (board + players) and the
    core turn-by-turn controlling logic.
    """

    def __init__(self, players: Iterable[Player]):
        self.id = str(uuid.uuid4())
        self.players = players
        self.players_by_color = {p.color: p for p in players}
        self.map = BaseMap()
        self.board = Board(self.map)
        self.actions = []  # log of all action taken by players
        self.resource_deck = ResourceDeck.starting_bank()
        self.development_deck = DevelopmentDeck.starting_bank()

        # variables to keep track of what to do next
        self.current_player_index = 0
        self.current_player_has_roll = False
        self.moving_robber = False
        self.tick_queue = []
        random.shuffle(self.players)

        self.num_turns = 0

    def play(self, action_callback=None):
        """Runs the game until the end"""
        self.play_initial_build_phase(action_callback)
        while self.winning_player() == None and self.num_turns < TURNS_LIMIT:
            self.play_tick(action_callback)

    def play_initial_build_phase(self, action_callback=None):
        """First player goes, settlement and road, ..."""
        for player in self.players + list(reversed(self.players)):
            # Place a settlement first
            buildable_nodes = self.board.buildable_nodes(
                player.color, initial_build_phase=True
            )
            actions = map(
                lambda node: Action(player, ActionType.BUILD_SETTLEMENT, node),
                buildable_nodes,
            )
            action = player.decide(self, list(actions))
            self.execute(
                action, initial_build_phase=True, action_callback=action_callback
            )

            # Then a road, ensure its connected to this last settlement
            buildable_edges = filter(
                lambda e: action.value in e.nodes,
                self.board.buildable_edges(player.color),
            )
            actions = map(
                lambda edge: Action(player, ActionType.BUILD_ROAD, edge),
                buildable_edges,
            )
            action = player.decide(self, list(actions))
            self.execute(
                action, initial_build_phase=True, action_callback=action_callback
            )

        # yield resources of second settlement
        second_settlements = map(
            lambda a: (a.player, a.value),
            filter(
                lambda a: a.action_type == ActionType.BUILD_SETTLEMENT,
                self.actions[len(self.players) * 2 :],
            ),
        )
        for (player, node) in second_settlements:
            for tile in self.board.get_adjacent_tiles(node):
                if tile.resource != None:
                    player.resource_deck.replenish(1, tile.resource)

        # TODO: For the robot to better learn, refactor this method to use .execute via
        #   ActionType.BUILD_FIRST_SETTLEMENT and ActionType.BUILD_SECOND_SETTLEMENT.
        #   So that it learns second settlement yields.
        # if action_callback:
        #     action_callback()

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
            (player, actions) = self.tick_queue.pop()
        else:
            player = self.players[self.current_player_index]
            actions = self.playable_actions(player)
        action = player.decide(self.board, actions)
        self.execute(action, action_callback=action_callback)

    def playable_actions(self, player):
        if self.moving_robber:
            return robber_possibilities(player, self.board, self.players)

        if not self.current_player_has_roll:
            actions = [Action(player, ActionType.ROLL, None)]
            if player.has_knight_card():  # maybe knight
                # TODO: Change to PLAY_KNIGHT_CARD
                actions.extend(robber_possibilities(player, self.board, self.players))

            return actions

        actions = [Action(player, ActionType.END_TURN, None)]
        for action in road_possible_actions(player, self.board):
            actions.append(action)
        for action in settlement_possible_actions(player, self.board):
            actions.append(action)
        for action in city_possible_actions(player, self.board):
            actions.append(action)

        # Can only do if the player has not already played a development card
        if player.has_year_of_plenty_card():
            for action in year_of_plenty_possible_actions(player, self.resource_deck):
                actions.append(action)
        if player.has_monopoly_card():
            for action in monopoly_possible_actions(player):
                actions.append(action)

        if (
            player.resource_deck.includes(ResourceDeck.development_card_cost())
            and self.development_deck.num_cards() > 0
        ):
            actions.append(Action(player, ActionType.BUY_DEVELOPMENT_CARD, None))

        return actions

    def execute(self, action, initial_build_phase=False, action_callback=None):
        if action.action_type == ActionType.END_TURN:
            self.current_player_index = (self.current_player_index + 1) % len(
                self.players
            )
            self.current_player_has_roll = False
            self.num_turns += 1
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            self.board.build_settlement(
                action.player.color,
                action.value,
                initial_build_phase=initial_build_phase,
            )
            if not initial_build_phase:
                action.player.resource_deck -= ResourceDeck.settlement_cost()
                self.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
        elif action.action_type == ActionType.BUILD_ROAD:
            self.board.build_road(action.player.color, action.value)
            if not initial_build_phase:
                action.player.resource_deck -= ResourceDeck.road_cost()
                self.resource_deck += ResourceDeck.road_cost()  # replenish bank
        elif action.action_type == ActionType.BUILD_CITY:
            self.board.build_city(action.player.color, action.value)
            action.player.resource_deck -= ResourceDeck.city_cost()
            self.resource_deck += ResourceDeck.city_cost()  # replenish bank
        elif action.action_type == ActionType.ROLL:
            dices = roll_dice()
            number = dices[0] + dices[1]

            if number == 7:
                players_to_discard = [
                    p for p in self.players if p.resource_deck.num_cards() > 7
                ]
                self.tick_queue.extend(
                    [
                        (p, [Action(p, ActionType.DISCARD, None)])
                        for p in players_to_discard
                    ]
                )
                self.moving_robber = True
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
            self.current_player_has_roll = True
        elif action.action_type == ActionType.MOVE_ROBBER:
            (coordinate, player_to_steal_from) = action.value
            self.board.robber_coordinate = coordinate
            if player_to_steal_from is not None:
                resource = player_to_steal_from.resource_deck.random_draw()
                self.current_player().resource_deck.replenish(1, resource)

            self.moving_robber = False
        elif action.action_type == ActionType.DISCARD:
            num_cards = action.player.resource_deck.num_cards()
            discarded = action.player.discard()
            assert len(discarded) == num_cards // 2

            to_discard = ResourceDeck()
            for resource in discarded:
                to_discard.replenish(1, resource)
            action.player.resource_deck -= to_discard
            self.resource_deck += to_discard

            action = Action(action.player, action.action_type, discarded)
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
        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            cards_selected = action.value  # Assuming action.value is a resource deck
            player_to_act = action.player
            if (
                not player_to_act.development_deck.count(DevelopmentCard.YEAR_OF_PLENTY)
                > 0
            ):
                raise ValueError("Player doesn't have year of plenty card")
            if not self.resource_deck.includes(cards_selected):
                raise ValueError(
                    "Not enough resources of this type (these types?) in bank"
                )
            player_to_act.resource_deck += cards_selected
            player_to_act.development_deck.draw(1, DevelopmentCard.YEAR_OF_PLENTY)
            self.resource_deck -= cards_selected
        elif action.action_type == ActionType.PLAY_MONOPOLY:
            player_to_act = action.player
            card_type_to_steal = action.value
            cards_stolen = ResourceDeck()
            if not player_to_act.has_monopoly_card():
                raise ValueError("Player doesn't have monopoly card")
            for player in self.players:
                if not player_to_act.color == player.color:
                    number_of_cards_to_steal = player.resource_deck.count(
                        card_type_to_steal
                    )
                    cards_stolen.replenish(number_of_cards_to_steal, card_type_to_steal)
                    player.resource_deck.draw(
                        number_of_cards_to_steal, card_type_to_steal
                    )
            player_to_act.resource_deck += cards_stolen
            player_to_act.development_deck.draw(1, DevelopmentCard.MONOPOLY)

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
        (color, path) = longest_road(self.board, self.players, self.actions)
        # TODO: Count largest army

        for player in self.players:
            public_vps = 0
            public_vps += len(
                self.board.get_player_buildings(player.color, BuildingType.SETTLEMENT)
            )  # count settlements
            public_vps += 2 * len(
                self.board.get_player_buildings(player.color, BuildingType.CITY)
            )  # count cities
            if color != None and self.players_by_color[color] == player:
                public_vps += 2

            player.public_victory_points = public_vps
            player.actual_victory_points = public_vps + player.development_deck.count(
                DevelopmentCard.VICTORY_POINT
            )
