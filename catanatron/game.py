import random
from typing import Iterable
from collections import namedtuple, defaultdict

from catanatron.models.map import BaseMap
from catanatron.models.board import Board, BuildingType
from catanatron.models.enums import Action, ActionType, Resource
from catanatron.models.player import Player
from catanatron.models.decks import ResourceDecks


def roll_dice():
    return (random.randint(1, 6), random.randint(1, 6))


def playable_actions(player, has_roll, board):
    if not has_roll:
        actions = [Action(player, ActionType.ROLL, roll_dice())]
        if player.has_knight_card():  # maybe knight
            for coordinate in board.tiles.keys():
                if coordinate != board.robber_tile.coordinate:
                    actions.append(
                        Action(player, ActionType.PLAY_KNIGHT_CARD, coordinate)
                    )

        return actions

    raise NotImplementedError


def yield_resources(board, resource_decks, number):
    """
    Returns:
        (payouts, depleted): tuple where:
        payouts: dictionary of "resource_decks" keyed by player
            e.g. {Color.RED: {Resource.WEAT: 3}}
            depleted: list of resources that couldn't yield
    """
    payout = defaultdict(lambda: defaultdict(int))
    resource_totals = defaultdict(int)
    for coordinate, tile in board.resource_tiles():
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for node_ref, node in tile.nodes.items():
            building = board.buildings.get(node)
            if building == None:
                continue
            elif building.building_type == BuildingType.SETTLEMENT:
                payout[building.color][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building.building_type == BuildingType.CITY:
                payout[building.color][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource_i in Resource:
        total = resource_totals[resource_i]
        if not resource_decks.can_draw(total, resource_i):
            depleted.append(resource_i)
            for player, player_payout in payout.items():
                payout[player] = dict(player_payout)
                for resource_j, count in player_payout.items():
                    del payout[player][resource_j]

    return dict(payout), depleted


class Game:
    """
    This contains the complete state of the game (board + players) and the
    core turn-by-turn controlling logic.
    """

    def __init__(self, players: Iterable[Player]):
        self.players = players
        self.map = BaseMap()
        self.board = Board(self.map)
        self.actions = []  # log of all action taken by players

        self.current_player_index = 0
        self.current_player_has_roll = False
        random.shuffle(self.players)

    def play(self):
        """Runs the game until the end"""
        self.play_initial_build_phase()
        while self.winning_player() == None:
            self.play_tick()

    def play_initial_build_phase(self):
        """First player goes, settlement and road, ..."""
        for player in self.players + list(reversed(self.players)):
            # Place a settlement first
            buildable_nodes = self.board.buildable_nodes(
                player.color, initial_build_phase=True
            )
            actions = list(
                map(
                    lambda node: Action(player, ActionType.BUILD_SETTLEMENT, node),
                    buildable_nodes,
                )
            )
            action = player.decide(self, actions)
            self.execute(action, initial_build_phase=True)

            # Then a road, ensure its connected to this last settlement
            buildable_edges = filter(
                lambda e: action.value in e.nodes,
                self.board.buildable_edges(player.color),
            )
            actions = list(
                map(
                    lambda edge: Action(player, ActionType.BUILD_ROAD, edge),
                    buildable_edges,
                )
            )
            action = player.decide(self, actions)
            self.execute(action, initial_build_phase=True)

    def winning_player(self):
        raise NotImplementedError

    def play_tick(self):
        current_player = self.players[self.current_player_index]

        actions = playable_actions(
            current_player, self.current_player_has_roll, self.board
        )
        action = current_player.decide(self.board, actions)

        self.execute(action)

    def execute(self, action, initial_build_phase=False):
        self.actions.append(action)

        if action.action_type == ActionType.END_TURN:
            self.current_player_index = (self.current_player_index + 1) % len(
                self.players
            )
            self.current_player_has_roll = False
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            self.board.build_settlement(
                action.player.color,
                action.value,
                initial_build_phase=initial_build_phase,
            )
        elif action.action_type == ActionType.BUILD_ROAD:
            self.board.build_road(action.player.color, action.value)
