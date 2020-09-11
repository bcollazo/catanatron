import random
from typing import Iterable
from collections import namedtuple

from catanatron.models.map import BaseMap
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Player


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
