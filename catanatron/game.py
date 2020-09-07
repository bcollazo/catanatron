import random
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


# TODO: This will contain the turn-by-turn controlling logic.
class Game:
    """This will contain the rest of the state information (# victory points,
    # dev cards, etc... via the players attr)"""

    def __init__(self, players):
        self.players = players
        self.map = BaseMap()
        self.board = Board(self.map)
        self.actions = []  # log of all action taken by players

        self.current_player_index = 0
        self.current_player_has_roll = False

    def play(self):
        """Runs the game until the end"""
        self.play_initial_building_phase()
        while self.winning_player() == None:
            self.play_tick()

    def play_initial_building_phase(self):
        raise NotImplementedError

    def winning_player(self):
        raise NotImplementedError

    def play_tick(self):
        current_player = self.players[self.current_player_index]

        playable_actions = playable_actions(
            current_player, self.current_player_has_roll
        )
        action = current_player.decide(self.board, playable_actions)

        self.execute(action)

    def execute(self, action, initial_building_phase=False):
        self.actions.append(action)

        if action.action_type == ActionType.END_TURN:
            self.current_player_index = (self.current_player_index + 1) % len(
                self.players
            )
            self.current_player_has_roll = False
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            (coordinate, node_ref) = action.value
            self.board.build_settlement(
                action.player.color,
                coordinate,
                node_ref,
                initial_placement=initial_building_phase,
            )
        elif action.action_type == ActionType.BUILD_ROAD:
            (coordinate, edge_ref) = action.value
            self.board.build_road(action.player.color, coordinate, edge_ref)
