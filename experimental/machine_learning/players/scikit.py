import pickle
from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import Action
from catanatron.models.enums import Resource, BuildingType

from experimental.machine_learning.features import create_sample
from experimental.simple_model import FEATURES


with open("experimental/models/simple-scikit.model", "rb") as file:
    clf = pickle.load(file)


def value_fn(game, p0_color):
    sample = create_sample(game, p0_color)
    vector = [float(sample[feature]) for feature in FEATURES]
    return clf.predict([vector])


class ScikitPlayer(Player):
    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = None
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value = value_fn(game_copy, self.color)
            if best_value is None or value > best_value:
                best_value = value
                best_action = action

        return best_action
