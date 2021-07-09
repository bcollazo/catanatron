from pprint import pprint
import numpy as np
import pickle
from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.actions import Action
from catanatron.models.enums import Resource, BuildingType

from catanatron_gym.features import create_sample, create_sample_vector


from experimental.simple_model import FEATURES

# from experimental.simple_forest import FEATURES

clf = None


def load_model():
    global clf
    with open("experimental/models/simple-scikit-linear.model", "rb") as file:
        clf = pickle.load(file)
    # with open("experimental/models/simple-rf.model", "rb") as file:
    #     clf = pickle.load(file)


class ScikitPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        load_model()

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        samples = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            sample = create_sample_vector(game_copy, self.color, FEATURES)
            samples.append(sample)

        scores = clf.predict(samples)
        best_idx = np.argmax(scores)
        best_action = playable_actions[best_idx]

        # pprint(list(zip(playable_actions, scores)))
        # breakpoint()
        return best_action
