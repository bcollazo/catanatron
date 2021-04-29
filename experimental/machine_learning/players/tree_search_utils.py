from collections import defaultdict

from catanatron.models.decks import DevelopmentDeck
from catanatron.models.enums import DevelopmentCard, Resource
from catanatron.models.actions import Action, ActionType


def execute_spectrum(game, action):
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
        copy = game.copy()
        copy.execute(action, validate_action=False)
        return [(copy, 1)]
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        results = []
        for card in DevelopmentCard:
            option_action = Action(action.color, action.action_type, card)
            option_game = game.copy()
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
                option_game = game.copy()
                option_game.execute(option_action, validate_action=False)
                results.append((option_game, 1 / 36.0))
        return results
    elif action.action_type in [
        ActionType.MOVE_ROBBER,
        ActionType.PLAY_KNIGHT_CARD,
    ]:
        (coordinate, robbed_color, _) = action.value
        if robbed_color is None:  # no one to steal, then deterministic
            copy = game.copy()
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
                option_game = game.copy()
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


def expand_spectrum(game, actions):
    """Consumes game if playable_actions not specified"""
    children = defaultdict(list)
    for action in actions:
        outprobas = execute_spectrum(game, action)
        children[action] = outprobas
    return children  # action => (game, proba)[]
