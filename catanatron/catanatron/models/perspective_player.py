"""
The engine-side adapter that feeds fair agents Observations.

The game loop always hands Player.decide a perfect-information Game.
PerspectivePlayer converts that Game into an Observation for the wrapped
agent on every decision, so the agent never perceives more than a real
player could observe.
"""

from catanatron.features import create_sample
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.observation import Observation
from catanatron.models.player import Player


def _sanitize_record(record, observer_color):
    """Returns a copy of the record with hidden identities redacted.

    Full detail is retained for the observer's own records. For opponent
    records: development-card purchases are redacted (both channels), and
    stolen-card identities are redacted unless the observer was the victim.
    Discards are public per tournament convention.
    """
    action = record.action
    if action.color == observer_color:
        return record

    action_type = action.action_type
    if action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return ActionRecord(Action(action.color, action_type, None), None)
    elif action_type == ActionType.MOVE_ROBBER:
        robbed_color = action.value[1] if action.value is not None else None
        if robbed_color == observer_color:
            return record
        return ActionRecord(action, None)
    return record


def _sanitize_history(game, observer_color):
    """Sanitizes the full-truth action log for the given observer color."""
    return [
        _sanitize_record(record, observer_color)
        for record in game.state.action_records
    ]


class PerspectivePlayer(Player):
    """Player that adapts an ObservationAgent into the perfect-info game loop.

    On every decision the given game is projected down to the agent's color
    as an Observation (features, sanitized history, and trade state) and the
    agent's decide_observation is called. The agent never receives the Game.

    Args:
        agent (ObservationAgent): the fair agent to drive.
        is_bot (bool): whether this player is a bot. Defaults to True.
    """

    def __init__(self, agent, is_bot=True):
        self.agent = agent
        super().__init__(agent.color, is_bot)

    def decide(self, game, playable_actions):
        color = self.agent.color
        observation = Observation(
            color=color,
            features=create_sample(game, color),
            public_history=_sanitize_history(game, color),
            current_prompt=game.state.current_prompt,
            current_trade=game.state.current_trade,
            acceptees=game.state.acceptees,
        )
        return self.agent.decide_observation(observation, playable_actions)

    def reset_state(self):
        self.agent.reset_state()
