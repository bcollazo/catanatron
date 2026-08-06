"""
A lazy, read-only, presentation-free view of the game for exactly one color.

The Observation is a bot's entire information world. It never exposes the
underlying Game or State, so a bot that only perceives Observations cannot
read information no human could observe.
"""

from functools import cached_property

from catanatron.features import create_sample
from catanatron.models.enums import Action, ActionRecord, ActionType


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


class Observation:
    """Lazy, read-only view of the game for exactly one color.

    Args:
        game (Game): full-truth engine state. Kept private.
        color (Color): the color whose perspective this view represents.
    """

    def __init__(self, game, color):
        self._game = game
        self.color = color

    @cached_property
    def features(self):
        """Imperfect-information snapshot, keyed relative to this color (P0).

        Lazy; computed once per Observation. Reuses the RL extractors, so
        fair bots share the exact representation agents train on.
        """
        return create_sample(self._game, self.color)

    @property
    def public_history(self):
        """Sanitized action log (list of ActionRecord).

        Hidden identities carried by opponent records are redacted according
        to the table in the ADR. Runs on access and is O(history length).
        """
        return [
            _sanitize_record(record, self.color)
            for record in self._game.state.action_records
        ]

    @property
    def current_prompt(self):
        """The ActionPrompt the current player must respond to."""
        return self._game.state.current_prompt

    @property
    def current_trade(self):
        """The current trade offer (10-tuple plus trader index), if any."""
        return self._game.state.current_trade

    @property
    def acceptees(self):
        """Tuple of booleans, one per color, marking current trade acceptees."""
        return self._game.state.acceptees
