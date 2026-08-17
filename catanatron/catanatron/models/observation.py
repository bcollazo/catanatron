"""
A read-only, presentation-free snapshot of what exactly one color perceives.

The Observation is an agent's entire information world. It is pure data:
it carries no reference to a Game or State, so it is equally at home in
engine mode (projected from a full-truth game) and deployed/site mode
(built from external events).
"""


class Observation:
    """Read-only, presentation-free snapshot of what one color perceives.

    Args:
        color (Color): the color whose perspective this snapshot represents.
        features (Dict[str, Any]): imperfect-information feature snapshot,
            keyed relative to this color (P0).
        public_history (List[ActionRecord]): sanitized action log.
        current_prompt (ActionPrompt): the prompt the current player must
            respond to.
        current_trade (Tuple): the current trade offer, if any.
        acceptees (Tuple[bool]): current trade acceptees, one per color.
        public_state (Dict[str, Any]): structured, pure-data snapshot of all
            public state, keyed absolutely (by node/edge id and Color). Built
            by the engine adapter; never holds a Game or State reference.
    """

    def __init__(
        self,
        color,
        features,
        public_history,
        current_prompt,
        current_trade,
        acceptees,
        public_state=None,
    ):
        self.color = color
        self.features = features
        self.public_history = public_history
        self.current_prompt = current_prompt
        self.current_trade = current_trade
        self.acceptees = acceptees
        self.public_state = public_state or {}
