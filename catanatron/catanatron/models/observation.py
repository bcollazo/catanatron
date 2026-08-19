"""
A read-only, presentation-free snapshot of what exactly one color perceives.

The Observation is an agent's entire information world. It is pure data:
it carries no reference to a Game or State, so it is equally at home in
engine mode (projected from a full-truth game) and deployed/site mode
(built from external events).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from catanatron.models.enums import ActionPrompt, ActionRecord
from catanatron.models.inventory import Inventory
from catanatron.models.player import Color
from catanatron.models.public_state import PublicPlayer, PublicState
from catanatron.models.trade import PendingTrades, TradeOffer


@dataclass(frozen=True)
class Observation:
    """Read-only, presentation-free snapshot of what one color perceives.

    The core information world is ``public_state`` + ``pending_trades`` +
    ``inventory`` + ``current_prompt``. ``features`` is the flat, P0-relative RL
    vector surface — optional today for training-loop compatibility and unused
    by agents that reason structurally.

    Args:
        color (Color): the color whose perspective this snapshot represents.
        features (Optional[Dict[str, Any]]): imperfect-information feature
            snapshot, keyed relative to this color (P0). Optional; only needed
            by RL consumers.
        public_history (Tuple[ActionRecord, ...]): sanitized action log.
        current_prompt (Optional[ActionPrompt]): the prompt the current player
            must respond to.
        pending_trades (PendingTrades): trades currently on the table awaiting
            responses; empty when none. The engine allows one at a time today;
            the container future-proofs site-mode rules with competing offers.
        public_state (Optional[PublicState]): structured, pure-data snapshot
            of all public state, keyed absolutely (by node/edge id and Color).
            Built by the engine adapter; never holds a Game or State reference.
        inventory (Optional[Inventory]): the observer's own private hand —
            exact resources and dev cards plus hidden actual victory points.
            Computed for this color only.

    Properties:
        own (Optional[PublicPlayer]): this observer's own entry in
            ``public_state.players``, for the idiomatic ``obs.own.public_vps``
            style access.
    """

    color: Color
    features: Optional[Dict[str, Any]] = None
    public_history: Tuple[ActionRecord, ...] = ()
    current_prompt: Optional[ActionPrompt] = None
    pending_trades: PendingTrades = PendingTrades()
    public_state: Optional[PublicState] = None
    inventory: Optional[Inventory] = None

    @property
    def own(self) -> Optional[PublicPlayer]:
        """This observer's own public-player entry, or None without a public_state."""
        if self.public_state is None:
            return None
        return self.public_state.players[self.color]
