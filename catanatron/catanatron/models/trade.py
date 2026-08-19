"""
Typed, pure-data snapshot of a pending trade offer.

The engine stores trades as raw positional tuples on ``State`` (``current_trade``
plus a parallel ``acceptees`` tuple), which is hostile to an agent that wants to
reason about *what* is being traded and with whom. This object re-keys that info
by resource and color so the ObservationAgent sees the offer structurally.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Union

from catanatron.models.enums import FastResource
from catanatron.models.player import Color


@dataclass(frozen=True)
class TradeOffer:
    """One offer that is currently on the table, awaiting responses.

    Only public facts appear: a trade is fully public in Catan.
    """

    offerer: Color
    """Color that made the offer."""
    offered: Dict[FastResource, int]
    """Resource -> count the offerer gives up."""
    asking: Dict[FastResource, int]
    """Resource -> count the offerer wants in return."""
    acceptees: Dict[Color, bool]
    """Color -> whether that color has accepted; the offerer is always False."""


@dataclass(frozen=True)
class PendingTrades:
    """Immutable container for the trades currently on the table.

    Wrapping the underlying collection in an object makes the snapshot's
    read-only nature explicit and lets the internal storage change (tuple,
    list, tree) without touching consumers. Behaves like a tuple for iteration
    and indexing.
    """

    offers: Tuple[TradeOffer, ...] = ()

    def __iter__(self) -> Iterator[TradeOffer]:
        return iter(self.offers)

    def __len__(self) -> int:
        return len(self.offers)

    def __getitem__(self, index) -> Union[TradeOffer, Tuple[TradeOffer, ...]]:
        return self.offers[index]

    @property
    def is_active(self) -> bool:
        """True when at least one offer is on the table."""
        return bool(self.offers)

    @property
    def single(self) -> Optional[TradeOffer]:
        """The lone offer in engine mode (today at most one fits), else None."""
        if len(self.offers) != 1:
            return None
        return self.offers[0]
