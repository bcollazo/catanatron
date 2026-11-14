"""Card counting: what each player can legitimately know about everyone's hand.

Nearly all card movement in Catan is public (roll payouts, build costs, bank
and player trades, Year of Plenty, Monopoly, and -- as in Catanatron's action
log -- discards on a 7). The one hidden movement is a **steal** (robber /
Knight): only the thief and the victim see which resource moved. Everyone
else sees a card of unknown type go from one hand to the other.

So "the" hand belief is observer-specific: each player gets its own view.
Following the discussion in issue #268, the game itself maintains these views
(so every agent has access to them, with no per-agent bookkeeping) and each
view is built *only* from information that observer saw.

Model
-----
For every observer we keep a probability distribution over *worlds*, where a
world is one full assignment of resource hands to all players:
``world[player_index] == (wood, brick, sheep, wheat, ore)``.

* A public event is applied to every world. Worlds in which it is impossible
  (someone would end up with a negative card count) are dropped and the rest
  renormalised. This is what resolves the linked uncertainty described in the
  issue: if A is later seen spending a clay, any world where A never held the
  clay that was stolen from them is eliminated, which retroactively fixes the
  thief's view of the stolen card too.
* A steal the observer is not part of branches each world once per resource
  the victim could have lost, weighted by the (known) chance the game picks
  it: ``hand[r] / sum(hand)``.
* A steal the observer is part of is certain, so it just updates every world.

Worlds that end up identical are merged, so the set stays small in practice;
``max_worlds`` caps it (keeping the most likely ones) as a safety valve.

Discards are treated as public. Making them hidden is a small change, but it
multiplies the number of worlds by up to 5 per discarded card, which made the
exact belief blow up (tens of thousands of worlds in a single game).

Not covered: development cards held in hand (their identity is hidden from
other players, and the issue is about resource cards).
"""

from typing import Dict, List, Sequence, Tuple

from catanatron.models.enums import RESOURCES

Hand = Tuple[int, int, int, int, int]
World = Tuple[Hand, ...]
NUM_RESOURCES = len(RESOURCES)
EMPTY_HAND: Hand = (0,) * NUM_RESOURCES

DEFAULT_MAX_WORLDS = 512


class CardCounter:
    """Per-observer beliefs about every player's resource hand.

    Mutators take seat indices (``state.color_to_index``) and are called by
    ``apply_action``. Queries take the observer's and target's ``Color``.
    """

    def __init__(self, colors: Sequence, max_worlds: int = DEFAULT_MAX_WORLDS):
        self.colors = tuple(colors)
        self.color_to_index = {c: i for i, c in enumerate(self.colors)}
        self.max_worlds = max_worlds
        initial: World = tuple(EMPTY_HAND for _ in self.colors)
        # One independent belief per observer. The dicts are never mutated in
        # place (updates build new ones) so copy() can share them.
        self._beliefs: List[Dict[World, float]] = [
            {initial: 1.0} for _ in self.colors
        ]
        #: True if a (truncated) belief ever contradicted a public event and
        #: had to be patched up. Never happens with an untruncated belief.
        self.degraded = False

    def copy(self) -> "CardCounter":
        other = CardCounter.__new__(CardCounter)
        other.colors = self.colors
        other.color_to_index = self.color_to_index
        other.max_worlds = self.max_worlds
        other._beliefs = list(self._beliefs)
        other.degraded = self.degraded
        return other

    # ===== Events (called from apply_action) =====
    def change(self, index: int, freqdeck: Sequence[int], sign: int = 1):
        """A public change to one player's hand (+1 gain / -1 spend)."""
        if not any(freqdeck):
            return
        delta = tuple(sign * n for n in freqdeck)
        for o in range(len(self._beliefs)):
            self._beliefs[o] = self._apply_known(
                self._beliefs[o], ((index, delta),)
            )

    def gain(self, index: int, freqdeck: Sequence[int]):
        self.change(index, freqdeck, 1)

    def spend(self, index: int, freqdeck: Sequence[int]):
        self.change(index, freqdeck, -1)

    def steal(self, thief: int, victim: int, resource: str):
        """``thief`` took one ``resource`` card from ``victim``'s hand."""
        r = RESOURCES.index(resource)
        for o in range(len(self._beliefs)):
            if o == thief or o == victim:
                unit = tuple(1 if i == r else 0 for i in range(NUM_RESOURCES))
                neg = tuple(-n for n in unit)
                self._beliefs[o] = self._apply_known(
                    self._beliefs[o], ((victim, neg), (thief, unit))
                )
            else:
                self._beliefs[o] = self._branch_steal(
                    self._beliefs[o], victim, thief
                )

    # ===== Belief updates =====
    def _apply_known(self, belief, changes) -> Dict[World, float]:
        new: Dict[World, float] = {}
        for world, weight in belief.items():
            hands = list(world)
            ok = True
            for index, delta in changes:
                hand = tuple(h + d for h, d in zip(hands[index], delta))
                if min(hand) < 0:
                    ok = False
                    break
                hands[index] = hand
            if ok:
                key = tuple(hands)
                new[key] = new.get(key, 0.0) + weight
        if not new:
            return self._patch(belief, changes)
        return self._normalize(new)

    def _branch_steal(self, belief, victim: int, thief: int):
        """A random card moves from ``victim`` to ``thief`` and the observer
        did not see which one."""
        new: Dict[World, float] = {}
        for world, weight in belief.items():
            hand = world[victim]
            total = sum(hand)
            if total == 0:
                continue
            for r in range(NUM_RESOURCES):
                n = hand[r]
                if n == 0:
                    continue
                hands = list(world)
                hands[victim] = hand[:r] + (n - 1,) + hand[r + 1 :]
                t = hands[thief]
                hands[thief] = t[:r] + (t[r] + 1,) + t[r + 1 :]
                key = tuple(hands)
                new[key] = new.get(key, 0.0) + weight * n / total
        if not new:
            # Only reachable after truncation dropped the true world.
            self.degraded = True
            return belief
        return self._normalize(new)

    def _patch(self, belief, changes) -> Dict[World, float]:
        """Every world contradicts a public event: the truncated belief lost
        the true world. Clamp at zero so play can continue."""
        self.degraded = True
        new: Dict[World, float] = {}
        for world, weight in belief.items():
            hands = list(world)
            for index, delta in changes:
                hands[index] = tuple(max(0, h + d) for h, d in zip(hands[index], delta))
            key = tuple(hands)
            new[key] = new.get(key, 0.0) + weight
        return self._normalize(new)

    def _normalize(self, belief: Dict[World, float]) -> Dict[World, float]:
        if len(belief) > self.max_worlds:
            top = sorted(belief.items(), key=lambda kv: kv[1], reverse=True)
            belief = dict(top[: self.max_worlds])
        total = sum(belief.values())
        return {w: p / total for w, p in belief.items()}

    # ===== Queries =====
    def _belief(self, observer, target):
        return self._beliefs[self.color_to_index[observer]], self.color_to_index[target]

    def num_cards(self, observer, target) -> int:
        """Hand size. Always exact: hand sizes are public information."""
        belief, t = self._belief(observer, target)
        world = next(iter(belief))
        return sum(world[t])

    def hand_distribution(self, observer, target) -> Dict[Hand, float]:
        """Probability of each possible hand (freqdeck) ``target`` holds,
        from ``observer``'s point of view."""
        belief, t = self._belief(observer, target)
        dist: Dict[Hand, float] = {}
        for world, p in belief.items():
            dist[world[t]] = dist.get(world[t], 0.0) + p
        return dist

    def known_hand(self, observer, target) -> Hand:
        """Cards ``observer`` is certain ``target`` holds (a lower bound of
        every possible hand). The rest are :meth:`num_unknown_cards`."""
        belief, t = self._belief(observer, target)
        hands = [world[t] for world in belief]
        return tuple(min(h[r] for h in hands) for r in range(NUM_RESOURCES))

    def num_unknown_cards(self, observer, target) -> int:
        """Cards in ``target``'s hand whose type ``observer`` can't pin down."""
        return self.num_cards(observer, target) - sum(self.known_hand(observer, target))

    def expected_hand(self, observer, target) -> Tuple[float, ...]:
        """Expected count of each resource in ``target``'s hand."""
        belief, t = self._belief(observer, target)
        return tuple(
            sum(p * world[t][r] for world, p in belief.items())
            for r in range(NUM_RESOURCES)
        )

    def steal_probabilities(self, observer, target) -> Tuple[float, ...]:
        """Chance that robbing ``target`` yields each resource (all zeros if
        their hand is empty). Replaces assuming a uniform or perfectly-known
        hand when searching over steal outcomes."""
        belief, t = self._belief(observer, target)
        probs = [0.0] * NUM_RESOURCES
        for world, p in belief.items():
            hand = world[t]
            total = sum(hand)
            if total == 0:
                continue
            for r in range(NUM_RESOURCES):
                probs[r] += p * hand[r] / total
        return tuple(probs)
