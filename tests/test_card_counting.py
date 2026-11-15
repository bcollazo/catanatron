import pytest

from catanatron.game import Game
from catanatron.models.enums import (
    RESOURCES,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.card_counting import CardCounter
from catanatron.state_functions import player_key


def true_hand(state, color):
    key = player_key(state, color)
    return tuple(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


def make_counter():
    colors = (Color.RED, Color.BLUE, Color.WHITE)
    return CardCounter(colors), colors


def test_disabled_by_default():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    assert game.state.card_counter is None
    assert game.copy().state.card_counter is None


def test_public_gain_and_spend_are_known_to_everyone():
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [2, 1, 0, 0, 0])
    counter.spend(0, [1, 1, 0, 0, 0])
    for observer in (red, blue, white):
        assert counter.known_hand(observer, red) == (1, 0, 0, 0, 0)
        assert counter.num_unknown_cards(observer, red) == 0


def test_steal_visible_only_to_thief_and_victim():
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [1, 1, 0, 0, 0])  # RED holds wood + brick
    counter.steal(1, 0, "BRICK")  # BLUE steals brick from RED

    # BLUE (thief) and RED (victim) know exactly.
    for observer in (red, blue):
        assert counter.known_hand(observer, blue) == (0, 1, 0, 0, 0)
        assert counter.known_hand(observer, red) == (1, 0, 0, 0, 0)
    # WHITE only knows a card moved: 50/50 wood or brick.
    assert counter.num_cards(white, blue) == 1
    assert counter.num_unknown_cards(white, blue) == 1
    assert counter.steal_probabilities(white, blue) == pytest.approx((0.5, 0.5, 0, 0, 0))
    assert counter.steal_probabilities(white, red) == pytest.approx((0.5, 0.5, 0, 0, 0))


def test_later_spend_resolves_linked_uncertainty():
    """The scenario from the issue: a spend reveals what an earlier,
    hidden steal must have been, for both thief and victim."""
    counter, (red, blue, white) = make_counter()
    counter.gain(0, [1, 1, 0, 0, 0])
    counter.steal(1, 0, "BRICK")  # WHITE can't tell which
    # RED is then seen paying a wood: so RED had a wood left, so the
    # stolen card must have been the brick.
    counter.spend(0, [1, 0, 0, 0, 0])
    assert counter.known_hand(white, blue) == (0, 1, 0, 0, 0)
    assert counter.num_unknown_cards(white, blue) == 0
    assert counter.num_cards(white, red) == 0


def test_copy_is_independent():
    counter, (red, blue, white) = make_counter()
    other = counter.copy()
    other.gain(0, [1, 0, 0, 0, 0])
    assert counter.num_cards(red, red) == 0
    assert other.num_cards(red, red) == 1


@pytest.mark.parametrize("seed", range(15))
def test_beliefs_always_contain_the_truth(seed):
    """Play whole games and check, after every action, that every
    observer's belief (a) includes the real hands and (b) never claims
    certainty about a card the target doesn't have. Own hand is exact."""
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)]
    game = Game(players, seed=seed, card_counting=True)
    counter = game.state.card_counter
    checked = 0
    while game.winning_color() is None and game.state.num_turns < 300:
        game.play_tick()
        for observer in game.state.colors:
            for target in game.state.colors:
                truth = true_hand(game.state, target)
                dist = counter.hand_distribution(observer, target)
                assert truth in dist, (observer, target)
                assert counter.num_cards(observer, target) == sum(truth)
                if observer == target:
                    assert dist == {truth: pytest.approx(1.0)}
        checked += 1
    assert not counter.degraded
    assert checked > 50


def test_state_copy_keeps_counting_independently():
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    game = Game(players, seed=1, card_counting=True)
    for _ in range(60):
        game.play_tick()
    copy = game.copy()
    for _ in range(20):
        copy.play_tick()
    for observer in game.state.colors:
        for target in game.state.colors:
            assert true_hand(game.state, target) in game.state.card_counter.hand_distribution(
                observer, target
            )
            assert true_hand(copy.state, target) in copy.state.card_counter.hand_distribution(
                observer, target
            )


def test_discards_are_public():
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE)]
    game = Game(players, seed=3, card_counting=True)
    counter = game.state.card_counter
    red = game.state.colors[0]
    counter.gain(0, [3, 0, 0, 0, 1])
    counter.spend(0, [0, 0, 0, 0, 1])
    assert counter.known_hand(game.state.colors[1], red)[4] == 0


# ===== Exact distribution check against an independent brute force =====
import random


def brute_force_distribution(num_players, events, observer, target):
    """Enumerates every history of hidden-steal outcomes the observer could
    have missed, one path at a time (no merging, no renormalising mid-way).
    Paths a later public event rules out are dropped; the rest are weighted by
    the product of steal chances and normalised once at the end."""
    result = {}

    def walk(k, hands, prob):
        if k == len(events):
            hand = hands[target]
            result[hand] = result.get(hand, 0.0) + prob
            return
        kind, *args = events[k]
        if kind in ("gain", "spend"):
            i, freqdeck = args
            sign = 1 if kind == "gain" else -1
            new = tuple(h + sign * n for h, n in zip(hands[i], freqdeck))
            if min(new) < 0:
                return  # impossible history
            walk(k + 1, hands[:i] + (new,) + hands[i + 1 :], prob)
            return
        thief, victim, r = args
        outcomes = range(5)
        if observer in (thief, victim):
            outcomes = [r]  # observer saw which card moved
        total = sum(hands[victim])
        for x in outcomes:
            if hands[victim][x] == 0:
                continue
            chance = 1.0 if observer in (thief, victim) else hands[victim][x] / total
            v = list(hands[victim])
            v[x] -= 1
            t = list(hands[thief])
            t[x] += 1
            new = list(hands)
            new[victim], new[thief] = tuple(v), tuple(t)
            walk(k + 1, tuple(new), prob * chance)

    walk(0, tuple((0,) * 5 for _ in range(num_players)), 1.0)
    total = sum(result.values())
    return {hand: p / total for hand, p in result.items()}


def random_script(rng, num_players, length):
    """Random but legal sequence of public gains/spends and hidden steals,
    generated against the true hands so every event is possible."""
    hands = [[0] * 5 for _ in range(num_players)]
    events = []
    while len(events) < length:
        kind = rng.choice(["gain", "gain", "steal", "steal", "spend"])
        if kind == "gain":
            i = rng.randrange(num_players)
            fd = [0] * 5
            for _ in range(rng.randint(1, 3)):
                fd[rng.randrange(5)] += 1
            hands[i] = [h + n for h, n in zip(hands[i], fd)]
            events.append(("gain", i, fd))
        elif kind == "spend":
            i = rng.randrange(num_players)
            cards = [r for r in range(5) for _ in range(hands[i][r])]
            if not cards:
                continue
            r = rng.choice(cards)
            fd = [0] * 5
            fd[r] = 1
            hands[i][r] -= 1
            events.append(("spend", i, fd))
        else:
            thief, victim = rng.sample(range(num_players), 2)
            cards = [r for r in range(5) for _ in range(hands[victim][r])]
            if not cards:
                continue
            r = rng.choice(cards)
            hands[victim][r] -= 1
            hands[thief][r] += 1
            events.append(("steal", thief, victim, r))
    return events


@pytest.mark.parametrize("seed", range(25))
def test_belief_distribution_matches_brute_force(seed):
    rng = random.Random(seed)
    num_players = rng.choice([2, 3, 4])
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)[:num_players]
    events = random_script(rng, num_players, length=14)

    counter = CardCounter(colors)
    for event in events:
        kind, *args = event
        if kind == "gain":
            counter.gain(*args)
        elif kind == "spend":
            counter.spend(*args)
        else:
            thief, victim, r = args
            counter.steal(thief, victim, RESOURCES[r])

    for o in range(num_players):
        for t in range(num_players):
            expected = brute_force_distribution(num_players, events, o, t)
            actual = counter.hand_distribution(colors[o], colors[t])
            assert sum(actual.values()) == pytest.approx(1.0)
            assert set(actual) == set(expected), (o, t)
            for hand, p in expected.items():
                assert actual[hand] == pytest.approx(p), (o, t, hand)
