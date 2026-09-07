---
icon: pen-to-square
---

# Custom Bots

A bot is a `Player` subclass with a `decide` method. Put it in a file:

```python
# mybot.py
import random
from catanatron import Player, ActionType


class MyBot(Player):
    """Builds cities whenever it can, else plays at random."""

    def decide(self, game, playable_actions):
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_CITY:
                return action
        return random.choice(playable_actions)
```

Point `--bot` at it. The bot is named after its class:

```bash
catanatron-play --bot ./mybot.py --players=R,R,R,MyBot --num=100
```

`--bot` is repeatable, takes an optional `NAME=` prefix, and the file can live
anywhere. If a file defines more than one player, name the one you want after
a `#`:

```bash
catanatron-play --bot RUSH=~/bots/mine.py#CityRusher --players=R,RUSH
```

## Parameters

Declare what your bot can tune as a nested `Params` dataclass:

```python
from catanatron import BaseParams


class MyBot(Player):
    class Params(BaseParams):
        aggression: int = 1
```

Set them on the command line, positionally or by name, in declaration order:

```bash
catanatron-play --bot ./mybot.py --players=R,MyBot:3
catanatron-play --bot ./mybot.py --players=R,MyBot:aggression=3
```

Values are parsed into the types you declared, and anything that does not fit
is an error rather than a surprise. The same declaration is what
`--help-players` lists and what `GET /api/players` publishes, so a UI can
render inputs for it.

## Watching the game

`decide` is only called on your turn. To see what everyone else did, override
the observer hooks — all optional:

```python
class CardCounter(Player):
    def before(self, game):
        """Once per game, before any action."""
        self.knights = 0

    def step(self, game_before_action, action):
        """Every action taken by anyone, including your own."""
        if action.action_type == ActionType.PLAY_KNIGHT_CARD:
            self.knights += 1

    def after(self, game):
        """Once the game is over."""
```

A bot that does not override `step` is skipped when actions are applied, so
watching costs nothing if you do not use it.

## Bots in other languages

A bot can be any program. Catanatron runs it and exchanges one JSON message
per line over its stdin and stdout:

```bash
catanatron-play --bot MYBOT=exec:"python examples/stdio_bot.py" \
    --players=R,R,R,MYBOT
```

| catanatron sends | you reply |
|---|---|
| `{"type": "hello", "protocol_version": 1}` | `{"protocol_version": 1, "name": "...", "observe": false}` |
| `{"type": "before", "state": {...}}` | nothing |
| `{"type": "decide", "state": {...}, "playable_actions": [...]}` | `{"action": [...]}` |
| `{"type": "step", "action": [...]}` | nothing (only if you set `observe`) |
| `{"type": "after", "winning_color": "RED"}` | nothing |

Answer `decide` with one of the `playable_actions` you were given, verbatim.
`before` carries the whole board; `decide` leaves out `map`, which cannot
change during a game. Set the deadline with `--players=R,MYBOT:timeout_ms=500`.

`state` is what a person in your seat would see, never more. Your own hand is
itemized; an opponent's arrives as `P1_NUM_RESOURCES_IN_HAND` and
`P1_NUM_DEVELOPMENT_CARDS_IN_HAND`, because across the table you can count
someone's cards but not read them. The card an opponent drew and the resource
their robber stole are missing from the history for the same reason, and the
development deck is a composition rather than an order. There is no flag for
this: a bot on the wire only ever gets this view.

A bot that answers too slowly, names an illegal action, writes something that
is not JSON, or dies forfeits that turn, and is dropped after three failures
in a row. A failed handshake or a program that cannot be run stops the run.

See `examples/stdio_bot.py` for a complete bot in about forty lines.
