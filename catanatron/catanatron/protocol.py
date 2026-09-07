"""The wire protocol for bots that run outside this process.

The messages *are* the observer lifecycle (see :mod:`catanatron.observer`),
so there is one vocabulary rather than two:

    hello    once, when the bot process starts. The only exchange that both
             sides must complete before play begins; it pins the version and
             lets the bot ask to observe.
    before   once per game. Carries the full board.
    decide   the bot must answer with an action.
    step     an action somebody took. Only sent to bots that asked to observe.
    after    once per game, with the result.

Only ``hello`` and ``decide`` expect a reply, so a decision costs one round
trip and observing costs none.

Every message is one JSON object on one line. ``before`` carries the whole
redacted document; ``decide`` omits ``map``, which cannot change during a
game. Neither ever carries the development deck's order or the game's seed —
see :func:`catanatron.serialization.client_view`.
"""

from catanatron.serialization import action_to_json, client_view, state_to_json

#: Bumped when a change would break bots written against the old shape.
PROTOCOL_VERSION = 1

#: Sent in ``decide`` messages; static for the whole game, so ``before`` alone
#: carries it.
STATIC_KEYS = ("map",)


def hello_message():
    return {"type": "hello", "protocol_version": PROTOCOL_VERSION}


def before_message(game, color):
    return {
        "type": "before",
        "protocol_version": PROTOCOL_VERSION,
        "game_id": game.id,
        "color": color.value,
        "state": client_view(state_to_json(game)),
    }


def decide_message(game, color):
    view = client_view(state_to_json(game))
    return {
        "type": "decide",
        "game_id": game.id,
        "color": color.value,
        "state": {k: v for k, v in view.items() if k not in STATIC_KEYS},
        "playable_actions": [action_to_json(a) for a in game.playable_actions],
    }


def step_message(game, action):
    return {"type": "step", "game_id": game.id, "action": action_to_json(action)}


def after_message(game):
    winner = game.winning_color()
    return {
        "type": "after",
        "game_id": game.id,
        "winning_color": winner.value if winner else None,
    }


class ProtocolError(RuntimeError):
    """The bot said something the protocol does not allow."""


def parse_hello_reply(payload):
    """Validate a bot's answer to ``hello``.

    Returns ``(name, wants_step)``.
    """
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected a JSON object, got {payload!r}")
    version = payload.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"bot speaks protocol_version {version!r}, this catanatron speaks "
            f"{PROTOCOL_VERSION}"
        )
    return payload.get("name"), bool(payload.get("observe", False))


def parse_decide_reply(payload, playable_actions):
    """Turn a bot's answer to ``decide`` into one of ``playable_actions``."""
    if not isinstance(payload, dict) or "action" not in payload:
        raise ProtocolError(f"expected {{'action': [...]}}, got {payload!r}")

    wanted = payload["action"]
    if not isinstance(wanted, list) or len(wanted) != 3:
        raise ProtocolError(f"expected [color, action_type, value], got {wanted!r}")

    # Match against what is legal rather than trusting the bot's encoding, so
    # a value that merely looks right cannot slip through.
    for action in playable_actions:
        if action_to_json(action) == wanted:
            return action
    raise ProtocolError(f"{wanted!r} is not one of the playable actions")
