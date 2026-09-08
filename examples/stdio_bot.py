"""A reference bot spoken to over stdin/stdout.

Run it with::

    catanatron-play --bot MYBOT=exec:"python examples/stdio_bot.py" \
        --players=R,R,R,MYBOT --num=5

It happens to be Python, but nothing here imports catanatron: the protocol is
JSON on stdin and stdout, so the same shape works in any language.
"""

import json
import random
import sys

PROTOCOL_VERSION = 1


def decide(state, playable_actions):
    """Build a city when possible, else play at random."""
    for action in playable_actions:
        if action[1] == "BUILD_CITY":
            return action
    return random.choice(playable_actions)


def main():
    board = None  # the map arrives once, in `before`
    for line in sys.stdin:
        message = json.loads(line)
        kind = message["type"]

        if kind == "hello":
            reply = {
                "protocol_version": PROTOCOL_VERSION,
                "name": "city-rusher",
                "observe": False,  # set True to also receive `step` messages
            }
        elif kind == "before":
            board = message["state"]["map"]
            continue  # no reply expected
        elif kind == "decide":
            reply = {"action": decide(message["state"], message["playable_actions"])}
        elif kind in ("step", "after"):
            continue  # no reply expected
        else:
            continue  # unknown message types are ignored, not fatal

        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
