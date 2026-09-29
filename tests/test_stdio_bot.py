"""Bots that run as a separate program.

Each test writes a small bot script and plays real games against it, so the
subprocess, the protocol and the failure handling are all exercised together.
"""

import random
import sys
import textwrap

import pytest

from catanatron import Color, Game
from catanatron.models.player import RandomPlayer
from catanatron.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decide_message,
    parse_decide_reply,
    parse_hello_reply,
)
from catanatron.players.stdio import build_stdio_player_class
from catanatron.registry import PlayerRegistry, SpecError, describe
from catanatron.serialization import action_to_json

HEADER = (
    """
import json, sys, time
def reply(payload):
    sys.stdout.write(json.dumps(payload) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "hello":
        reply({"protocol_version": %d, "name": "test"})
        continue
"""
    % PROTOCOL_VERSION
)

BOTS = {
    "good": HEADER
    + """
    if m["type"] == "decide":
        reply({"action": m["playable_actions"][0]})
""",
    "slow": HEADER
    + """
    if m["type"] == "decide":
        time.sleep(5)
""",
    "illegal": HEADER
    + """
    if m["type"] == "decide":
        reply({"action": ["RED", "BUILD_CITY", 999]})
""",
    "garbage": HEADER
    + """
    if m["type"] == "decide":
        sys.stdout.write("not json\\n"); sys.stdout.flush()
""",
    "dies": HEADER
    + """
    if m["type"] == "decide":
        sys.exit(1)
""",
    "future": """
import json, sys
for line in sys.stdin:
    if json.loads(line)["type"] == "hello":
        sys.stdout.write(json.dumps({"protocol_version": 99}) + "\\n"); sys.stdout.flush()
""",
    "mute": """
import sys, time
for line in sys.stdin:
    time.sleep(30)
""",
    "observer": HEADER
    + """
    if m["type"] == "decide":
        reply({"action": m["playable_actions"][0]})
""",
}


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.fixture
def bot_class(tmp_path):
    """Build a StdioPlayer subclass running one of the scripts above."""

    def make(which, name="TESTBOT"):
        path = tmp_path / f"{which}.py"
        path.write_text(textwrap.dedent(BOTS[which]))
        return build_stdio_player_class(name, f"{sys.executable} {path}")

    return make


def play(bot, seed=4, ticks=None):
    random.seed(seed)
    game = Game([bot, RandomPlayer(Color.BLUE)])
    if ticks is None:
        return game.play()
    for _ in range(ticks):
        game.play_tick()
    return game


# ===== the happy path =====
def test_plays_a_whole_game(bot_class):
    bot = bot_class("good")(Color.RED)
    game = Game([bot, RandomPlayer(Color.BLUE)])
    random.seed(4)
    game.play()
    assert len(game.state.action_records) > 20


def test_decisions_come_from_the_subprocess(bot_class):
    """The scripted bot always takes the first playable action."""
    bot = bot_class("good")(Color.RED)
    game = Game([bot, RandomPlayer(Color.BLUE)])
    for _ in range(6):
        expected = game.playable_actions[0]
        is_bots_turn = game.state.current_color() == Color.RED
        record = game.play_tick()
        if is_bots_turn and len(game.state.action_records) and expected is not None:
            assert record.action == expected


# ===== misbehaviour costs the bot, not the run =====
@pytest.mark.parametrize("which", ["slow", "illegal", "garbage", "dies"])
def test_a_broken_bot_does_not_end_the_run(bot_class, which):
    bot = bot_class(which)(Color.RED, build_params_timeout(200))
    game = Game([bot, RandomPlayer(Color.BLUE)])
    random.seed(4)
    assert game.play() in (Color.RED, Color.BLUE, None)


@pytest.mark.parametrize("which", ["slow", "illegal", "garbage", "dies"])
def test_a_broken_bot_is_dropped_after_repeated_failures(bot_class, which):
    """Otherwise a bot that always times out stretches a game out by
    timeout_ms per turn."""
    cls = bot_class(which)
    bot = cls(Color.RED, build_params_timeout(200))
    game = Game([bot, RandomPlayer(Color.BLUE)])
    random.seed(4)
    game.play()
    assert bot._given_up is True
    assert bot._process is None


def build_params_timeout(ms):
    from catanatron.players.stdio import StdioPlayer

    return StdioPlayer.Params(timeout_ms=ms)


# ===== configuration problems are fatal, and clean =====
def test_version_mismatch_is_fatal(bot_class):
    bot = bot_class("future")(Color.RED, build_params_timeout(2000))
    with pytest.raises(ProtocolError, match="protocol_version 99"):
        Game([bot, RandomPlayer(Color.BLUE)])


def test_a_bot_that_never_handshakes_is_fatal(bot_class):
    bot = bot_class("mute")(Color.RED, build_params_timeout(200))
    with pytest.raises(ProtocolError, match="did not answer the hello handshake"):
        Game([bot, RandomPlayer(Color.BLUE)])


def test_a_missing_program_is_reported_clearly():
    cls = build_stdio_player_class("GHOST", "./definitely-not-here")
    with pytest.raises(ProtocolError, match="cannot run"):
        Game([cls(Color.RED), RandomPlayer(Color.BLUE)])


def test_empty_command_is_rejected():
    with pytest.raises(ProtocolError, match="empty exec command"):
        build_stdio_player_class("NOPE", "   ")


# ===== protocol shape =====
def test_decide_message_omits_the_static_map():
    random.seed(2)
    game = Game([RandomPlayer(color) for color in Color])
    for _ in range(20):
        game.play_tick()
    message = decide_message(game, Color.RED)
    assert "map" not in message["state"]
    assert message["playable_actions"]


def test_decide_message_hides_the_deck_order():
    random.seed(2)
    game = Game([RandomPlayer(color) for color in Color])
    message = decide_message(game, Color.RED)
    assert isinstance(message["state"]["development_listdeck"], dict)
    assert "seed" not in message["state"]["game"]


def test_decide_message_hides_the_other_hands():
    """A bot on the wire gets the honest view, always. There is no flag."""
    random.seed(2)
    game = Game([RandomPlayer(color) for color in Color])
    for _ in range(40):
        game.play_tick()
    state = decide_message(game, Color.RED)["state"]

    mine = state["colors"].index(Color.RED.value)
    theirs = (mine + 1) % len(state["colors"])
    assert f"P{mine}_WOOD_IN_HAND" in state["player_state"]
    assert f"P{theirs}_WOOD_IN_HAND" not in state["player_state"]
    assert f"P{theirs}_NUM_RESOURCES_IN_HAND" in state["player_state"]


def test_reply_must_name_a_playable_action():
    random.seed(2)
    game = Game([RandomPlayer(color) for color in Color])
    legal = game.playable_actions[0]
    assert (
        parse_decide_reply({"action": action_to_json(legal)}, game.playable_actions)
        == legal
    )
    with pytest.raises(ProtocolError, match="not one of the playable actions"):
        parse_decide_reply(
            {"action": ["RED", "BUILD_CITY", 999]}, game.playable_actions
        )


@pytest.mark.parametrize("payload", [{}, {"action": "nope"}, {"action": [1, 2]}, "x"])
def test_malformed_replies_are_rejected(payload):
    with pytest.raises(ProtocolError):
        parse_decide_reply(payload, [])


def test_hello_reply_reports_whether_the_bot_observes():
    assert parse_hello_reply(
        {"protocol_version": PROTOCOL_VERSION, "name": "x", "observe": True}
    ) == ("x", True)
    assert parse_hello_reply({"protocol_version": PROTOCOL_VERSION})[1] is False


# ===== registry integration =====
def test_exec_source_registers_a_bot(tmp_path):
    registry = PlayerRegistry()
    path = tmp_path / "good.py"
    path.write_text(textwrap.dedent(BOTS["good"]))
    key = registry.register_source(f"exec:{sys.executable} {path}", name="RUSTY")
    assert key == "RUSTY"
    assert [p["name"] for p in describe(key, registry[key])["params"]] == ["timeout_ms"]


def test_exec_source_needs_a_name():
    registry = PlayerRegistry()
    with pytest.raises(SpecError, match="needs a name"):
        registry.register_source("exec:./bot")


def test_the_command_is_not_a_settable_param(tmp_path):
    """A published param could be re-pointed through the web API."""
    registry = PlayerRegistry()
    path = tmp_path / "good.py"
    path.write_text(textwrap.dedent(BOTS["good"]))
    key = registry.register_source(f"exec:{sys.executable} {path}", name="RUSTY")
    names = [p["name"] for p in describe(key, registry[key])["params"]]
    assert "COMMAND" not in names and "command" not in names


def test_http_sources_say_they_are_not_built_yet():
    registry = PlayerRegistry()
    with pytest.raises(SpecError, match="not supported yet"):
        registry.register_source("https://host/decide", name="X")
