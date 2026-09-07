"""A player that is a separate program, spoken to over its stdin/stdout.

Catanatron writes one JSON message per line to the process and reads one line
back for the messages that expect a reply. The process is started on the first
game and kept for the life of the run, so a bot pays startup once rather than
per game.

A bot that misbehaves must not take the simulation down with it, so a timeout
or a bad reply costs that one decision: catanatron plays a random legal action
and carries on, having said so once.
"""

import json
import logging
import queue
import random
import shlex
import subprocess
import threading
from typing import Optional

from catanatron.params import BaseParams

from catanatron.models.player import Player
from catanatron.protocol import (
    ProtocolError,
    after_message,
    before_message,
    decide_message,
    hello_message,
    parse_decide_reply,
    parse_hello_reply,
    step_message,
)

logger = logging.getLogger(__name__)


class StdioPlayer(Player):
    """Base for bots run as a subprocess.

    Subclasses set ``COMMAND``; ``--bot NAME=exec:...`` builds one for you.
    """

    #: argv of the program to run. Set by the subclass, never by a request.
    COMMAND: Optional[list] = None

    #: Consecutive failures after which the bot is dropped for the rest of the
    #: run. A bot that times out on every decision would otherwise stretch one
    #: game out by timeout_ms per turn.
    MAX_CONSECUTIVE_FAILURES = 3

    class Params(BaseParams):
        #: How long one decision may take before it is forfeited.
        timeout_ms: int = 1000

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self._process = None
        self._replies = None
        self._reader = None
        self._wants_step = False
        self._complained = False
        self._failures = 0
        self._given_up = False

    # ===== process =====
    def _start(self):
        if self._process is not None:
            return
        if not self.COMMAND:
            raise ProtocolError(f"{type(self).__name__} has no COMMAND")
        try:
            self._process = subprocess.Popen(
                self.COMMAND,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # let the bot's own logging reach the terminal
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise ProtocolError(f"cannot run {' '.join(self.COMMAND)}: {error}")

        self._replies = queue.Queue()
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()

        # A handshake that fails is fatal, but must not leave the child behind
        # for the interpreter to wait on at exit.
        try:
            self._send(hello_message())
            reply = self._receive(self.params.timeout_ms)
            if reply is None:
                raise ProtocolError(
                    f"{' '.join(self.COMMAND)} did not answer the hello "
                    f"handshake within {self.params.timeout_ms}ms"
                )
            name, self._wants_step = parse_hello_reply(reply)
        except ProtocolError:
            self.close()
            raise
        logger.info("connected to bot %s", name or self.COMMAND[0])

    def _read_lines(self):
        for line in self._process.stdout:
            self._replies.put(line)
        self._replies.put(None)  # process closed its output

    def _send(self, message):
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as error:
            raise ProtocolError(f"bot stopped listening: {error}")

    def _receive(self, timeout_ms):
        try:
            line = self._replies.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None
        if line is None:
            raise ProtocolError("bot closed its output")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"bot wrote a line that is not JSON: {error}")

    def close(self):
        if self._process is None:
            return
        try:
            self._process.stdin.close()
            self._process.terminate()
            self._process.wait(timeout=2)
        except Exception:  # a bot that will not die is not worth waiting on
            self._process.kill()
        finally:
            self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ===== lifecycle =====
    def before(self, game):
        # A failed handshake is a configuration problem, not a bad turn, so it
        # is deliberately not swallowed here.
        if self._given_up:
            return
        self._start()
        self._tell(before_message(game, self.color))

    def step(self, game_before_action, action):
        # Only bots that asked for it are observed; the engine still calls
        # this because the class overrides it, so filter here.
        if self._wants_step:
            self._tell(step_message(game_before_action, action))

    def after(self, game):
        self._tell(after_message(game))

    def _tell(self, message):
        """Send something that expects no reply. A dead bot loses its turns,
        it does not end the run."""
        if self._process is None or self._given_up:
            return
        try:
            self._send(message)
        except ProtocolError as error:
            self._note_failure(str(error))

    def decide(self, game, playable_actions):
        playable_actions = list(playable_actions)
        if len(playable_actions) == 1:
            return playable_actions[0]
        if self._given_up:
            return random.choice(playable_actions)
        try:
            self._start()
            self._send(decide_message(game, self.color))
            reply = self._receive(self.params.timeout_ms)
            if reply is None:
                return self._forfeit(
                    playable_actions,
                    f"took longer than {self.params.timeout_ms}ms",
                )
            action = parse_decide_reply(reply, playable_actions)
        except ProtocolError as error:
            return self._forfeit(playable_actions, str(error))
        self._failures = 0
        return action

    def _forfeit(self, playable_actions, reason):
        """One bad decision costs one turn, not the whole simulation."""
        self._note_failure(reason)
        return random.choice(playable_actions)

    def _note_failure(self, reason):
        if not self._complained:
            logger.warning(
                "%s: %s; playing random legal actions instead "
                "(further occurrences not reported)",
                type(self).__name__,
                reason,
            )
            self._complained = True
        self._failures += 1
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES and not self._given_up:
            logger.warning(
                "%s: gave up after %d consecutive failures; playing randomly "
                "for the rest of the run",
                type(self).__name__,
                self._failures,
            )
            self._given_up = True
            self.close()


def build_stdio_player_class(name, command):
    """Make a StdioPlayer subclass bound to one command.

    The command is a class attribute rather than a Param so that it is not
    settable through the web API's published parameter schema.
    """
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if not argv:
        raise ProtocolError("empty exec command")
    return type(name, (StdioPlayer,), {"COMMAND": argv, "__doc__": f"Runs {argv[0]}."})
