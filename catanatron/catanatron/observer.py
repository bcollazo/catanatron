"""The lifecycle a game exposes to anything watching it.

Players and accumulators watch the same game; they differ only in that a
player also has to answer :meth:`~catanatron.models.player.Player.decide`.
Both therefore share one set of hooks, so there is a single lifecycle to
learn and a single vocabulary for the out-of-process bot protocol.

Lives in its own module because ``catanatron.game`` imports the player
classes, so the shared base cannot live in either of them.
"""


class GameObserver:
    """Hooks into a game's lifecycle. Every method is optional."""

    def __init__(self, *args, **kwargs):
        """Accept and ignore anything.

        ``play_batch`` constructs CLI accumulators as
        ``cls(players=..., game_config=...)``, so a subclass that does not care
        about either must not have to declare them.
        """

    def before(self, game):
        """Called once, before any action is taken.

        The board is already decided, so this is where to look at the map or
        reset whatever state you carry between games.
        """

    def step(self, game_before_action, action):
        """Called for every action taken by anyone, including your own.

        ``game_before_action`` is the game as it was *before* the action was
        applied. Overriding this is what makes a player an observer; players
        that do not override it are skipped, so the hook costs them nothing.
        """

    def after(self, game):
        """Called once the game is over.

        ``game.winning_color()`` is None if the game hit the turn limit.
        """
