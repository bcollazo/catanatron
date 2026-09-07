"""The players a game can seat, shared by the CLI and the web server.

Lives in the core package (rather than under ``catanatron.cli``) so that
``catanatron.web`` can use it without depending on click or rich.

A player is addressed by a *spec*, in either of two equivalent forms::

    "AB:2:contender"                                    the CLI
    {"key": "AB", "params": {"depth": 2, ...}}          an API body, a DB row

Params may be positional, named, or positional-then-named, the same way
Python arguments work; declaration order is the positional order.
"""

from catanatron.models.player import Color, Player
from catanatron.params import ParamsError, build_params, schema_of
from catanatron.sources import EXEC_PREFIX, SourceError, identity, load_class


class SpecError(ValueError):
    """A spec that cannot be resolved. Always raised, never swallowed."""


def parse_spec(spec):
    """Normalize either spec form to ``(key, positional, named)``."""
    if isinstance(spec, dict):
        if "key" not in spec:
            raise SpecError(f"player spec is missing 'key': {spec!r}")
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise SpecError(f"'params' must be an object, got {params!r}")
        return str(spec["key"]).upper(), [], dict(params)

    if not isinstance(spec, str) or not spec.strip():
        raise SpecError(f"invalid player spec: {spec!r}")
    key, *rest = spec.strip().split(":")
    if not key:
        raise SpecError(f"invalid player spec: {spec!r}")

    args, named = [], {}
    for part in rest:
        if "=" in part:
            name, value = part.split("=", 1)
            named[name.strip()] = value
        elif named:
            raise SpecError(f"positional param {part!r} after a named one in {spec!r}")
        else:
            args.append(part)
    return key.upper(), args, named


def describe(key, player_class):
    """What ``--help-players`` renders and ``GET /api/players`` publishes."""
    return {
        "key": key,
        # Not inherited: a subclass of AlphaBetaPlayer is its own player.
        "name": vars(player_class).get("LABEL") or player_class.__name__,
        "description": " ".join((player_class.__doc__ or "").split()),
        "is_bot": player_class.IS_BOT,
        "params": schema_of(player_class),
    }


class PlayerRegistry(dict):
    """Maps keys to the player classes that implement them."""

    def register(self, key, player_class, *, replace=False):
        key = key.upper()
        if key in self and not replace:
            raise SpecError(f"player key {key!r} is already registered")
        self[key] = player_class
        return player_class

    def register_source(self, source, name=None, base_dir=None):
        """Register one ``--bot`` declaration. Returns its key."""
        try:
            if source.startswith(EXEC_PREFIX):
                from catanatron.players.stdio import build_stdio_player_class

                if name is None:
                    raise SpecError(
                        f"{source!r}: an exec bot needs a name, e.g. RUSTY={source}"
                    )
                player_class = build_stdio_player_class(
                    name, source[len(EXEC_PREFIX) :]
                )
            else:
                player_class = load_class(source, Player, base_dir)
        except SourceError as error:
            raise SpecError(str(error))

        key = (name or player_class.__name__).upper()
        taken = self.get(key)
        # Re-importing the same file yields a fresh class object, so compare
        # where the class came from rather than the object itself.
        if taken is not None and identity(taken) != identity(player_class):
            raise SpecError(f"{key} collides with the existing {taken.__name__!r}")
        self.register(key, player_class, replace=True)
        return key

    def lookup(self, key):
        key = str(key).upper()
        if key not in self:
            raise SpecError(
                f"Unknown player {key!r}. Available: {', '.join(sorted(self))}"
            )
        return self[key]

    def build(self, spec, color: Color) -> Player:
        key, args, named = parse_spec(spec)
        player_class = self.lookup(key)
        try:
            player = player_class(color, build_params(player_class, args, named))
        except ParamsError as error:
            raise SpecError(str(error))
        # Remember which key built this player, so spec_of() stays exact even
        # when several keys map to the same class.
        player.registry_key = key
        return player

    def build_all(self, specs, colors=None):
        """Build the players for one game.

        Unlike the old ``parse_cli_string``, an unrecognized key raises rather
        than silently dropping the player.
        """
        if isinstance(specs, str):
            specs = [s for s in specs.split(",") if s.strip()]
        specs = list(specs)
        if not 2 <= len(specs) <= 4:
            raise SpecError(f"a game needs 2 to 4 players, got {len(specs)}")
        return [self.build(spec, color) for spec, color in zip(specs, colors or Color)]

    def spec_of(self, player: Player) -> dict:
        """Structured spec for a built player, for persisting to a database."""
        key = getattr(player, "registry_key", None)
        if key not in self:
            key = next((k for k, c in self.items() if type(player) is c), None)
        if key is None:
            raise SpecError(f"{type(player).__name__} is not registered")
        names = [p["name"] for p in schema_of(self[key])]
        return {"key": key, "params": {n: getattr(player.params, n) for n in names}}


#: The process-wide registry. Builtins register themselves on import of
#: ``catanatron.players``.
REGISTRY = PlayerRegistry()
