"""The players a game can seat, shared by the CLI and the web server.

Lives in the core package (rather than under ``catanatron.cli``) so that
``catanatron.web`` can use it without depending on click or rich.

A player is addressed by a *spec*, in either of two equivalent forms::

    "AB:2:contender"                                    the CLI
    {"key": "AB", "params": {"depth": 2, ...}}          an API body, a DB row

Params may be positional, named, or positional-then-named, the same way
Python arguments work; field declaration order is the positional order.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from catanatron.models.player import Color, Player
from catanatron.params import ParamsError, build_params, schema_of
from catanatron.sources import EXEC_PREFIX, SourceError, identity, load_class


class SpecError(ValueError):
    """A spec that cannot be resolved. Always raised, never swallowed."""


@dataclass(frozen=True)
class PlayerEntry:
    """One registered player: how to build it, and how to describe it."""

    key: str
    name: str
    description: str
    player_class: Any

    @property
    def is_bot(self) -> bool:
        return self.player_class.IS_BOT

    @property
    def params_schema(self) -> List[Dict[str, Any]]:
        return schema_of(self.player_class)

    def build(self, color: Color, args=(), named=None) -> Player:
        player = self.player_class(color, build_params(self.player_class, args, named))
        # Remember which key built this player, so spec_of() stays exact even
        # when several keys map to the same class.
        player.registry_key = self.key
        return player

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": (self.description or "").strip(),
            "is_bot": self.is_bot,
            "params": self.params_schema,
        }


def parse_spec(spec: Union[str, dict]) -> tuple:
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


class PlayerRegistry:
    """Maps keys to the player classes that implement them."""

    def __init__(self):
        self._entries: Dict[str, PlayerEntry] = {}

    def register(self, key, player_class, *, name=None, replace=False) -> PlayerEntry:
        key = key.upper()
        if key in self._entries and not replace:
            raise SpecError(f"player key {key!r} is already registered")
        entry = PlayerEntry(
            key=key,
            name=name or player_class.__name__,
            description=player_class.__doc__ or "",
            player_class=player_class,
        )
        self._entries[key] = entry
        return entry

    def register_source(self, source: str, name=None, base_dir=None) -> PlayerEntry:
        """Register one ``--bot`` declaration."""
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

        key = name or player_class.__name__
        existing = self._entries.get(key.upper())
        if existing is not None and identity(existing.player_class) != identity(
            player_class
        ):
            raise SpecError(f"{key} collides with the existing {existing.name!r}")
        return self.register(key, player_class, replace=True)

    def get(self, key: str) -> PlayerEntry:
        key = str(key).upper()
        if key not in self._entries:
            raise SpecError(
                f"Unknown player {key!r}. "
                f"Available: {', '.join(sorted(self._entries))}"
            )
        return self._entries[key]

    def entries(self) -> List[PlayerEntry]:
        return sorted(self._entries.values(), key=lambda e: e.key)

    def build(self, spec: Union[str, dict], color: Color) -> Player:
        key, args, named = parse_spec(spec)
        try:
            return self.get(key).build(color, args, named)
        except ParamsError as error:
            raise SpecError(str(error))

    def build_all(self, specs, colors=None) -> List[Player]:
        """Build the players for one game.

        Unlike the old ``parse_cli_string``, an unrecognized key raises rather
        than silently dropping the player.
        """
        if isinstance(specs, str):
            specs = [s for s in specs.split(",") if s.strip()]
        specs = list(specs)
        if not 2 <= len(specs) <= 4:
            raise SpecError(f"a game needs 2 to 4 players, got {len(specs)}")
        colors = list(colors or Color)[: len(specs)]
        return [self.build(spec, color) for spec, color in zip(specs, colors)]

    def spec_of(self, player: Player) -> dict:
        """Structured spec for a built player, for persisting to a database."""
        key = getattr(player, "registry_key", None)
        entry = self._entries.get(key) if key else None
        if entry is None:
            for candidate in self._entries.values():
                if type(player) is candidate.player_class:
                    entry = candidate
                    break
        if entry is None:
            raise SpecError(f"{type(player).__name__} is not registered")
        names = [p["name"] for p in entry.params_schema]
        return {
            "key": entry.key,
            "params": {n: getattr(player.params, n) for n in names},
        }


#: The process-wide registry. Builtins register themselves on import of
#: ``catanatron.players``.
REGISTRY = PlayerRegistry()
