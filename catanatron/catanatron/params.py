"""A player's tunable configuration.

Each player class declares what it can tune as a nested ``Params``, a frozen
dataclass. That declaration is the only source of truth: the CLI coerces
command-line strings into the declared types, ``--help-players`` lists them,
and ``GET /api/players`` publishes them so a UI can render a form.
"""

import copy
import dataclasses
from typing import Literal, get_args, get_origin, get_type_hints


class ParamsError(ValueError):
    """Params that cannot be built from what the user asked for."""


class BaseParams:
    """Base for a player's configuration. A frozen dataclass, so it is
    immutable, comparable, and rejects params it does not declare::

        class MyBot(Player):
            class Params(BaseParams):
                aggression: int = 1
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in getattr(cls, "__annotations__", {}):
            default = getattr(cls, name, None)
            # A mutable default would otherwise be shared by every instance.
            if isinstance(default, (dict, list, set)):
                factory = lambda value=default: copy.deepcopy(value)  # noqa: E731
                setattr(cls, name, dataclasses.field(default_factory=factory))
        dataclasses.dataclass(frozen=True)(cls)


class NoParams(BaseParams):
    """For players with nothing to tune."""


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).strip().lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(value)


#: The only types settable from a CLI string or an API request. A param of any
#: other type stays available to code, but is not published or parsed.
TYPES = {int: "int", float: "float", str: "str", bool: "bool"}
COERCE = {"int": int, "float": float, "str": str, "bool": _to_bool}


def schema_of(player_class):
    """The settable params, in declaration order.

    This is both what ``--help-players`` lists and what ``GET /api/players``
    publishes, and it is the only description ``build_params`` works from.
    """
    hints = get_type_hints(player_class.Params)
    schema = []
    for field in dataclasses.fields(player_class.Params):
        annotation, choices = hints[field.name], None
        if get_origin(annotation) is Literal:
            # Literal["base", "contender"] is how a param declares its choices.
            choices = list(get_args(annotation))
            annotation = type(choices[0])
        elif get_args(annotation):
            # Optional[T] is Union[T, None]; take T.
            rest = [a for a in get_args(annotation) if a is not type(None)]
            annotation = rest[0] if len(rest) == 1 else None
        if annotation not in TYPES:
            continue
        entry = {
            "name": field.name,
            "type": TYPES[annotation],
            "default": (
                field.default if field.default is not dataclasses.MISSING else None
            ),
            "help": "",
        }
        if choices:
            entry["choices"] = choices
        schema.append(entry)
    return schema


def build_params(player_class, args=(), named=None):
    """Build ``player_class.Params`` from positional and named values.

    Positional values bind to params in declaration order, so ``AB:2:contender``
    means depth then value_fn.
    """
    name_of = player_class.__name__
    schema = {entry["name"]: entry for entry in schema_of(player_class)}
    if len(args) > len(schema):
        raise ParamsError(
            f"{name_of} takes at most {len(schema)} positional param(s) "
            f"({', '.join(schema) or 'none'}), got {len(args)}"
        )

    values = dict(zip(schema, args))
    for name, value in (named or {}).items():
        if name in values:
            raise ParamsError(f"{name_of}.{name} given twice")
        values[name] = value

    for name in list(values):
        entry, value = schema.get(name), values[name]
        if entry is None:
            raise ParamsError(
                f"{name_of} has no param {name!r}; "
                f"try one of: {', '.join(schema) or 'none'}"
            )
        if value is None and entry["default"] is None:
            continue  # an optional param, explicitly left unset
        try:
            values[name] = COERCE[entry["type"]](value)
        except (TypeError, ValueError):
            raise ParamsError(
                f"{name_of}.{name}: {value!r} is not a valid {entry['type']}"
            )
        if choices := entry.get("choices"):
            if values[name] not in choices:
                raise ParamsError(
                    f"{name_of}.{name}: {value!r} is not one of "
                    f"{', '.join(map(str, choices))}"
                )
    return player_class.Params(**values)
