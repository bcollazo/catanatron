"""A player's tunable configuration.

Each player class declares what it can tune as a nested ``Params`` model.
That declaration is the only source of truth: the CLI coerces command-line
strings into the declared types, ``--help-players`` lists them, and
``GET /api/players`` publishes them so a UI can render a form.

Validation is pydantic's, so a wrong type or an unknown name fails at
construction with a clear message rather than somewhere deep in a game.
"""

from typing import Any, Dict, List, Literal, Sequence, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError


class ParamsError(ValueError):
    """Params that cannot be built from what the user asked for."""


class BaseParams(BaseModel):
    """Base for a player's configuration. Immutable, and closed to extras.

    Named ``BaseParams`` rather than ``Params`` so that the nested class can
    be called ``Params`` without shadowing its own base:

        class MyBot(Player):
            class Params(BaseParams):
                aggression: int = 1
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoParams(BaseParams):
    """For players with nothing to tune."""


#: Types a param can have and still be settable from a CLI string or an API
#: request. Anything else is for programmatic use only.
SETTABLE = {int: "int", float: "float", str: "str", bool: "bool"}


def _type_name(annotation):
    """A JSON-friendly name for a field's type, or None if it is not settable."""
    if annotation in SETTABLE:
        return SETTABLE[annotation]
    if get_origin(annotation) is Literal:
        # Literal["base", "contender"] is how a param declares its choices.
        kinds = {type(a) for a in get_args(annotation)}
        if len(kinds) == 1 and kinds.pop() in SETTABLE:
            return SETTABLE[type(get_args(annotation)[0])]
        return None
    # Optional[T] is Union[T, None]; take T if it is settable.
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1 and args[0] in SETTABLE:
        return SETTABLE[args[0]]
    return None


def schema_of(player_class) -> List[Dict[str, Any]]:
    """What ``GET /api/players`` publishes, and what ``--help-players`` lists."""
    schema = []
    for name, field in player_class.Params.model_fields.items():
        type_name = _type_name(field.annotation)
        if type_name is None:
            continue
        entry = {
            "name": name,
            "type": type_name,
            "default": field.default,
            "help": field.description or "",
        }
        if get_origin(field.annotation) is Literal:
            entry["choices"] = list(get_args(field.annotation))
        schema.append(entry)
    return schema


def build_params(player_class, args: Sequence = (), named: Dict[str, Any] = None):
    """Build ``player_class.Params`` from positional and named values.

    Positional values bind to fields in declaration order, so ``AB:2:contender``
    means depth then value_fn.
    """
    model = player_class.Params
    settable = [entry["name"] for entry in schema_of(player_class)]
    if len(args) > len(settable):
        raise ParamsError(
            f"{player_class.__name__} takes at most {len(settable)} positional "
            f"param(s) ({', '.join(settable) or 'none'}), got {len(args)}"
        )

    values = dict(zip(settable, args))
    for name, value in (named or {}).items():
        if name in values:
            raise ParamsError(f"{player_class.__name__}.{name} given twice")
        values[name] = value

    try:
        return model(**values)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "params"
        raise ParamsError(f"{player_class.__name__}.{where}: {first['msg']}")
