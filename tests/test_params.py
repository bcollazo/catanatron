"""A player's Params model is the single source of truth for what it tunes."""

import dataclasses
from typing import Literal, Optional

import pytest

from catanatron.models.player import Color, Player
from catanatron.params import ParamsError, build_params, schema_of
from catanatron.players.minimax import AlphaBetaPlayer


class ExampleBot(Player):
    """An example bot."""

    @dataclasses.dataclass(frozen=True)
    class Params:
        aggression: int = 1
        chatty: bool = False
        nickname: str = "bot"
        mood: Literal["calm", "wild"] = "calm"
        patience: Optional[float] = None
        #: Not a settable type: programmatic use only. A mutable default has
        #: to be a factory -- dataclasses reject the plain value.
        weights: dict = dataclasses.field(default_factory=dict)

    def decide(self, game, playable_actions):
        return playable_actions[0]


def test_schema_lists_the_settable_fields_in_order():
    assert [p["name"] for p in schema_of(ExampleBot)] == [
        "aggression",
        "chatty",
        "nickname",
        "mood",
        "patience",
    ]


def test_schema_reports_types_and_defaults():
    schema = {p["name"]: p for p in schema_of(ExampleBot)}
    assert schema["aggression"]["type"] == "int"
    assert schema["aggression"]["default"] == 1
    assert schema["chatty"]["type"] == "bool"
    assert schema["patience"]["type"] == "float"


def test_literal_fields_publish_their_choices():
    mood = next(p for p in schema_of(ExampleBot) if p["name"] == "mood")
    assert mood["type"] == "str"
    assert mood["choices"] == ["calm", "wild"]


def test_non_scalar_fields_are_not_settable():
    """A dict of weights stays available to code but not to the CLI or API."""
    assert "weights" not in [p["name"] for p in schema_of(ExampleBot)]
    assert "weights" in [f.name for f in dataclasses.fields(ExampleBot.Params)]


def test_players_without_params_have_an_empty_schema():
    from catanatron.models.player import RandomPlayer

    assert schema_of(RandomPlayer) == []


# ===== building =====
def test_positional_values_bind_in_declaration_order():
    params = build_params(ExampleBot, ["3", "true"])
    assert params.aggression == 3
    assert params.chatty is True


def test_named_values():
    assert build_params(ExampleBot, [], {"nickname": "zed"}).nickname == "zed"


def test_positional_then_named():
    params = build_params(ExampleBot, ["2"], {"nickname": "zed"})
    assert (params.aggression, params.nickname) == (2, "zed")


def test_defaults_when_nothing_is_given():
    assert build_params(ExampleBot).aggression == 1


def test_params_are_frozen():
    params = build_params(ExampleBot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.aggression = 5


def test_a_mutable_default_is_not_shared_between_instances():
    """Two bots tuning their own weights must not tune each other's."""
    first, second = build_params(ExampleBot), build_params(ExampleBot)
    assert first.weights == second.weights
    first.weights["public_vps"] = 1
    assert second.weights == {}


def test_a_plain_mutable_default_is_rejected_at_declaration():
    """The stdlib says so, before any game runs, and names the fix."""
    with pytest.raises(ValueError, match="use default_factory"):

        @dataclasses.dataclass(frozen=True)
        class Params:
            weights: dict = {}


def test_params_that_are_not_a_dataclass_are_rejected():
    class Undecorated(Player):
        class Params:
            aggression: int = 1

        def decide(self, game, playable_actions):
            return playable_actions[0]

    with pytest.raises(ParamsError, match="must be a dataclass"):
        schema_of(Undecorated)


def test_repr_renders_the_scalar_params():
    """The CLI's game summary prints players this way."""
    player = ExampleBot(Color.RED, build_params(ExampleBot, ["3"]))
    assert (
        repr(player)
        == "ExampleBot:RED(aggression=3,chatty=False,nickname=bot,mood=calm)"
    )


@pytest.mark.parametrize(
    "raw,expected", [("true", True), ("1", True), ("false", False)]
)
def test_bool_coercion(raw, expected):
    assert build_params(ExampleBot, [], {"chatty": raw}).chatty is expected


def test_optional_scalar_accepts_none():
    assert build_params(AlphaBetaPlayer, [], {"epsilon": None}).epsilon is None


# ===== failing fast =====
def test_wrong_type_is_rejected():
    with pytest.raises(ParamsError, match="'lots' is not a valid int"):
        build_params(ExampleBot, [], {"aggression": "lots"})


def test_a_word_that_is_not_a_boolean_is_rejected():
    """main's 'AB:2:C' silently meant prunning=True."""
    with pytest.raises(ParamsError, match="'C' is not a valid bool"):
        build_params(ExampleBot, [], {"chatty": "C"})


def test_unknown_param_is_rejected():
    with pytest.raises(ParamsError, match="has no param 'agression'"):
        build_params(ExampleBot, [], {"agression": 2})


def test_value_outside_the_choices_is_rejected():
    with pytest.raises(ParamsError, match="not one of calm, wild"):
        build_params(ExampleBot, [], {"mood": "grumpy"})


def test_too_many_positional_values_is_rejected():
    with pytest.raises(ParamsError, match="at most 5 positional"):
        build_params(ExampleBot, ["1", "2", "3", "4", "5", "6"])


def test_a_param_given_twice_is_rejected():
    with pytest.raises(ParamsError, match="given twice"):
        build_params(ExampleBot, ["3"], {"aggression": 4})
