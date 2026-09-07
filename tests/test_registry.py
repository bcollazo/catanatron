"""The registry maps keys to players, for both the CLI and the web."""

import sys
import textwrap

import pytest

import catanatron.players  # noqa: F401  (registers the builtins)
from catanatron.models.player import Color, Player
from catanatron.params import BaseParams
from catanatron.registry import REGISTRY, PlayerRegistry, SpecError, parse_spec


class ExampleBot(Player):
    """An example bot."""

    class Params(BaseParams):
        aggression: int = 1

    def decide(self, game, playable_actions):
        return playable_actions[0]


@pytest.fixture
def registry():
    registry = PlayerRegistry()
    registry.register("EX", ExampleBot)
    return registry


# ===== specs =====
def test_parse_named_params():
    assert parse_spec("AB:depth=3") == ("AB", [], {"depth": "3"})


def test_parse_positional_params():
    assert parse_spec("AB:3:contender") == ("AB", ["3", "contender"], {})


def test_parse_positional_then_named():
    assert parse_spec("AB:3:prunning=true") == ("AB", ["3"], {"prunning": "true"})


def test_parse_bare_key_is_case_insensitive():
    assert parse_spec("r") == ("R", [], {})


def test_parse_dict_spec():
    assert parse_spec({"key": "ab", "params": {"depth": 3}}) == ("AB", [], {"depth": 3})


def test_both_spec_forms_agree():
    from_string = REGISTRY.build("AB:3", Color.RED)
    from_dict = REGISTRY.build({"key": "AB", "params": {"depth": 3}}, Color.RED)
    assert from_string.params == from_dict.params


@pytest.mark.parametrize("spec", ["", "   ", ":", None, 5])
def test_malformed_specs_are_rejected(spec):
    with pytest.raises(SpecError):
        parse_spec(spec)


def test_positional_after_named_is_rejected():
    with pytest.raises(SpecError, match="after a named one"):
        parse_spec("AB:depth=2:3")


# ===== building =====
def test_build_applies_params(registry):
    assert registry.build("EX:5", Color.RED).params.aggression == 5


def test_build_assigns_the_colour(registry):
    assert registry.build("EX", Color.BLUE).color == Color.BLUE


def test_builtin_positional_and_named_agree():
    assert (
        REGISTRY.build("AB:2:contender", Color.RED).params
        == REGISTRY.build("AB:depth=2:value_fn=contender", Color.RED).params
    )


def test_unknown_key_is_rejected(registry):
    with pytest.raises(SpecError, match="Unknown player 'NOPE'"):
        registry.build("NOPE", Color.RED)


def test_unknown_key_is_not_silently_dropped(registry):
    """The old parse_cli_string returned a short player list instead."""
    with pytest.raises(SpecError):
        registry.build_all("EX,EX,NOPE")


def test_bad_params_surface_as_a_spec_error(registry):
    with pytest.raises(SpecError, match="valid integer"):
        registry.build("EX:lots", Color.RED)


def test_build_all_assigns_colors_in_order():
    players = REGISTRY.build_all("R,W,AB:2")
    assert [p.color for p in players] == [Color.RED, Color.BLUE, Color.ORANGE]


@pytest.mark.parametrize("specs", ["R", "R,R,R,R,R"])
def test_build_all_rejects_bad_player_counts(specs):
    with pytest.raises(SpecError, match="2 to 4 players"):
        REGISTRY.build_all(specs)


# ===== registration =====
def test_duplicate_registration_requires_replace(registry):
    with pytest.raises(SpecError, match="already registered"):
        registry.register("EX", ExampleBot)
    registry.register("EX", ExampleBot, replace=True)


def test_entry_to_json_shape(registry):
    payload = registry.get("EX").to_json()
    assert payload["key"] == "EX"
    assert payload["name"] == "ExampleBot"
    assert payload["description"] == "An example bot."
    assert payload["is_bot"] is True
    assert [p["name"] for p in payload["params"]] == ["aggression"]


def test_is_bot_comes_from_the_class():
    from catanatron.models.player import HumanPlayer, RandomPlayer

    assert RandomPlayer(Color.RED).is_bot is True
    assert HumanPlayer(Color.RED).is_bot is False


# ===== spec_of, for persisting a game =====
def test_spec_of_round_trips(registry):
    player = registry.build("EX:4", Color.WHITE)
    spec = registry.spec_of(player)
    assert spec == {"key": "EX", "params": {"aggression": 4}}
    assert registry.build(spec, Color.WHITE).params == player.params


def test_spec_of_is_exact_when_two_keys_share_a_class(registry):
    registry.register("ALIAS", ExampleBot)
    assert registry.spec_of(registry.build("ALIAS", Color.RED))["key"] == "ALIAS"
    assert registry.spec_of(registry.build("EX", Color.RED))["key"] == "EX"


def test_spec_of_unregistered_player_is_rejected(registry):
    class Unregistered(Player):
        def decide(self, game, playable_actions):
            return playable_actions[0]

    with pytest.raises(SpecError, match="not registered"):
        registry.spec_of(Unregistered(Color.RED))


# ===== --bot sources =====
BOT_FILE = '''
from catanatron import Player, BaseParams


class SoloBot(Player):
    """The only player here."""

    class Params(BaseParams):
        aggression: int = 1

    def decide(self, game, playable_actions):
        return playable_actions[0]
'''

TWO_BOTS_FILE = """
from catanatron import Player


class AlphaBot(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]


class BetaBot(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]
"""


@pytest.fixture
def bot_file(tmp_path):
    path = tmp_path / "mybot.py"
    path.write_text(textwrap.dedent(BOT_FILE))
    return path


def test_loads_a_file_outside_the_working_directory(bot_file, registry):
    entry = registry.register_source(str(bot_file))
    assert entry.key == "SOLOBOT"
    assert registry.build("SoloBot:3", Color.RED).params.aggression == 3


def test_explicit_name_overrides_the_class_name(bot_file, registry):
    assert registry.register_source(str(bot_file), name="RUSH").key == "RUSH"


def test_relative_paths_resolve_against_base_dir(bot_file, registry):
    entry = registry.register_source("mybot.py", base_dir=str(bot_file.parent))
    assert entry.name == "SoloBot"


def test_hash_selects_a_class_when_several_are_defined(tmp_path, registry):
    path = tmp_path / "two.py"
    path.write_text(textwrap.dedent(TWO_BOTS_FILE))
    assert registry.register_source(f"{path}#BetaBot").name == "BetaBot"


def test_ambiguous_file_lists_the_candidates(tmp_path, registry):
    path = tmp_path / "two.py"
    path.write_text(textwrap.dedent(TWO_BOTS_FILE))
    with pytest.raises(SpecError, match="AlphaBot, BetaBot"):
        registry.register_source(str(path))


def test_missing_file_is_rejected(tmp_path, registry):
    with pytest.raises(SpecError, match="no such file"):
        registry.register_source(str(tmp_path / "nope.py"))


def test_importable_module_form(registry):
    entry = registry.register_source(
        "catanatron.players.weighted_random#WeightedRandomPlayer"
    )
    assert entry.name == "WeightedRandomPlayer"


def test_colliding_name_is_rejected(bot_file, registry):
    with pytest.raises(SpecError, match="collides with"):
        registry.register_source(str(bot_file), name="EX")


def test_redeclaring_the_same_bot_is_allowed(bot_file, registry):
    """Re-importing a file yields a fresh class object, not a collision."""
    first = registry.register_source(str(bot_file))
    assert registry.register_source(str(bot_file), name=first.key).key == first.key


@pytest.mark.parametrize("source", ["http://host/decide", "https://host:8080/x"])
def test_http_sources_say_they_are_not_built_yet(source, registry):
    with pytest.raises(SpecError, match="not supported yet"):
        registry.register_source(source, name="X")


def test_windows_style_path_keeps_its_drive_letter(registry):
    r"""'#' separates the class precisely so C:\bots\x.py survives."""
    with pytest.raises(SpecError, match="no such file"):
        registry.register_source(r"C:\bots\mybot.py#MyBot")


def test_exec_source_needs_a_name(registry):
    with pytest.raises(SpecError, match="needs a name"):
        registry.register_source("exec:./bot")
