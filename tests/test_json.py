import pytest
import json

from catanatron.game import Game
from catanatron.models.enums import ActionType, WOOD, BRICK, SHEEP, ORE
from catanatron.models.player import SimplePlayer, Color
from catanatron.json import GameEncoder, action_from_json


def test_serialization_matches_gui_contract():
    game = Game(
        players=[
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ],
        seed=123,
    )

    result = json.loads(json.dumps(game, cls=GameEncoder))

    assert {
        "tiles",
        "adjacent_tiles",
        "nodes",
        "edges",
        "action_records",
        "player_state",
        "colors",
        "bot_colors",
        "is_initial_build_phase",
        "robber_coordinate",
        "current_color",
        "current_prompt",
        "current_discard_count",
        "current_playable_actions",
        "longest_roads_by_player",
        "winning_color",
        "state_index",
    } <= set(result)
    assert "random" not in result
    assert isinstance(result["tiles"], list)
    assert isinstance(result["nodes"], dict)
    assert isinstance(result["edges"], list)
    assert isinstance(result["action_records"], list)
    assert isinstance(result["robber_coordinate"], list)
    assert result["winning_color"] is None

    tile_types = {placed_tile["tile"]["type"] for placed_tile in result["tiles"]}
    assert {"RESOURCE_TILE", "DESERT", "PORT", "WATER"} <= tile_types
    assert any(
        placed_tile["tile"]["type"] == "PORT"
        and placed_tile["tile"]["resource"] is None
        for placed_tile in result["tiles"]
    )


def test_action_from_json_maritime_trade():
    data = ["RED", "MARITIME_TRADE", [SHEEP, SHEEP, SHEEP, SHEEP, ORE]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MARITIME_TRADE
    assert action.value == (SHEEP, SHEEP, SHEEP, SHEEP, ORE)


def test_action_from_json_play_year_of_plenty_two_resources():
    data = ["RED", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (WOOD, BRICK)


def test_action_from_json_play_year_of_plenty_one_resource():
    data = ["BLUE", "PLAY_YEAR_OF_PLENTY", [SHEEP]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.PLAY_YEAR_OF_PLENTY
    assert action.value == (SHEEP,)


def test_action_from_json_discard():
    data = ["BLUE", "DISCARD_RESOURCE", WOOD]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.DISCARD_RESOURCE
    assert action.value == WOOD


def test_action_from_json_play_year_of_plenty_invalid():
    data = ["WHITE", "PLAY_YEAR_OF_PLENTY", [WOOD, BRICK, SHEEP]]
    with pytest.raises(
        ValueError, match="Year of Plenty action must have 1 or 2 resources"
    ):
        action_from_json(data)


def test_action_from_json_move_robber_with_victim():
    data = ["ORANGE", "MOVE_ROBBER", [[0, 0, 0], "RED"]]
    action = action_from_json(data)
    assert action.color == Color.ORANGE
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((0, 0, 0), Color.RED)


def test_action_from_json_move_robber_without_victim():
    data = ["RED", "MOVE_ROBBER", [[1, -1, 0], None]]
    action = action_from_json(data)
    assert action.color == Color.RED
    assert action.action_type == ActionType.MOVE_ROBBER
    assert action.value == ((1, -1, 0), None)


def test_action_from_json_build_road():
    data = ["BLUE", "BUILD_ROAD", [0, 1]]
    action = action_from_json(data)
    assert action.color == Color.BLUE
    assert action.action_type == ActionType.BUILD_ROAD
    assert action.value == (0, 1)
