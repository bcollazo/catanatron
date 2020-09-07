import json
import uuid

from flask import Flask, jsonify, abort
from flask_cors import CORS

from catanatron.game import Game
from catanatron.models.map import Water, Port
from catanatron.models.player import RandomPlayer, Color


app = Flask(__name__)
CORS(app)


def serialize_tile(tile):
    if isinstance(tile, Water):
        return {"type": "WATER"}
    elif isinstance(tile, Port):
        return {
            "type": "PORT",
            "direction": tile.direction.value,
            "resource": None if tile.resource == None else tile.resource.value,
        }
    elif tile.resource == None:
        return {"type": "DESERT"}
    return {
        "type": "RESOURCE_TILE",
        "resource": tile.resource.value,
        "number": tile.number,
    }


def serialize_game(game):
    tiles = []
    nodes = {}
    edges = {}
    for coordinate, tile in game.board.tiles.items():
        tiles.append({"coordinate": coordinate, "tile": serialize_tile(tile)})
        for direction, node in tile.nodes.items():
            nodes[node.id] = {
                "tile_coordinate": coordinate,
                "direction": direction.value,
            }
        for direction, edge in tile.edges.items():
            edges[edge.id] = {
                "tile_coordinate": coordinate,
                "direction": direction.value,
            }

    return {"tiles": tiles, "nodes": nodes, "edges": edges}


games = {}


@app.route("/games", methods=["POST"])
def create_game():
    game = Game(players=[RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    game_id = uuid.uuid4()
    games[str(game_id)] = game
    return jsonify({"game_id": game_id})


@app.route("/games/<string:game_id>", methods=["GET"])
def get_game(game_id):
    print(games.keys())
    if game_id not in games:
        abort(404, description="Resource not found")

    game = games[game_id]
    return jsonify(serialize_game(game))
