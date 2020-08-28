import json

from flask import Flask, jsonify
from flask_cors import CORS

from catanatron.models import Game, Water, Port


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


@app.route("/board")
def board():
    game = Game()  # Make new in-memory Game, in the future we read board#i
    return jsonify(serialize_game(game))

