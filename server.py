import json
import uuid
from enum import Enum

from flask import Flask, jsonify, abort
from flask_cors import CORS

from catanatron.game import Game
from catanatron.models.map import Water, Port, Tile
from catanatron.models.player import RandomPlayer, Color, Player
from catanatron.models.actions import Action
from catanatron.models.decks import ResourceDecks
from catanatron.models.board import Building
from catanatron.models.board_initializer import Node, Edge, NodeRef, EdgeRef


app = Flask(__name__)
CORS(app)


games = {}


@app.route("/games", methods=["POST"])
def create_game():
    game = Game(
        players=[
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
    )
    game.play_initial_build_phase()
    game_id = uuid.uuid4()
    games[str(game_id)] = game
    return jsonify({"game_id": game_id})


@app.route("/games/<string:game_id>", methods=["GET"])
def get_game(game_id):
    if game_id not in games:
        abort(404, description="Resource not found")

    game = games[game_id]
    return json.dumps(game, cls=GameEncoder)


@app.route("/games/<string:game_id>/tick", methods=["POST"])
def tick_game(game_id):
    if game_id not in games:
        abort(404, description="Resource not found")

    game = games[game_id]
    game.play_tick()
    return json.dumps(game, cls=GameEncoder)


class GameEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Building) or isinstance(obj, Action):
            return obj
        if isinstance(obj, Game):
            nodes = {}
            edges = {}
            for coordinate, tile in obj.board.tiles.items():
                for direction, node in tile.nodes.items():
                    building = obj.board.buildings.get(node, None)
                    nodes[node.id] = {
                        "id": node.id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building),
                    }
                for direction, edge in tile.edges.items():
                    building = obj.board.buildings.get(edge, None)
                    edges[edge.id] = {
                        "id": edge.id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building),
                    }

            return {
                "tiles": [
                    {"coordinate": coordinate, "tile": self.default(tile)}
                    for coordinate, tile in obj.board.tiles.items()
                ],
                "nodes": nodes,
                "edges": edges,
                "actions": [self.default(a) for a in obj.actions],
                "players": [self.default(p) for p in obj.players],
                "robber_coordinate": obj.board.robber_coordinate,
            }
        if isinstance(obj, ResourceDecks):
            return {resource.value: count for resource, count in obj.decks.items()}
        if isinstance(obj, Player):
            return obj.__dict__
        if isinstance(obj, Node) or isinstance(obj, Edge):
            return obj.id
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            return {
                "type": "PORT",
                "direction": self.default(obj.direction),
                "resource": self.default(obj.resource),
            }
        if isinstance(obj, Tile):
            if obj.resource == None:
                return {"type": "DESERT"}
            return {
                "type": "RESOURCE_TILE",
                "resource": self.default(obj.resource),
                "number": obj.number,
            }
        return json.JSONEncoder.default(self, obj)


@app.route("/test", methods=["GET"])
def test():
    game = Game(
        players=[
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
    )
    game.play_initial_build_phase()

    return json.dumps(game, cls=GameEncoder)
