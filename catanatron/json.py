import json
from enum import Enum

from catanatron.game import Game
from catanatron.models.map import Water, Port, Tile
from catanatron.models.player import Player
from catanatron.models.actions import Action
from catanatron.models.decks import Deck
from catanatron.models.board import Building
from catanatron.models.board_initializer import Node, Edge, NodeRef, EdgeRef


class GameEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
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
        if isinstance(obj, Deck):
            return {resource.value: count for resource, count in obj.cards.items()}
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
