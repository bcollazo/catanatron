import json
from enum import Enum

from catanatron.game import Game
from catanatron.models.map import Water, Port, Tile
from catanatron.models.player import Player
from catanatron.models.decks import Deck


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
                for direction, node_id in tile.nodes.items():
                    building = obj.board.nxgraph.nodes[node_id].get("building", None)
                    color = obj.board.nxgraph.nodes[node_id].get("color", None)
                    nodes[node_id] = {
                        "id": node_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building),
                        "color": self.default(color),
                    }
                for direction, edge in tile.edges.items():
                    color = obj.board.nxgraph.edges[edge].get("color", None)
                    edges[edge] = {
                        "id": edge,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "color": self.default(color),
                    }
            return {
                "tiles": [
                    {"coordinate": coordinate, "tile": self.default(tile)}
                    for coordinate, tile in obj.board.tiles.items()
                ],
                "nodes": nodes,
                "edges": list(edges.values()),
                "actions": [self.default(a) for a in obj.actions],
                "players": [self.default(p) for p in obj.players],
                "robber_coordinate": obj.board.robber_coordinate,
            }
        if isinstance(obj, Deck):
            return {resource.value: count for resource, count in obj.cards.items()}
        if isinstance(obj, Player):
            return {k: v for k, v in obj.__dict__.items() if k != "buildings"}
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            return {
                "id": obj.id,
                "type": "PORT",
                "direction": self.default(obj.direction),
                "resource": self.default(obj.resource),
            }
        if isinstance(obj, Tile):
            if obj.resource is None:
                return {"id": obj.id, "type": "DESERT"}
            return {
                "id": obj.id,
                "type": "RESOURCE_TILE",
                "resource": self.default(obj.resource),
                "number": obj.number,
            }
        return json.JSONEncoder.default(self, obj)
