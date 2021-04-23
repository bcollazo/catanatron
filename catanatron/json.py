import json
from enum import Enum

from catanatron.game import Game
from catanatron.models.map import Water, Port, Tile
from catanatron.models.player import Player
from catanatron.models.decks import Deck


def longest_roads_by_player(state):
    roads = {
        player.color.value: state.board.continuous_roads_by_player(player.color)
        for player in state.players
    }

    return {
        key: 0 if len(value) == 0 else max(map(len, value))
        for key, value in roads.items()
    }


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
            for coordinate, tile in obj.state.board.map.tiles.items():
                for direction, node_id in tile.nodes.items():
                    building = obj.state.board.buildings.get(node_id, None)
                    color = None if building is None else building[0]
                    building_type = None if building is None else building[1]
                    nodes[node_id] = {
                        "id": node_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "building": self.default(building_type),
                        "color": self.default(color),
                    }
                for direction, edge in tile.edges.items():
                    color = obj.state.board.roads.get(edge, None)
                    edges[edge] = {
                        "id": edge,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "color": self.default(color),
                    }
            return {
                "tiles": [
                    {"coordinate": coordinate, "tile": self.default(tile)}
                    for coordinate, tile in obj.state.board.map.tiles.items()
                ],
                "nodes": nodes,
                "edges": list(edges.values()),
                "actions": [self.default(a) for a in obj.state.actions],
                "players": [self.default(p) for p in obj.state.players],
                "robber_coordinate": obj.state.board.robber_coordinate,
                "current_color": obj.state.current_player().color,
                "current_prompt": obj.state.current_prompt,
                "current_playable_actions": obj.state.playable_actions,
                # TODO: Use cached value when we cache it.
                "longest_roads_by_player": longest_roads_by_player(obj.state),
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
