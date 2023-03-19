from enum import Enum


# We'll be using Cube coordinates in https://math.stackexchange.com/questions/2254655/hexagon-grid-coordinate-system
class Direction(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


UNIT_VECTORS = {
    # X-axis
    Direction.NORTHEAST: (1, 0, -1),
    Direction.SOUTHWEST: (-1, 0, 1),
    # Y-axis
    Direction.NORTHWEST: (0, 1, -1),
    Direction.SOUTHEAST: (0, -1, 1),
    # Z-axis
    Direction.EAST: (1, -1, 0),
    Direction.WEST: (-1, 1, 0),
}


def add(acoord, bcoord):
    (x, y, z) = acoord
    (u, v, w) = bcoord
    return (x + u, y + v, z + w)


def num_tiles_for(layer):
    """Including inner-layer tiles"""
    if layer == 0:
        return 1

    return 6 * layer + num_tiles_for(layer - 1)


def generate_coordinate_system(num_layers):
    """
    Generates a set of coordinates by expanding outward from a center tile on
    (0,0,0) with the given number of layers (as in an onion :)). Follows BFS.
    """
    num_tiles = num_tiles_for(num_layers)

    agenda = [(0, 0, 0)]
    visited = set()
    while len(visited) < num_tiles:
        node = agenda.pop(0)
        visited.add(node)

        neighbors = [add(node, UNIT_VECTORS[d]) for d in Direction]
        new_neighbors = filter(
            lambda x: x not in visited and x not in agenda, neighbors
        )
        agenda.extend(new_neighbors)
    return visited


def cube_to_axial(cube):
    q = cube[0]
    r = cube[2]
    return (q, r)


def cube_to_offset(cube):
    col = cube[0] + (cube[2] - (cube[2] & 1)) / 2
    row = cube[2]
    return (col, row)


def offset_to_cube(offset):
    x = offset[0] - (offset[1] - (offset[1] & 1)) / 2
    z = offset[1]
    y = -x - z
    return (x, y, z)
