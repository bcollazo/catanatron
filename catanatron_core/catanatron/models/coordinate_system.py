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


def offset_to_cube(offset):
    x = offset[0] - (offset[1] - (offset[1] & 1)) / 2
    z = offset[1]
    y = -x - z
    return (x, y, z)
