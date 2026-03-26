from collections.abc import Generator
from typing import Dict, Mapping
import random

from catanatron.models.coordinate_system import Direction, UNIT_VECTORS, add, Coordinate
from catanatron.models.tiles import LandTile, Tile

COUNTERCLOCKWISE_RING_DIRECTIONS = (
    Direction.NORTHWEST,
    Direction.WEST,
    Direction.SOUTHWEST,
    Direction.SOUTHEAST,
    Direction.EAST,
    Direction.NORTHEAST,
)

def spiral_land_coordinates(
    all_tiles: Mapping[Coordinate, Tile], start: Coordinate
) -> Generator[Coordinate, None, None]:
    """
    Yield land-tile coordinates in coast-following spiral order from an outer-edge
    start coordinate toward the center.

    Requirements:
    - ``start`` must be a land tile.
    - ``start`` must be on the outer ring of the land mass.
    - The land mass must be a single contiguous spiral-walkable shape, such as
      the standard Catan board layouts.

    Raises:
    - ValueError: If ``start`` is not land.
    - ValueError: If ``start`` is not on the outer edge.
    - ValueError: If an initial coast-following direction cannot be determined.
    """

    def is_land(coord: Coordinate) -> bool:
        return isinstance(all_tiles.get(coord), LandTile)

    if not is_land(start):
        raise ValueError("start must be a land tile")

    if all(is_land(add(start, UNIT_VECTORS[direction])) for direction in Direction):
        raise ValueError("start must be on the outer edge of the land mass")

    directions = list(Direction)
    directions.reverse()

    direction = None
    for i, candidate in enumerate(directions):
        previous = directions[i - 1]
        if is_land(add(start, UNIT_VECTORS[candidate])) and not is_land(
            add(start, UNIT_VECTORS[previous])
        ):
            direction = candidate
            break

    if direction is None:
        raise ValueError("could not determine an initial coast-following direction")

    total_land_tiles = sum(
        1 for tile in all_tiles.values() if isinstance(tile, LandTile)
    )
    visited = set()
    coord = start

    while len(visited) < total_land_tiles:
        if coord not in visited:
            visited.add(coord)
            yield coord

        next_coord = add(coord, UNIT_VECTORS[direction])
        if is_land(next_coord) and next_coord not in visited:
            coord = next_coord
            continue

        direction = directions[(directions.index(direction) + 1) % len(directions)]

def cube_radius(coord: Coordinate) -> int:
    return max(abs(component) for component in coord)

def ring_coordinates(radius: int) -> tuple[Coordinate, ...]:
    if radius == 0:
        return ((0, 0, 0),)

    coord = (radius, -radius, 0)
    ring = []
    for direction in COUNTERCLOCKWISE_RING_DIRECTIONS:
        for _ in range(radius):
            ring.append(coord)
            coord = add(coord, UNIT_VECTORS[direction])
    return tuple(ring)

def  outer_land_coordinates(
    all_tiles: Mapping[Coordinate, Tile],
) -> tuple[Coordinate, ...]:
    """Return outer-ring land coordinates in deterministic coast-following order."""
    land_coords = {
        coord for coord, tile in all_tiles.items() if isinstance(tile, LandTile)
    }
    if not land_coords:
        return tuple()

    radius = max(cube_radius(coord) for coord in land_coords)
    return tuple(coord for coord in ring_coordinates(radius) if coord in land_coords)


def get_starting_spiral_coordinates(all_tiles: Dict[Coordinate, Tile]) -> Coordinate:
    """Return a randomly chosen corner coordinate from the outer ring of land coordinates."""
    outer_ring = outer_land_coordinates(all_tiles)

    corners = [coord for coord in outer_ring if 0 in coord]
    return random.choice(corners)