from collections.abc import Generator
from typing import Mapping

from catanatron.models.coordinate_system import Direction, UNIT_VECTORS, add, Coordinate
from catanatron.models.tiles import LandTile, Tile


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
