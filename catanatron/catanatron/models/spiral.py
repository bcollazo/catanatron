from collections.abc import Generator
from typing import Mapping

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


def outer_land_coordinates(
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


def spiral_land_coordinates(
    all_tiles: Mapping[Coordinate, Tile], start: Coordinate
) -> Generator[Coordinate, None, None]:
    """
    Yield land-tile coordinates ring by ring, from the outer ring to the center.

    The walk is based on cube-coordinate radius, not coastline tracing: it rotates
    each ring so it starts at ``start`` and then steps inward to the next smaller
    ring until reaching the center.

    This works for the standard convex Catan layouts, including the base and mini
    boards. It is not intended for land masses with holes or interior water tiles,
    because the algorithm assumes each radius forms a continuous ring of land.

    Raises:
    - ValueError: If ``start`` is not land.
    - ValueError: If ``start`` is not on the outer ring.
    """

    def is_land(coord: Coordinate) -> bool:
        return isinstance(all_tiles.get(coord), LandTile)

    if not is_land(start):
        raise ValueError("start must be a land tile")

    if all(is_land(add(start, UNIT_VECTORS[direction])) for direction in Direction):
        raise ValueError("start must be on the outer edge of the land mass")

    # The spiral is defined by concentric cube-coordinate rings of land tiles.
    land_coords = {
        coord for coord, tile in all_tiles.items() if isinstance(tile, LandTile)
    }
    max_radius = max(cube_radius(coord) for coord in land_coords)
    if start not in ring_coordinates(max_radius):
        raise ValueError("could not determine an initial coast-following direction")

    current_start = start
    for radius in range(max_radius, 0, -1):
        ring = tuple(
            coord for coord in ring_coordinates(radius) if coord in land_coords
        )
        # Rotate each ring so traversal begins at the requested start for that ring.
        start_index = ring.index(current_start)
        yield from ring[start_index:]
        yield from ring[:start_index]

        if radius > 1:
            inner_ring = tuple(
                coord for coord in ring_coordinates(radius - 1) if coord in land_coords
            )
            # Project the current ring index inward to the matching inner-ring index.
            # Example: on the base map, outer index 0..11 maps to inner index 0..5,
            # so outer 0/1 -> inner 0, outer 2/3 -> inner 1, and so on.
            current_start = inner_ring[start_index * (radius - 1) // radius]

    # Emit the center after all non-zero-radius rings have been walked.
    if (0, 0, 0) in land_coords:
        yield (0, 0, 0)
