from catanatron.models.spiral import get_starting_spiral_coordinates
import pytest

from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    LandTile,
    initialize_tiles,
)
from catanatron.models.spiral import spiral_land_coordinates


def test_spiral_land_coordinates_rejects_non_land_start():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")

    with pytest.raises(ValueError, match="start must be a land tile"):
        list(spiral_land_coordinates(all_tiles, (2, -3, 1)))


def test_spiral_land_coordinates_rejects_non_edge_base_land_start():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")

    with pytest.raises(ValueError, match="start must be on the outer edge"):
        list(spiral_land_coordinates(all_tiles, (0, 0, 0)))


def test_spiral_land_coordinates_rejects_non_edge_mini_land_start():
    all_tiles = initialize_tiles(MINI_MAP_TEMPLATE, number_placement="random")

    with pytest.raises(ValueError, match="start must be on the outer edge"):
        list(spiral_land_coordinates(all_tiles, (0, 0, 0)))


def test_spiral_land_coordinates_base_map_order_from_tile_seven():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")

    ids = [
        all_tiles[coord].id for coord in spiral_land_coordinates(all_tiles, (2, -2, 0))
    ]

    assert ids == [7, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 1, 6, 5, 4, 3, 2, 0]


def test_spiral_land_coordinates_mini_map_order_from_tile_one():
    all_tiles = initialize_tiles(MINI_MAP_TEMPLATE, number_placement="random")
    start = next(
        coord
        for coord, tile in all_tiles.items()
        if isinstance(tile, LandTile) and tile.id == 1
    )

    ids = [all_tiles[coord].id for coord in spiral_land_coordinates(all_tiles, start)]

    assert ids == [1, 6, 5, 4, 3, 2, 0]


def test_spiral_starts_at_random_corners():
    """Prove that the spiral starting coordinate is a randomly chosen corner."""

    # We only need to generate the board layout once to test the coordinate picker
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")
    starts = set()

    for _ in range(50):
        # Directly call the function we modified to see what it picks
        start_coord = get_starting_spiral_coordinates(all_tiles)

        # 1. Prove it is a corner (cube coordinates for corners always contain a 0)
        assert 0 in start_coord

        starts.add(start_coord)

    # 2. Prove it picked more than one unique corner over 50 calls
    assert len(starts) > 1
