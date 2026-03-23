import pytest

from catanatron.models.coordinate_system import Direction, UNIT_VECTORS, add
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    LandTile,
    initialize_tiles,
)
from catanatron.models.spiral import outer_land_coordinates, spiral_land_coordinates


def test_outer_land_coordinates_base_map():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")

    assert outer_land_coordinates(all_tiles) == (
        (2, -2, 0),
        (2, -1, -1),
        (2, 0, -2),
        (1, 1, -2),
        (0, 2, -2),
        (-1, 2, -1),
        (-2, 2, 0),
        (-2, 1, 1),
        (-2, 0, 2),
        (-1, -1, 2),
        (0, -2, 2),
        (1, -2, 1),
    )


def test_outer_land_coordinates_mini_map():
    all_tiles = initialize_tiles(MINI_MAP_TEMPLATE, number_placement="random")

    assert outer_land_coordinates(all_tiles) == (
        (1, -1, 0),
        (1, 0, -1),
        (0, 1, -1),
        (-1, 1, 0),
        (-1, 0, 1),
        (0, -1, 1),
    )


def test_outer_land_coordinates_excludes_interior_land_tiles():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")
    outer_coords = outer_land_coordinates(all_tiles)

    assert (0, 0, 0) not in outer_coords
    assert all(isinstance(all_tiles[coord], LandTile) for coord in outer_coords)
    assert all(
        any(
            not isinstance(all_tiles.get(add(coord, UNIT_VECTORS[direction])), LandTile)
            for direction in Direction
        )
        for coord in outer_coords
    )


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


def test_spiral_land_coordinates_accepts_all_base_outer_starts():
    all_tiles = initialize_tiles(BASE_MAP_TEMPLATE, number_placement="random")

    for start in outer_land_coordinates(all_tiles):
        coords = tuple(spiral_land_coordinates(all_tiles, start))

        assert len(coords) == 19
        assert len(set(coords)) == 19
        assert coords[0] == start
        assert coords[-1] == (0, 0, 0)


def test_spiral_land_coordinates_accepts_all_mini_outer_starts():
    all_tiles = initialize_tiles(MINI_MAP_TEMPLATE, number_placement="random")

    for start in outer_land_coordinates(all_tiles):
        coords = tuple(spiral_land_coordinates(all_tiles, start))

        assert len(coords) == 7
        assert len(set(coords)) == 7
        assert coords[0] == start
        assert coords[-1] == (0, 0, 0)
