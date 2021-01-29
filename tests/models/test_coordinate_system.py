from catanatron.models.coordinate_system import (
    cube_to_axial,
    cube_to_offset,
    offset_to_cube,
    num_tiles_for,
    generate_coordinate_system,
)


def test_num_tiles_for():
    assert num_tiles_for(0) == 1
    assert num_tiles_for(1) == 7
    assert num_tiles_for(2) == 19
    assert num_tiles_for(3) == 37


def test_generate_coordinate_system():
    assert generate_coordinate_system(0) == set([(0, 0, 0)])


def test_generate_coordinate_system_one_layer():
    assert generate_coordinate_system(1) == set(
        [
            (0, 0, 0),
            (1, -1, 0),
            (0, -1, 1),
            (-1, 0, 1),
            (-1, 1, 0),
            (0, 1, -1),
            (1, 0, -1),
        ]
    )


def test_generate_coordinate_system_two_layer():
    assert generate_coordinate_system(2) == set(
        [
            (0, 0, 0),  # center
            # first layer
            (1, -1, 0),
            (0, -1, 1),
            (-1, 0, 1),
            (-1, 1, 0),
            (0, 1, -1),
            (1, 0, -1),
            # second layer
            (2, -2, 0),
            (1, -2, 1),
            (0, -2, 2),
            (-1, -1, 2),
            (-2, 0, 2),
            (-2, 1, 1),
            (-2, 2, 0),  # westmost
            (-1, 2, -1),
            (0, 2, -2),
            (1, 1, -2),
            (2, 0, -2),
            (2, -1, -1),
        ]
    )


def test_cube_to_axial():
    assert cube_to_axial((0, 0, 0)) == (0, 0)
    assert cube_to_axial((2, 0, -2)) == (2, -2)
    assert cube_to_axial((0, 1, -1)) == (0, -1)


def test_cube_to_offset():
    assert cube_to_offset((0, 0, 0)) == (0, 0)
    assert cube_to_offset((1, -1, 0)) == (1, 0)
    assert cube_to_offset((0, 1, -1)) == (-1, -1)
    assert cube_to_offset((0, -2, 2)) == (1, 2)


def test_offset_to_cube():
    assert offset_to_cube((0, 0)) == (0, 0, 0)
    assert offset_to_cube((1, 0)) == (1, -1, 0)
    assert offset_to_cube((-1, -1)) == (0, 1, -1)
    assert offset_to_cube((1, 2)) == (0, -2, 2)
