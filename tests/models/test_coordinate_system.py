from catanatron.models.coordinate_system import (
    offset_to_cube,
)


def test_offset_to_cube():
    assert offset_to_cube((0, 0)) == (0, 0, 0)
    assert offset_to_cube((1, 0)) == (1, -1, 0)
    assert offset_to_cube((-1, -1)) == (0, 1, -1)
    assert offset_to_cube((1, 2)) == (0, -2, 2)
