from catanatron.coordinate_system import num_tiles_for, generate_coordinate_system


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
