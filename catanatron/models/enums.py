from enum import Enum
from collections import namedtuple


class Resource(Enum):
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"

    def __repr__(self) -> str:
        return self.value


class DevelopmentCard(Enum):
    KNIGHT = "KNIGHT"
    YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
    MONOPOLY = "MONOPOLY"
    ROAD_BUILDING = "ROAD_BUILDING"
    VICTORY_POINT = "VICTORY_POINT"


class BuildingType(Enum):
    SETTLEMENT = "SETTLEMENT"
    CITY = "CITY"
    ROAD = "ROAD"
