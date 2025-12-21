"""
Pygame renderer for Catanatron environment.

Renders the Catan board using hexagonal tiles in a minimalist style.
"""

import math
from typing import Tuple, Dict
import numpy as np
import pygame

from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, SETTLEMENT, CITY
from catanatron.models.player import Color
from catanatron.game import Game


# Constants
HEX_SIZE = 50  # pixels
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
ROAD_WIDTH = 5
SETTLEMENT_RADIUS = 10
CITY_RADIUS = 15

# Colors (RGB)
COLORS = {
    WOOD: (34, 139, 34),  # forest green
    BRICK: (178, 34, 34),  # brick red
    SHEEP: (144, 238, 144),  # light green
    WHEAT: (255, 215, 0),  # gold
    ORE: (128, 128, 128),  # gray
    None: (194, 178, 128),  # tan (desert)
}

PLAYER_COLORS = {
    Color.BLUE: (0, 0, 255),
    Color.RED: (255, 0, 0),
    Color.ORANGE: (255, 165, 0),
    Color.WHITE: (255, 255, 255),
}

BACKGROUND_COLOR = (240, 240, 240)  # light gray
OUTLINE_COLOR = (0, 0, 0)  # black
ROBBER_COLOR = (0, 0, 0)  # black
TEXT_COLOR = (0, 0, 0)  # black


class PygameRenderer:
    """
    Renders the Catan board using pygame.

    Returns numpy arrays (rgb_array) for compatibility with gymnasium's RecordVideo.
    """

    def __init__(self):
        pygame.init()
        pygame.font.init()

        # Create surface for rendering (headless)
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 32)  # Default font, size 32

        # Center offset to position the map in the middle of the screen
        self.center_x = SCREEN_WIDTH // 2
        self.center_y = SCREEN_HEIGHT // 2

    def cube_to_pixel(self, coord: Tuple[int, int, int]) -> Tuple[float, float]:
        """Convert cube coordinate to pixel position (flat-top hexagon).

        Args:
            coord: Cube coordinate (x, y, z) where x + y + z = 0

        Returns:
            Pixel position (px, py)
        """
        x, y, z = coord
        # Convert cube to axial
        q = x
        r = z
        # Axial to pixel (flat-top hexagon)
        px = HEX_SIZE * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        py = HEX_SIZE * (3 / 2 * r)

        # Apply center offset
        px += self.center_x
        py += self.center_y

        return (px, py)

    def hexagon_corners(self, center: Tuple[float, float], size: float) -> list:
        """Get the 6 corner points of a flat-top hexagon.

        Args:
            center: Center position (x, y)
            size: Radius of the hexagon

        Returns:
            List of 6 corner points
        """
        cx, cy = center
        corners = []
        for i in range(6):
            angle = math.pi / 3 * i
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            corners.append((x, y))
        return corners

    def draw_hexagon(self, center: Tuple[float, float], size: float,
                    fill_color: Tuple[int, int, int], outline: bool = True):
        """Draw a hexagon on the surface.

        Args:
            center: Center position (x, y)
            size: Radius of the hexagon
            fill_color: RGB color for fill
            outline: Whether to draw black outline
        """
        corners = self.hexagon_corners(center, size)
        pygame.draw.polygon(self.surface, fill_color, corners)
        if outline:
            pygame.draw.polygon(self.surface, OUTLINE_COLOR, corners, 2)

    def draw_tile(self, coord: Tuple[int, int, int], tile, robber_coord: Tuple[int, int, int]):
        """Draw a single land tile.

        Args:
            coord: Tile coordinate (x, y, z)
            tile: LandTile object
            robber_coord: Coordinate of the robber
        """
        center = self.cube_to_pixel(coord)

        # Draw hexagon with resource color
        resource_color = COLORS.get(tile.resource, COLORS[None])
        self.draw_hexagon(center, HEX_SIZE, resource_color, outline=True)

        # Draw number if not desert
        if tile.number is not None:
            text = self.font.render(str(tile.number), True, TEXT_COLOR)
            text_rect = text.get_rect(center=center)
            self.surface.blit(text, text_rect)

        # Draw robber if present
        if coord == robber_coord:
            self.draw_robber(center)

    def draw_robber(self, center: Tuple[float, float]):
        """Draw the robber as a black circle.

        Args:
            center: Center position (x, y)
        """
        pygame.draw.circle(self.surface, ROBBER_COLOR, (int(center[0]), int(center[1])), 15)
        # White outline to make it visible on dark tiles
        pygame.draw.circle(self.surface, (255, 255, 255), (int(center[0]), int(center[1])), 15, 2)

    def get_node_pixel_position(self, node_id: int, game: Game) -> Tuple[float, float]:
        """Get pixel position for a node.

        Args:
            node_id: Node ID
            game: Game object

        Returns:
            Pixel position (x, y)
        """
        # Find which tiles contain this node and average their positions
        board = game.state.board
        adjacent_tiles = board.map.adjacent_tiles.get(node_id, [])

        if not adjacent_tiles:
            return (0, 0)

        # Get coordinates of adjacent tiles
        tile_coords = []
        for tile in adjacent_tiles:
            # Find the coordinate of this tile in the land_tiles dict
            for coord, land_tile in board.map.land_tiles.items():
                if land_tile.id == tile.id:
                    tile_coords.append(coord)
                    break

        # Average the pixel positions of adjacent tiles
        if not tile_coords:
            return (0, 0)

        pixel_positions = [self.cube_to_pixel(coord) for coord in tile_coords]
        avg_x = sum(p[0] for p in pixel_positions) / len(pixel_positions)
        avg_y = sum(p[1] for p in pixel_positions) / len(pixel_positions)

        return (avg_x, avg_y)

    def draw_node(self, node_id: int, color: Color, building_type: str, game: Game):
        """Draw a settlement or city at a node.

        Args:
            node_id: Node ID
            color: Player color
            building_type: "SETTLEMENT" or "CITY"
            game: Game object
        """
        pos = self.get_node_pixel_position(node_id, game)
        player_color = PLAYER_COLORS.get(color, (128, 128, 128))

        if building_type == SETTLEMENT:
            pygame.draw.circle(self.surface, player_color,
                             (int(pos[0]), int(pos[1])), SETTLEMENT_RADIUS)
            pygame.draw.circle(self.surface, OUTLINE_COLOR,
                             (int(pos[0]), int(pos[1])), SETTLEMENT_RADIUS, 2)
        elif building_type == CITY:
            # Draw city as a larger circle
            pygame.draw.circle(self.surface, player_color,
                             (int(pos[0]), int(pos[1])), CITY_RADIUS)
            pygame.draw.circle(self.surface, OUTLINE_COLOR,
                             (int(pos[0]), int(pos[1])), CITY_RADIUS, 2)

    def draw_edge(self, edge_id: Tuple[int, int], color: Color, game: Game):
        """Draw a road along an edge.

        Args:
            edge_id: Edge ID (node_id_1, node_id_2)
            color: Player color
            game: Game object
        """
        node1_pos = self.get_node_pixel_position(edge_id[0], game)
        node2_pos = self.get_node_pixel_position(edge_id[1], game)

        player_color = PLAYER_COLORS.get(color, (128, 128, 128))
        pygame.draw.line(self.surface, player_color,
                        (int(node1_pos[0]), int(node1_pos[1])),
                        (int(node2_pos[0]), int(node2_pos[1])),
                        ROAD_WIDTH)

    def render(self, game: Game) -> np.ndarray:
        """Render the game state and return as numpy array.

        Args:
            game: Game object

        Returns:
            RGB array (height, width, 3) for gymnasium RecordVideo
        """
        # Clear surface
        self.surface.fill(BACKGROUND_COLOR)

        # Get board state
        board = game.state.board

        # Draw all land tiles
        for coord, tile in board.map.land_tiles.items():
            self.draw_tile(coord, tile, board.robber_coordinate)

        # Draw all roads
        for edge_id, color in board.roads.items():
            self.draw_edge(edge_id, color, game)

        # Draw all buildings (settlements and cities)
        for node_id, (color, building_type) in board.buildings.items():
            self.draw_node(node_id, color, building_type, game)

        # Convert surface to numpy array
        # pygame.surfarray.array3d returns (width, height, 3), we need (height, width, 3)
        array = pygame.surfarray.array3d(self.surface)
        array = np.transpose(array, (1, 0, 2))  # Transpose to (height, width, channels)

        return array

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
