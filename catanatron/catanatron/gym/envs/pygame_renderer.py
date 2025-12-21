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


# Base constants (scale with render_scale)
HEX_SIZE = 70  # pixels (balanced for both MINI and BASE maps)
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
ROAD_WIDTH = 6  # Scaled proportionally
SETTLEMENT_RADIUS = 12  # Scaled proportionally
CITY_RADIUS = 18  # Scaled proportionally

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
NUMBER_TOKEN_COLOR = (245, 222, 179)  # beige/wheat color like physical game
RED_NUMBER_COLOR = (200, 0, 0)  # red for 6 and 8


class PygameRenderer:
    """
    Renders the Catan board using pygame.

    Returns numpy arrays (rgb_array) for compatibility with gymnasium's RecordVideo.
    """

    def __init__(self, render_scale: float = 1.0):
        pygame.init()
        pygame.font.init()

        self.render_scale = max(1.0, float(render_scale))
        self.base_width = SCREEN_WIDTH
        self.base_height = SCREEN_HEIGHT
        self.render_width = int(self.base_width * self.render_scale)
        self.render_height = int(self.base_height * self.render_scale)

        # Scaled sizing for higher-res rendering
        self.hex_size = int(HEX_SIZE * self.render_scale)
        self.road_width = max(1, int(ROAD_WIDTH * self.render_scale))
        self.settlement_radius = int(SETTLEMENT_RADIUS * self.render_scale)
        self.city_radius = int(CITY_RADIUS * self.render_scale)
        self.outline_width = max(1, int(2 * self.render_scale))

        # Create surface for rendering (headless)
        self.surface = pygame.Surface((self.render_width, self.render_height))
        self.font = pygame.font.Font(None, int(32 * self.render_scale))

        # Center offset to position the map in the middle of the screen
        self.center_x = self.render_width // 2
        self.center_y = self.render_height // 2

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
        px = self.hex_size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        py = self.hex_size * (3 / 2 * r)

        # Apply center offset
        px += self.center_x
        py += self.center_y

        return (px, py)

    def hexagon_corners(self, center: Tuple[float, float], size: float) -> list:
        """Get the 6 corner points of a pointy-top hexagon.

        Args:
            center: Center position (x, y)
            size: Radius of the hexagon

        Returns:
            List of 6 corner points
        """
        cx, cy = center
        corners = []
        for i in range(6):
            # Pointy-top: vertices at 30°, 90°, 150°, 210°, 270°, 330°
            angle = math.pi / 6 + math.pi / 3 * i
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
            pygame.draw.polygon(self.surface, OUTLINE_COLOR, corners, self.outline_width)

    def get_number_pips(self, number: int) -> int:
        """Get the number of probability pips for a dice number.

        Args:
            number: Dice number (2-12)

        Returns:
            Number of pips to display
        """
        pips = {
            2: 1, 12: 1,
            3: 2, 11: 2,
            4: 3, 10: 3,
            5: 4, 9: 4,
            6: 5, 8: 5,
        }
        return pips.get(number, 0)

    def draw_number_token(self, center: Tuple[float, float], number: int):
        """Draw a number token like in the physical Catan game.

        Args:
            center: Center position (x, y)
            number: The dice number (2-12)
        """
        # Token dimensions (scaled proportionally with HEX_SIZE)
        token_radius = int(30 * self.render_scale)

        # Determine if this is a red number (6 or 8)
        is_red = number in [6, 8]
        number_color = RED_NUMBER_COLOR if is_red else TEXT_COLOR

        # Draw token circle (beige background with black border)
        pygame.draw.circle(self.surface, NUMBER_TOKEN_COLOR,
                          (int(center[0]), int(center[1])), token_radius)
        pygame.draw.circle(self.surface, OUTLINE_COLOR,
                          (int(center[0]), int(center[1])), token_radius, self.outline_width)

        # Draw the number
        number_font = pygame.font.Font(None, int(38 * self.render_scale))
        text = number_font.render(str(number), True, number_color)
        text_rect = text.get_rect(center=(int(center[0]), int(center[1] - int(6 * self.render_scale))))
        self.surface.blit(text, text_rect)

        # Draw pips below the number
        num_pips = self.get_number_pips(number)
        if num_pips > 0:
            pip_size = max(1, int(3 * self.render_scale))
            pip_spacing = int(5 * self.render_scale)
            total_width = num_pips * pip_size + (num_pips - 1) * pip_spacing
            start_x = center[0] - total_width / 2 + pip_size / 2
            pip_y = center[1] + int(10 * self.render_scale)

            for i in range(num_pips):
                pip_x = start_x + i * (pip_size + pip_spacing)
                pygame.draw.circle(self.surface, number_color,
                                 (int(pip_x), int(pip_y)), pip_size)

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
        self.draw_hexagon(center, self.hex_size, resource_color, outline=True)

        # Draw number token if not desert
        if tile.number is not None:
            self.draw_number_token(center, tile.number)

        # Draw robber if present
        if coord == robber_coord:
            self.draw_robber(center)

    def draw_robber(self, center: Tuple[float, float]):
        """Draw the robber as a black circle.

        Args:
            center: Center position (x, y)
        """
        robber_radius = int(18 * self.render_scale)
        pygame.draw.circle(self.surface, ROBBER_COLOR, (int(center[0]), int(center[1])), robber_radius)
        # White outline to make it visible on dark tiles
        pygame.draw.circle(
            self.surface,
            (255, 255, 255),
            (int(center[0]), int(center[1])),
            robber_radius,
            self.outline_width,
        )

    def get_node_delta(self, direction: str, size: float) -> Tuple[float, float]:
        """Get the offset from tile center to node based on direction.

        Matches the frontend getNodeDelta function.

        Args:
            direction: NodeRef direction (NORTH, NORTHEAST, etc.)
            size: Hexagon size

        Returns:
            (delta_x, delta_y) offset from tile center
        """
        w = math.sqrt(3) * size  # SQRT3 * size
        h = 2 * size

        deltas = {
            "NORTH": (0, -h / 2),
            "NORTHEAST": (w / 2, -h / 4),
            "SOUTHEAST": (w / 2, h / 4),
            "SOUTH": (0, h / 2),
            "SOUTHWEST": (-w / 2, h / 4),
            "NORTHWEST": (-w / 2, -h / 4),
        }
        return deltas.get(direction, (0, 0))

    def get_node_pixel_position(self, node_id: int, game: Game) -> Tuple[float, float]:
        """Get pixel position for a node.

        Args:
            node_id: Node ID
            game: Game object

        Returns:
            Pixel position (x, y)
        """
        board = game.state.board

        # Find a tile that contains this node and get its direction
        for coord, tile in board.map.land_tiles.items():
            for node_ref, nid in tile.nodes.items():
                if nid == node_id:
                    # Found the tile and direction
                    tile_center = self.cube_to_pixel(coord)
                    delta = self.get_node_delta(node_ref.value, self.hex_size)
                    return (tile_center[0] + delta[0], tile_center[1] + delta[1])

        return (0, 0)

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
            # Draw settlement as a small square
            size = self.settlement_radius * 2
            rect = pygame.Rect(int(pos[0] - size/2), int(pos[1] - size/2), size, size)
            pygame.draw.rect(self.surface, player_color, rect)
            pygame.draw.rect(self.surface, OUTLINE_COLOR, rect, self.outline_width)
        elif building_type == CITY:
            # Draw city as a larger square
            size = self.city_radius * 2
            rect = pygame.Rect(int(pos[0] - size/2), int(pos[1] - size/2), size, size)
            pygame.draw.rect(self.surface, player_color, rect)
            pygame.draw.rect(self.surface, OUTLINE_COLOR, rect, self.outline_width)

    def draw_edge(self, edge_id: Tuple[int, int], color: Color, game: Game):
        """Draw a road along an edge.

        Args:
            edge_id: Edge ID (node_id_1, node_id_2)
            color: Player color
            game: Game object
        """
        # Get positions of both nodes that define this edge
        node1_pos = self.get_node_pixel_position(edge_id[0], game)
        node2_pos = self.get_node_pixel_position(edge_id[1], game)

        # Check if either position is invalid
        if node1_pos == (0, 0) or node2_pos == (0, 0):
            return

        player_color = PLAYER_COLORS.get(color, (128, 128, 128))
        pygame.draw.line(
            self.surface,
            player_color,
            (int(node1_pos[0]), int(node1_pos[1])),
            (int(node2_pos[0]), int(node2_pos[1])),
            self.road_width,
        )

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
        if self.render_scale > 1.0:
            output_surface = pygame.transform.smoothscale(
                self.surface, (self.base_width, self.base_height)
            )
        else:
            output_surface = self.surface

        array = pygame.surfarray.array3d(output_surface)
        array = np.transpose(array, (1, 0, 2))  # Transpose to (height, width, channels)

        return array

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
