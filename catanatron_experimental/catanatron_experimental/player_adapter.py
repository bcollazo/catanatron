"""
Player adapter classes for cross-backend compatibility.

This module provides adapter classes to make players compatible
with both Python and Rust backends.
"""

import logging
from typing import Dict, Any, List, Optional

from catanatron.models.player import Player

logger = logging.getLogger(__name__)

class RustCompatibleMixin:
    """
    Mixin to add Rust compatibility to any player class.
    
    Adds the necessary attributes and methods for a player
    to work with the Rust backend.
    """
    
    def __init__(self, *args, **kwargs):
        # Call the parent init
        super().__init__(*args, **kwargs)
        
        # Mark this player as Rust-compatible
        self.is_rust_player = True
        
        # Add Rust-specific color mapping
        if hasattr(self, 'color'):
            self._setup_rust_color()
    
    def _setup_rust_color(self):
        """Map color to numeric value for Rust"""
        color_map = {
            "RED": 0,
            "BLUE": 1,
            "ORANGE": 2,
            "WHITE": 3,
        }
        
        color_name = str(self.color.name) if hasattr(self.color, 'name') else str(self.color)
        if color_name in color_map:
            self._rust_color = color_map[color_name]
            logger.debug(f"Mapped color {color_name} to _rust_color={self._rust_color}")
        else:
            logger.warning(f"Unknown color {color_name}, Rust compatibility may be affected")
            
    def get_rust_color(self):
        """Get the Rust numeric color value."""
        if hasattr(self, '_rust_color'):
            return self._rust_color
        return None

def make_rust_compatible(player_class):
    """
    Factory function to create a Rust-compatible version of a player class.
    
    Args:
        player_class: The player class to make Rust-compatible
        
    Returns:
        A new class that inherits from both RustCompatibleMixin and the original player class
    """
    class_name = f"Rust{player_class.__name__}"
    
    # Create a new class that inherits from both RustCompatibleMixin and the original class
    new_class = type(
        class_name,
        (RustCompatibleMixin, player_class),
        {
            "__doc__": f"Rust-compatible version of {player_class.__name__}",
            "__module__": player_class.__module__,
        }
    )
    
    return new_class 