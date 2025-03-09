import logging
import sys
import os
from typing import List, Optional, Any, Dict, Tuple, Union

# Add all necessary paths to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_experimental'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_rust'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Rust implementation
try:
    from catanatron_rust import Game as RustGame
    RUST_AVAILABLE = True
    logger.info("Rust backend available and loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    RustGame = None  # Create a placeholder
    logger.warning(f"Rust backend not available: {str(e)}")
    # Try the import again with a different path strategy
    try:
        # Try direct relative import
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rust_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'catanatron_rust')
        if rust_dir not in sys.path:
            sys.path.insert(0, rust_dir)
        from catanatron_rust import Game as RustGame
        RUST_AVAILABLE = True
        logger.info("Rust backend available through alternate path")
    except ImportError as e:
        logger.warning(f"Second attempt to import Rust backend failed: {str(e)}")

def is_rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE

def create_game(players, use_rust=False, **kwargs):
    """
    Factory function to create either a Python or Rust game.
    
    Args:
        players: List of player objects
        use_rust: Whether to use Rust backend if available
        **kwargs: Additional game configuration parameters
    
    Returns:
        A Game instance (either Python or Rust-backed)
    """
    if use_rust and RUST_AVAILABLE:
        logger.info("Creating game with Rust backend")
        
        # The Rust Game expects players to have a 'color' attribute
        # We need to ensure the color is properly set for the Rust backend
        processed_players = []
        for player in players:
            # Check if player already has _rust_color attribute (added by prepare_players_for_rust)
            if hasattr(player, '_rust_color'):
                # Ensure the player also has a color attribute that the Rust backend expects
                if not hasattr(player, 'color'):
                    # Map the numeric _rust_color back to a color object if needed
                    from catanatron.models.player import Color
                    color_map = {0: Color.RED, 1: Color.BLUE, 2: Color.ORANGE, 3: Color.WHITE}
                    player.color = color_map.get(player._rust_color)
                    logger.info(f"Added color={player.color} attribute based on _rust_color={player._rust_color}")
                processed_players.append(player)
                logger.info(f"Using pre-processed player with _rust_color={player._rust_color}")
            # Check if player has a color attribute
            elif hasattr(player, 'color'):
                # Convert Color enum to numeric value for Rust
                # Rust expects colors as integers: RED=0, BLUE=1, ORANGE=2, WHITE=3
                color_value = None
                if hasattr(player.color, 'value'):
                    # Handle Python Color enum which has a value attribute
                    color_name = str(player.color.name)
                    if color_name == 'RED':
                        color_value = 0
                    elif color_name == 'BLUE': 
                        color_value = 1
                    elif color_name == 'ORANGE':
                        color_value = 2
                    elif color_name == 'WHITE':
                        color_value = 3
                else:
                    # If color is already a numeric value, use it
                    try:
                        color_value = int(player.color)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert player color {player.color} to numeric value")
                
                # Create a wrapper for the player if needed
                if color_value is not None:
                    # Attach the numeric color value to the player for Rust
                    player._rust_color = color_value
                    processed_players.append(player)
                else:
                    logger.warning(f"Skipping player with invalid color: {player.color}")
            else:
                logger.warning(f"Player {player} does not have a color attribute")
        
        logger.info(f"Processed {len(processed_players)} players for Rust backend")
        
        # Now try to create the RustGame with the processed players
        try:
            return RustGame(processed_players)
        except Exception as e:
            logger.error(f"Failed to create RustGame: {e}")
            logger.error("Falling back to Python backend")
            from catanatron.game import Game
            return Game(players, **kwargs)
    else:
        if use_rust and not RUST_AVAILABLE:
            logger.warning("Rust backend requested but not available. Using Python backend.")
        from catanatron.game import Game
        return Game(players, **kwargs)

class RustAccumulatorAdapter:
    """
    Adapter for Python accumulators to work with Rust backend.
    
    This translates between Rust and Python representations
    of game state and actions for each accumulator method.
    """
    def __init__(self, python_accumulator):
        self.accumulator = python_accumulator
    
    def before(self, state=None):
        """Called before game simulation starts."""
        # Convert state if needed
        return self.accumulator.before(state)
    
    def step(self, state=None, action=None):
        """Called after each game action."""
        # Convert state and action if needed
        return self.accumulator.step(state, action)
    
    def after(self, state=None):
        """Called after game simulation ends."""
        # Convert state if needed
        return self.accumulator.after(state)

def adapt_accumulators_for_backend(accumulators, use_rust):
    """
    Adapt accumulators to work with the specified backend.
    
    Args:
        accumulators: List of accumulator objects
        use_rust: Whether we're using the Rust backend
    
    Returns:
        List of adapted accumulators
    """
    if not accumulators:
        return []
        
    if use_rust and RUST_AVAILABLE:
        return [RustAccumulatorAdapter(acc) for acc in accumulators]
    else:
        return accumulators 