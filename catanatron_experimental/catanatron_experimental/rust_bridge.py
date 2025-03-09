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
logger.setLevel(logging.INFO)  # Set this module's logger to INFO level

# Initialize flags
RUST_AVAILABLE = False
RustGame = None

# Try to import Rust implementation with a more flexible approach
try:
    import catanatron_rust
    logger.info("Base catanatron_rust module imported successfully")
    
    # Try different approaches to find the Game class
    # First, try direct import from the top-level module
    try:
        if hasattr(catanatron_rust, 'Game'):
            RustGame = catanatron_rust.Game
            RUST_AVAILABLE = True
            logger.info("Found Game class at top level in catanatron_rust")
        # Next, try the python submodule if it exists
        elif hasattr(catanatron_rust, 'python') and hasattr(catanatron_rust.python, 'Game'):
            RustGame = catanatron_rust.python.Game
            RUST_AVAILABLE = True
            logger.info("Found Game class in catanatron_rust.python")
        # Try one more approach with explicit import
        else:
            try:
                from catanatron_rust.python import Game as RustGame
                RUST_AVAILABLE = True
                logger.info("Successfully imported Game from catanatron_rust.python")
            except ImportError as e:
                logger.warning(f"Could not import Game class: {e}")
                # As a last resort, look at all available attributes
                logger.info(f"Available attributes in catanatron_rust: {dir(catanatron_rust)}")
                if hasattr(catanatron_rust, '__file__'):
                    logger.info(f"catanatron_rust module location: {catanatron_rust.__file__}")
    except Exception as e:
        logger.warning(f"Error finding Game class: {e}")
        logger.info(f"Available attributes in catanatron_rust: {dir(catanatron_rust)}")
        if hasattr(catanatron_rust, '__file__'):
            logger.info(f"catanatron_rust module location: {catanatron_rust.__file__}")

except ImportError as e:
    logger.warning(f"Base catanatron_rust module import failed: {e}")
    
    # Try an alternative import path
    try:
        # Add the path to the compiled extension
        rust_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'catanatron_rust')
        target_debug = os.path.join(rust_dir, 'target', 'debug')
        
        if os.path.exists(target_debug):
            logger.info(f"Adding {target_debug} to Python path")
            if target_debug not in sys.path:
                sys.path.insert(0, target_debug)
        
        try:
            import catanatron_rust
            logger.info("Base catanatron_rust module imported through alternate path")
            
            # Try both ways again
            if hasattr(catanatron_rust, 'Game'):
                RustGame = catanatron_rust.Game
                RUST_AVAILABLE = True
                logger.info("Found Game class at top level in alternate path")
            elif hasattr(catanatron_rust, 'python') and hasattr(catanatron_rust.python, 'Game'):
                RustGame = catanatron_rust.python.Game
                RUST_AVAILABLE = True
                logger.info("Found Game class in python submodule in alternate path")
            else:
                logger.warning("Game class not found in alternate path")
                
        except ImportError as e:
            logger.warning(f"Alternate import path failed: {e}")
    except Exception as e:
        logger.warning(f"Error during alternate import attempt: {e}")

# Print Rust availability for debugging
print(f"Rust backend available: {RUST_AVAILABLE}")

def is_rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE

def create_game(players, use_rust=False, strict_rust=False, **kwargs):
    """
    Create a game with either Python or Rust backend
    
    Args:
        players: List of Player objects
        use_rust: Whether to use Rust backend if available
        strict_rust: Whether to strictly enforce Rust compatibility
        **kwargs: Additional game configuration parameters
    
    Returns:
        A Game instance (either Python or Rust-backed)
        
    Raises:
        ImportError: If requested backend is not available
        TypeError: If any player is fundamentally incompatible with the Rust backend
        RuntimeError: If there's a failure in creating a Rust game
    """
    # If Rust is not available but requested, fail immediately
    if use_rust and not is_rust_available():
        raise ImportError("Rust backend requested but not available. Please build the Rust backend first.")
    
    # If Rust is not requested, use Python backend
    if not use_rust:
        from catanatron.game import Game
        logger.info("Using Python backend as requested")
        return Game(players, **kwargs)
    
    # At this point, Rust is available and requested
    logger.info("Creating game with Rust backend")
    
    # Process players for Rust compatibility - allow basic adaptation
    adapted_players = []
    for i, player in enumerate(players):
        logger.debug(f"Checking player {i}: {player}")
        if _is_player_rust_compatible(player, strict_check=strict_rust):
            logger.debug(f"Player {i} ({player}) is adaptable for Rust")
            
            # If player doesn't have _rust_color attribute, try to add it
            if not hasattr(player, '_rust_color') and hasattr(player, 'color'):
                try:
                    # Extract the color name - handle both enum.name and string representations
                    if hasattr(player.color, 'name'):
                        color_name = str(player.color.name)
                    else:
                        color_name = str(player.color)
                    
                    # Map color to numeric value for Rust
                    color_map = {
                        "RED": 0,
                        "BLUE": 1,
                        "ORANGE": 2,
                        "WHITE": 3,
                    }
                    
                    if color_name in color_map:
                        # Dynamically add _rust_color attribute
                        player._rust_color = color_map[color_name]
                        logger.debug(f"Added _rust_color={player._rust_color} to player {player}")
                    else:
                        logger.warning(f"Could not map color '{color_name}' to Rust color value")
                except Exception as e:
                    logger.warning(f"Failed to adapt player {player}: {e}")
                    
            adapted_players.append(player)
        else:
            # This player is fundamentally incompatible or fails strict check
            if strict_rust:
                error_msg = f"Player {i} ({player}) is not explicitly marked as Rust-compatible."
            else:
                error_msg = f"Player {i} ({player}) is fundamentally incompatible with Rust backend."
            logger.error(error_msg)
            raise TypeError(error_msg)
    
    # Extract parameters expected by the Rust Game constructor
    seed = kwargs.get('seed', None)
    discard_limit = kwargs.get('discard_limit', 7)
    vps_to_win = kwargs.get('vps_to_win', 10)
    map_type = kwargs.get('map_type', 'BASE')
    
    logger.debug(f"Creating Rust game with: discard_limit={discard_limit}, vps_to_win={vps_to_win}, map_type={map_type}")
    
    # Create the Rust game with the explicit parameters
    game = RustGame(adapted_players, seed, discard_limit, vps_to_win, map_type)
    
    # No longer try to set is_rust_backed - that's handled by engine_interface.py
    # The Game class should be treated as a black box
    
    return game

def debug_player(player):
    """Helper function to log detailed information about a player object for debugging."""
    # Only collect and log at DEBUG level
    if logger.level <= logging.DEBUG:
        attr_info = []
        for attr_name in ['color', '_rust_color', 'decide', 'name', 'is_rust_player']:
            if hasattr(player, attr_name):
                value = getattr(player, attr_name)
                attr_info.append(f"{attr_name}={value} (type: {type(value).__name__})")
            else:
                attr_info.append(f"{attr_name}=<missing>")
        
        logger.debug(f"Player debug info - {player}: {', '.join(attr_info)}")
        return attr_info
    return []

def _is_player_rust_compatible(player, strict_check=False):
    """
    Check if a player is compatible with the Rust backend.
    
    A player is considered strictly Rust-compatible if:
    1. It has the required base attributes (color, decide)
    2. It has either:
       a. An explicit _rust_color attribute (must be an integer)
       b. The is_rust_player attribute set to True
    
    If strict_check=False, any player with the basic required attributes (color, decide)
    will be considered adaptable, even if not explicitly marked as Rust-compatible.
    
    Args:
        player: The player object to check
        strict_check: If True, require explicit Rust compatibility markers
        
    Returns:
        bool: True if the player is Rust-compatible, False otherwise
    """
    # Debug log player information
    debug_player(player)
    
    # Check for essential attributes
    for attr in ['color', 'decide']:
        if not hasattr(player, attr):
            logger.debug(f"Player {player} is missing required attribute: {attr}")
            return False
    
    # If not doing a strict check, any player with color and decide is adaptable
    if not strict_check:
        logger.debug(f"Player {player} has basic attributes needed for adaptation")
        return True
    
    # When doing a strict check, require explicit Rust compatibility markers
    has_rust_color = hasattr(player, '_rust_color')
    is_rust_player = getattr(player, 'is_rust_player', False)
    
    # Validate _rust_color if present
    if has_rust_color and not isinstance(player._rust_color, int):
        logger.debug(f"Player {player} has _rust_color that is not an integer: {type(player._rust_color)}")
        return False
    
    # Must have at least one compatibility marker for strict check
    if has_rust_color or is_rust_player:
        logger.debug(f"Player {player} is explicitly Rust compatible!")
        return True
    
    logger.debug(f"Player {player} is not explicitly Rust compatible - no compatibility markers found")
    return False

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
    Adapts accumulators for the backend (Python or Rust).
    
    For now, this simply returns the original accumulators as we're using 
    the Python backend under the hood.
    
    Args:
        accumulators: List of accumulators
        use_rust: Whether Rust backend is being used
        
    Returns:
        List of adapted accumulators
    """
    # For now, just return the original accumulators
    return accumulators 