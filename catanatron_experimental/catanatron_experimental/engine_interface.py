"""
Interface definitions for Catanatron game engines.

This module defines the abstract interfaces that both Python and Rust
implementations must adhere to, providing a consistent API regardless
of the underlying implementation.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple, Callable

logger = logging.getLogger(__name__)

# Import implementations with proper error handling
try:
    from catanatron.game import Game as PythonGame
    PYTHON_AVAILABLE = True
except ImportError:
    logger.warning("Python implementation not available")
    PYTHON_AVAILABLE = False
    PythonGame = None

try:
    from catanatron_experimental.rust_bridge import (
        is_rust_available as _is_rust_available,
        create_game as _create_rust_game,
        adapt_accumulators_for_backend,
    )
    RUST_AVAILABLE = _is_rust_available()
except ImportError:
    logger.warning("Rust implementation not available")
    RUST_AVAILABLE = False
    _create_rust_game = None
    adapt_accumulators_for_backend = lambda accumulators, use_rust: accumulators

def is_rust_available() -> bool:
    """Check if the Rust implementation is available."""
    return RUST_AVAILABLE

def is_python_available() -> bool:
    """Check if the Python implementation is available."""
    return PYTHON_AVAILABLE

def create_game(players, use_rust=False, strict_rust=False, **kwargs):
    """
    Create a game instance using either Python or Rust backend.
    
    Args:
        players: List of player objects
        use_rust: Whether to use Rust backend if available
        strict_rust: Whether to strictly enforce Rust compatibility
        **kwargs: Additional game configuration parameters
        
    Returns:
        A Game instance (either Python or Rust-backed)
        
    Raises:
        ImportError: If requested backend is not available
    """
    # Check if the requested backend is available
    if use_rust and not RUST_AVAILABLE:
        raise ImportError("Rust backend requested but not available. Please build the Rust backend first.")
    
    if not use_rust and not PYTHON_AVAILABLE:
        raise ImportError("Python implementation not available")
    
    # Create appropriate game instance
    if use_rust:
        logger.info("Creating game with Rust backend")
        game = _create_rust_game(players, use_rust=True, strict_rust=strict_rust, **kwargs)
        
        # We don't need to set any flags on the game object
        # Just return it as is
        return game
    else:
        logger.info("Creating game with Python backend")
        return PythonGame(players, **kwargs)

def prepare_accumulators(accumulators, use_rust=False):
    """
    Prepare accumulators for use with the selected backend.
    
    Args:
        accumulators: List of accumulator objects
        use_rust: Whether using Rust backend
        
    Returns:
        List of properly adapted accumulators
    """
    if use_rust and RUST_AVAILABLE:
        return adapt_accumulators_for_backend(accumulators, use_rust=True)
    return accumulators 