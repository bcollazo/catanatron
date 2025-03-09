"""
Registry for Rust-compatible player types.

This module provides registration and retrieval mechanisms for
tracking which player types have Rust-compatible implementations.
"""

import logging
from typing import Dict, Any, Type, Optional, Set

from catanatron.models.player import Player

logger = logging.getLogger(__name__)

# Registry of player classes that have Rust-compatible versions
_RUST_COMPATIBLE_PLAYERS: Set[Type[Player]] = set()

def register_rust_compatible(player_class: Type[Player]) -> Type[Player]:
    """
    Register a player class as having a Rust-compatible implementation.
    
    This is a decorator that can be used on player classes to mark them
    as having Rust-compatible implementations.
    
    Args:
        player_class: The player class to register
        
    Returns:
        The same player class (for decorator chaining)
    """
    _RUST_COMPATIBLE_PLAYERS.add(player_class)
    logger.debug(f"Registered {player_class.__name__} as Rust-compatible")
    return player_class

def is_rust_compatible_type(player_class: Type[Player]) -> bool:
    """
    Check if a player class has a Rust-compatible implementation.
    
    Args:
        player_class: The player class to check
        
    Returns:
        bool: True if the player class has a Rust-compatible implementation
    """
    return player_class in _RUST_COMPATIBLE_PLAYERS

def is_rust_compatible_instance(player: Player) -> bool:
    """
    Check if a player instance is compatible with the Rust backend.
    
    Args:
        player: The player instance to check
        
    Returns:
        bool: True if the player is Rust-compatible
    """
    # Check for the is_rust_player attribute
    if hasattr(player, 'is_rust_player') and player.is_rust_player:
        return True
        
    # Check if the player's class is registered as Rust-compatible
    return is_rust_compatible_type(player.__class__)

def get_rust_compatible_players() -> Set[Type[Player]]:
    """
    Get the set of registered Rust-compatible player classes.
    
    Returns:
        Set of player classes that have Rust-compatible implementations
    """
    return _RUST_COMPATIBLE_PLAYERS.copy() 