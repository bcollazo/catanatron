"""
Tests for feature parity between Python and Rust backends.

This module contains tests that verify that both Python and Rust
backends provide the same functionality and produce equivalent results.
"""

import unittest
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

from catanatron.models.player import RandomPlayer, Color
from catanatron_experimental.engine_interface import (
    create_game, 
    is_python_available, 
    is_rust_available,
)
from catanatron_experimental.cli.cli_players import RustRandomPlayerProxy

@unittest.skipIf(not is_python_available(), "Python backend not available")
class PythonBackendTests(unittest.TestCase):
    """Tests for the Python backend functionality."""
    
    def setUp(self):
        """Set up test players."""
        self.players = [
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
        ]
    
    def test_game_creation(self):
        """Test that a game can be created with the Python backend."""
        game = create_game(self.players, use_rust=False)
        self.assertIsNotNone(game)
        self.assertFalse(hasattr(game, 'is_rust_backed') and game.is_rust_backed)
    
    def test_game_play(self):
        """Test that a game can be played with the Python backend."""
        game = create_game(self.players, use_rust=False)
        winner = game.play()
        self.assertIsNotNone(winner)

@unittest.skipIf(not is_rust_available(), "Rust backend not available")
class RustBackendTests(unittest.TestCase):
    """Tests for the Rust backend functionality."""
    
    def setUp(self):
        """Set up test players."""
        self.players = [
            RustRandomPlayerProxy(Color.RED),
            RustRandomPlayerProxy(Color.BLUE),
        ]
    
    def test_game_creation(self):
        """Test that a game can be created with the Rust backend."""
        game = create_game(self.players, use_rust=True)
        self.assertIsNotNone(game)
        self.assertTrue(hasattr(game, 'is_rust_backed') and game.is_rust_backed)
    
    def test_game_play(self):
        """Test that a game can be played with the Rust backend."""
        game = create_game(self.players, use_rust=True)
        winner = game.play()
        self.assertIsNotNone(winner)

@unittest.skipIf(not (is_python_available() and is_rust_available()),
               "Both Python and Rust backends must be available")
class BackendParityTests(unittest.TestCase):
    """Tests for feature parity between Python and Rust backends."""
    
    def setUp(self):
        """Set up test players for both backends."""
        self.py_players = [
            RandomPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
        ]
        self.rust_players = [
            RustRandomPlayerProxy(Color.RED),
            RustRandomPlayerProxy(Color.BLUE),
        ]
    
    def test_game_interface_consistency(self):
        """Test that both backends present a consistent game interface."""
        py_game = create_game(self.py_players, use_rust=False)
        rust_game = create_game(self.rust_players, use_rust=True)
        
        # Check that both games have the same essential methods
        essential_methods = ['play', 'winning_color']
        for method in essential_methods:
            self.assertTrue(hasattr(py_game, method), f"Python game missing method: {method}")
            self.assertTrue(hasattr(rust_game, method), f"Rust game missing method: {method}")
    
    def test_simulation_count(self):
        """Test that both backends can run the same number of simulations."""
        # Run a small number of simulations with each backend
        num_games = 3
        py_winners = []
        rust_winners = []
        
        # Python games
        for _ in range(num_games):
            game = create_game(self.py_players, use_rust=False)
            py_winners.append(game.play())
        
        # Rust games
        for _ in range(num_games):
            game = create_game(self.rust_players, use_rust=True)
            rust_winners.append(game.play())
        
        # Check that both backends completed all games
        self.assertEqual(len(py_winners), num_games)
        self.assertEqual(len(rust_winners), num_games)

if __name__ == '__main__':
    unittest.main() 