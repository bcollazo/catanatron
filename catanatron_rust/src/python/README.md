# Catanatron PyO3 Bindings

This directory contains Python bindings for the Catanatron Rust implementation using PyO3. These bindings allow Python code to interact with the Rust-based game engine, providing significant performance improvements while maintaining a familiar API.

## Architecture

The bindings follow a modular structure:

- `action.rs` - Provides the `Action` class for representing and creating game actions
- `player.rs` - Defines the `Player` class that can be subclassed in Python
- `game.rs` - Implements the `Game` class to manage game state and flow
- `state.rs` - Utilities for converting between Rust and Python state representations

## Enhanced State Representation

The bindings include a comprehensive Python-friendly dictionary representation of the game state:

```python
{
    # Basic game state
    'current_player': 0,                # Player index (0-3)
    'current_player_color': 'RED',      # Player color name
    'is_initial_build_phase': True,     # Whether in initial build phase
    'current_tick_seat': 0,             # Current player's seat 
    'action_prompt': 'BUILD_SETTLEMENT', # Current action being prompted
    'has_rolled': False,                # Whether current player has rolled
    
    # Bank information
    'bank': {
        'wood': 19,
        'brick': 19,
        'sheep': 19,
        'wheat': 19,
        'ore': 19
    },
    
    # Robber information
    'robber_tile': 7,                   # Current robber tile
    
    # Player information
    'players': {
        'RED': {
            'resources': {'wood': 0, 'brick': 0, ...},
            'development_cards': [],
            'settlements': [0, 3],      # Node IDs of settlements
            'cities': [],               # Node IDs of cities
            'roads': [0, 5]             # Edge IDs of roads
        },
        'BLUE': {...},
        ...
    },
    
    # Board state
    'buildable_nodes': [5, 7, 9, ...],  # Node IDs buildable by current player
    'buildable_edges': [8, 10, 12, ...] # Edge IDs buildable by current player
}
```

This state representation makes it easy to implement sophisticated strategies and AI players in Python.

## Usage Examples

### Creating a Game

```python
from catanatron_rust import Game, Player

# Create players
class RandomPlayer(Player):
    def __init__(self, color):
        super().__init__(color, f"RandomBot-{color}")
    
    def decide(self, game_state, actions):
        import random
        return random.choice(actions)

# Create and run a game
players = [RandomPlayer(0), RandomPlayer(1), RandomPlayer(2), RandomPlayer(3)]
game = Game(players, seed=42, discard_limit=7, vps_to_win=10)
winner = game.play()
print(f"Winner: Player {winner}")
```

### Using Actions

```python
from catanatron_rust import Action, Game, Player

# Create build action
action = Action.build_settlement(0, 3)

# Use in game
game = Game([...])
game.play_tick()  # Automatic action
```

### Exploring the Game State

```python
# Get detailed state representation
state = game.get_state_repr()

# Check current player and phase
print(f"Current player: {state['current_player_color']}")
print(f"Action prompt: {state['action_prompt']}")

# Check available resources for a player
resources = state['players']['RED']['resources']
print(f"Wood: {resources['wood']}, Brick: {resources['brick']}")

# See available build locations
print(f"Buildable nodes: {state['buildable_nodes']}")
```

## Example Scripts

In the `examples/` directory, you'll find these scripts demonstrating the bindings:

- `enhanced_pyrust_interface.py`: Demonstrates the full PyO3 bindings capabilities
- `benchmark_pyrust_vs_python.py`: Compares performance between Rust and Python implementations
- `state_inspector.py`: An interactive tool to explore and visualize the game state

## Design Philosophy

These bindings are designed with the following principles:

1. **Pythonic Interface** - Follows Python conventions while leveraging Rust performance
2. **Clear Separation** - Python bindings are separate from core Rust implementation
3. **Extensibility** - Easy to add new features and functionality
4. **Backward Compatibility** - Maintains compatibility with existing Python code
5. **Performance** - Optimized for speed while providing a user-friendly API

## Development Guidelines

When extending these bindings:

1. Keep each binding in its own module
2. Use descriptive method names and clear documentation
3. Return Rust errors as Python exceptions
4. Prefer immutable methods where possible
5. Follow the existing pattern for new bindings
6. Minimize unnecessary conversions between Rust and Python
7. Document all public interfaces thoroughly

## Building the Bindings

To build and install the bindings:

```bash
cd catanatron_rust
maturin develop
```

This will compile the Rust code and make it available for import in your Python environment. 