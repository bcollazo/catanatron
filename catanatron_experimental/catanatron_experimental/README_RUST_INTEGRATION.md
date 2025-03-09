# Catanatron Rust Integration

This document outlines the approach for integrating the Rust backend into the Catanatron project.

## Integration Strategy

We're using the "Strangler Fig" pattern for the integration, which allows us to:

1. Maintain both Python and Rust implementations in parallel
2. Gradually transition from Python to Rust
3. Ensure continuous operation during the migration

This approach is named after the strangler fig vine, which gradually envelops and replaces its host tree, much like how our Rust implementation will eventually replace the Python one.

## Current Phase: Phase 1 (Initial Integration)

In Phase 1, we've established the foundation for the integration:

1. **Defined Interface Boundaries**: Created clear interfaces between the Python and Rust implementations
2. **Implemented Factory Pattern**: Added factory functions to create games with either backend
3. **Added Player Compatibility**: Developed adapter classes for players to work with both backends
4. **Created Testing Infrastructure**: Built tests to verify feature parity between implementations

## Key Components

### Engine Interface

The `engine_interface.py` module provides the core interface for working with either backend:

```python
from catanatron_experimental.engine_interface import (
    create_game,
    is_rust_available,
    prepare_accumulators,
)

# Create a game with default backend (Python)
game = create_game(players)

# Create a game with Rust backend (if available)
game = create_game(players, use_rust=True)
```

### Player Adapters

The `player_adapter.py` module provides classes to make players compatible with both backends:

```python
from catanatron_experimental.player_adapter import RustCompatibleMixin

# Create a Rust-compatible player class
class MyRustPlayer(RustCompatibleMixin, MyPlayerClass):
    # Your implementation here
    pass
```

### Player Registry

The `player_registry.py` module keeps track of which player types have Rust-compatible versions:

```python
from catanatron_experimental.player_registry import register_rust_compatible

# Register a player class as Rust-compatible
@register_rust_compatible
class MyRustPlayer(Player):
    # Your implementation here
    pass
```

## Usage

### Using the CLI

The `catanatron-play` CLI now supports a `--rust` flag to use the Rust backend:

```bash
# Use Python backend
catanatron-play --players=R,R,R,R --num=10

# Use Rust backend with Rust-compatible players
catanatron-play --players=RR,RR --num=10 --rust
```

Note that the Rust backend is currently in development, so some functionality may not be available yet. When using the `--rust` flag with players that are not Rust-compatible, the system will issue a warning.

### Using the API

You can use the engine interface directly in your code:

```python
from catanatron.models.player import RandomPlayer, Color
from catanatron_experimental.cli.cli_players import RustRandomPlayerProxy
from catanatron_experimental.engine_interface import create_game, is_rust_available

# Create players
py_players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
]

rust_players = [
    RustRandomPlayerProxy(Color.RED),
    RustRandomPlayerProxy(Color.BLUE),
]

# Create games
py_game = create_game(py_players, use_rust=False)
rust_game = create_game(rust_players, use_rust=True)

# Play games
py_winner = py_game.play()
rust_winner = rust_game.play()
```

## Next Steps

Future phases will include:

1. **Phase 2**: Complete player type support, feature parity
2. **Phase 3**: Performance optimization and benchmarking
3. **Phase 4**: Default to Rust backend for most operations
4. **Phase 5**: Optional removal of Python implementation

## Contributing

If you're adding a new player type, consider making it Rust-compatible by:

1. Using the `RustCompatibleMixin` class
2. Registering it with the `@register_rust_compatible` decorator
3. Ensuring it works with both Python and Rust backends

## Testing

Run the backend parity tests to ensure your changes work with both backends:

```bash
python -m unittest catanatron_experimental.tests.test_backend_parity
``` 