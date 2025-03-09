"""
Utility functions for CLI operations
"""
import logging
import re
from rich.table import Table
from rich.theme import Theme

from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points as core_get_actual_vp

# Import common utilities from main utils.py
from catanatron_experimental.utils import formatSecs, ensure_dir

# Define empty lists that will be populated by cli_players.py
# This avoids circular imports
CUSTOM_ACCUMULATORS = []

# Custom theme for rich console output
custom_theme = Theme(
    {
        "red": "#FF0000",
        "blue": "#0000FF",
        "orange": "#FF7F00",
        "white": "#FFFFFF",
    }
)

def get_actual_victory_points(game, color):
    """
    Wrapper around the core get_actual_victory_points function
    to handle both Python and Rust game states
    """
    try:
        # For Python games
        return core_get_actual_vp(game.state, color)
    except (AttributeError, TypeError):
        # For Rust games
        return game.get_victory_points(color)

def player_help_table():
    """
    Create a table showing available player types and their codes
    """
    # Import here to avoid circular imports
    from catanatron_experimental.cli.cli_players import CLI_PLAYERS
    
    table = Table(title="Player Codes")
    table.add_column("CODE", style="cyan")
    table.add_column("DESCRIPTION", style="green")
    table.add_column("PARAMS", style="yellow")
    
    for cli_player in sorted(CLI_PLAYERS, key=lambda p: p.code):
        table.add_row(
            cli_player.code,
            cli_player.description,
            cli_player.params_help or "",
        )
    
    return table

def parse_player_arg(players_str):
    """
    Parse player argument string into player objects
    
    Args:
        players_str: Comma-separated player codes, e.g. "R,W,AB:2,VP"
        
    Returns:
        List of tuples (player, params)
    """
    # Import here to avoid circular imports
    from catanatron_experimental.cli.cli_players import CLI_PLAYERS
    
    player_keys = players_str.split(",")
    players = []
    colors = [c for c in Color]
    
    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        
        for cli_player in CLI_PLAYERS:
            if cli_player.code == code:
                color = colors[i % len(colors)]
                params = [color] + parts[1:]
                player = cli_player.import_fn(*params)
                players.append((player, parts[1:] if len(parts) > 1 else []))
                break
        else:
            logging.warning(f"Unknown player code: {code}")
    
    return players 