import importlib.util
import logging
import os
import sys
import time
import random
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Union

import click
from rich.box import MINIMAL
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme
from rich.text import Text

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.map_instance import build_map

# Import from utils
from catanatron_experimental.utils import formatSecs, ensure_dir

# Import CLI-related modules
from catanatron_experimental.cli.cli_players import CLI_PLAYERS
from catanatron_experimental.cli.accumulators import (
    CsvDataAccumulator,
    DatabaseAccumulator,
    JsonDataAccumulator,
    StatisticsAccumulator,
    VpDistributionAccumulator,
)
from catanatron_experimental.cli.utils import (
    CUSTOM_ACCUMULATORS,
    custom_theme,
    get_actual_victory_points,
    player_help_table,
    parse_player_arg,
)
from catanatron_experimental.cli.simulation_accumulator import SimulationAccumulator
from catanatron_experimental.engine_interface import (
    create_game,
    is_rust_available,
    prepare_accumulators,
)

# Try to import is_rust_compatible_instance from player_registry
try:
    from catanatron_experimental.player_registry import is_rust_compatible_instance
    PLAYER_REGISTRY_AVAILABLE = True
except ImportError:
    PLAYER_REGISTRY_AVAILABLE = False
    # Create a dummy function if the registry module isn't available
    def is_rust_compatible_instance(player):
        return hasattr(player, 'is_rust_player') and player.is_rust_player

# try to suppress TF output before any potentially tf-importing modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CustomTimeRemainingColumn(TimeRemainingColumn):
    """Custom column for displaying time remaining in Rich progress bar."""

    def render(self, task):
        """Render the column."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("-:--:--", style="progress.remaining")
        return Text(formatSecs(int(remaining)), style="progress.remaining")


@click.command()
@click.option("-n", "--num", default=5, help="Number of games to play.")
@click.option(
    "--players",
    default="R,R,R,R",
    help="""
    Comma-separated players to use. Use ':' to set player-specific params.
    (e.g. --players=R,G:25,AB:2:C,W).\n
    See player legend with '--help-players'.
    """,
)
@click.option(
    "--code",
    default=None,
    help="Path to file with custom Players and Accumulators to import and use.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Directory where to save game data.",
)
@click.option(
    "--json",
    default=None,
    is_flag=True,
    help="Save game data in JSON format.",
)
@click.option(
    "--csv", default=False, is_flag=True, help="Save game data in CSV format."
)
@click.option(
    "--db",
    default=False,
    is_flag=True,
    help="""
        Save game in PGSQL database.
        Expects docker-compose provided database to be up and running.
        This allows games to be watched.
        """,
)
@click.option(
    "--rust",
    default=False,
    is_flag=True,
    help="Use the Rust backend for faster simulation. Fails if Rust backend is not available.",
)
@click.option(
    "--strict-rust",
    default=False,
    is_flag=True,
    help="When used with --rust, only allow players explicitly marked as Rust-compatible.",
)
@click.option(
    "--config-discard-limit",
    default=7,
    help="Sets Discard Limit to use in games.",
)
@click.option(
    "--config-vps-to-win",
    default=10,
    help="Sets Victory Points needed to win games.",
)
@click.option(
    "--config-map",
    default="BASE",
    type=click.Choice(["BASE", "MINI", "TOURNAMENT"], case_sensitive=False),
    help="Sets Map to use. MINI is a 7-tile smaller version. TOURNAMENT uses a fixed balanced map.",
)
@click.option(
    "--quiet",
    default=False,
    is_flag=True,
    help="Silence console output. Useful for debugging.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the Python logging level. Use DEBUG for verbose output.",
)
@click.option(
    "--help-players",
    default=False,
    type=bool,
    help="Show player codes and exits.",
    is_flag=True,
)
def simulate(
    num,
    players,
    code,
    output,
    json,
    csv,
    db,
    rust,
    strict_rust,
    config_discard_limit,
    config_vps_to_win,
    config_map,
    quiet,
    log_level,
    help_players,
):
    """
    Catan Bot Simulator.

    Examples:
        catanatron-play --players=R,R,R,R --num=1000\n
        catanatron-play --players=W,W,R,R --num=50000 --output=data/ --csv\n
        catanatron-play --players=VP,F --num=10 --output=data/ --json\n
        catanatron-play --players=W,F,AB:3 --num=1 --csv --json --db --quiet\n
        catanatron-play --players=R,R,R,R --num=100 --rust
        catanatron-play --players=RR,RR --rust
        catanatron-play --players=RR,RR --rust --log-level=ERROR
    """
    # Set Python logging level
    import logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.basicConfig(level=numeric_level)
    
    # Set logging for the Rust backend to reduce verbosity
    if rust:
        # Set the RUST_LOG environment variable to control Rust log levels
        import os
        # Map Python log levels to Rust log levels
        rust_level = {
            "DEBUG": "debug",
            "INFO": "info", 
            "WARNING": "warn",
            "ERROR": "error",
            "CRITICAL": "error"
        }.get(log_level.upper(), "warn")
        os.environ['RUST_LOG'] = rust_level
    
    # If Rust is requested but not available, fail fast
    if rust and not is_rust_available():
        raise ImportError("Rust backend requested but not available. Please build the Rust backend first.")
        
    # Show information about Rust backend status
    if rust and is_rust_available():
        console = Console()
        console.print("[yellow]Note: The Rust backend integration is still in development.[/yellow]")
        console.print("[yellow]Only players marked with 'RR' are fully Rust-compatible.[/yellow]")
        # Remove the message about fallback since we don't fall back with --rust
        # console.print("[yellow]Games with mixed player types (R and RR) will use the Python backend.[/yellow]")

    if code:
        abspath = os.path.abspath(code)
        spec = importlib.util.spec_from_file_location("module.name", abspath)
        if spec is not None and spec.loader is not None:
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)

    if help_players:
        return Console().print(player_help_table())

    if output and not (json or csv):
        click.echo("--output specified but neither --json nor --csv specified. No output.")

    # Figure out Players objects from CLI options
    player_specs = parse_player_arg(players)
    players = [player for player, _ in player_specs]
    
    # Create output options and game config options with proper values
    output_options = OutputOptions(
        output=output,
        csv=csv,
        json=json,
        db=db
    )
    
    game_config = GameConfigOptions(
        discard_limit=config_discard_limit,
        vps_to_win=config_vps_to_win,
        map_instance=config_map
    )

    # When using Rust, check for player compatibility based on strictness setting
    if rust and is_rust_available():
        from catanatron_experimental.rust_bridge import _is_player_rust_compatible
        console = Console()
        console.print("[yellow]Note: The Rust backend integration is still in development.[/yellow]")
        
        # Check players and provide warnings for non-Rust-compatible players
        non_rust_compatible = []
        for i, player in enumerate(players):
            # Use strict checking if strict_rust flag is set
            if not _is_player_rust_compatible(player, strict_check=strict_rust):
                non_rust_compatible.append((i, player))
        
        if non_rust_compatible:
            if strict_rust:
                # In strict mode, refuse to run with incompatible players
                console.print("[red]Error: Some players are not Rust-compatible:[/red]")
                for i, player in non_rust_compatible:
                    console.print(f"[red]  - Player {i}: {player}[/red]")
                console.print("[red]When using --strict-rust, all players must be explicitly Rust-compatible.[/red]")
                console.print("[red]Use player type 'RR' for Rust-compatible players.[/red]")
                return
            else:
                # In non-strict mode, warn but continue
                console.print("[yellow]Warning: Some players are not explicitly marked as Rust-compatible:[/yellow]")
                for i, player in non_rust_compatible:
                    console.print(f"[yellow]  - Player {i}: {player}[/yellow]")
                console.print("[yellow]The system will attempt to adapt these players for the Rust backend.[/yellow]")
                console.print("[yellow]For best results, consider using player type 'RR' for all players.[/yellow]")
                console.print("[yellow]You can use --strict-rust to enforce strict compatibility.[/yellow]")
                
                # Ask for confirmation before proceeding
                if not quiet and click.confirm("Do you want to continue?", default=True):
                    pass
                elif not quiet:
                    return

    # Play the games with no silent exception catching for Rust
    return play_batch(
        num,
        players,
        output_options=output_options,
        game_config=game_config,
        quiet=quiet,
        use_rust=rust,
        strict_rust=strict_rust,
    )


@dataclass
class OutputOptions:
    """Class to keep track of output CLI flags"""

    output: Union[str, None] = None  # path to store files
    csv: bool = False
    json: bool = False
    db: bool = False


@dataclass
class GameConfigOptions:
    discard_limit: int = 7
    vps_to_win: int = 10
    map_instance: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"


def rich_player_name(player):
    color = player.color
    return f"[{color.value}]{player}[/{color.value}]"


def rich_color(color):
    """
    Format a color for rich display.
    
    Args:
        color: Either a Color enum object (from Python backend) or an integer (from Rust backend)
              where 0=RED, 1=BLUE, 2=ORANGE, 3=WHITE
    
    Returns:
        A formatted string for rich display
    """
    # Define color name mapping for Rust integer colors
    COLOR_NAMES = {
        0: "red",
        1: "blue", 
        2: "orange",
        3: "white"
    }
    
    if isinstance(color, int):
        # Handle Rust integer colors
        name = COLOR_NAMES.get(color, "gray")
        style = name
        # Return the text representation for the color
        color_text = {0: "RED", 1: "BLUE", 2: "ORANGE", 3: "WHITE"}.get(color, str(color))
        return f"[{style}]{color_text}[/{style}]"
    else:
        # Handle Python Color enum objects
        name = color.name.lower()
        style = name
        return f"[{style}]{color.value}[/{style}]"


def play_batch_core(num_games, players, game_config, accumulators=[], use_rust=False, strict_rust=False):
    """Plays a batch of games with the given players and config.
    
    Args:
        num_games: Number of games to play
        players: List of Player objects
        game_config: GameConfigOptions
        accumulators: List of accumulators to use
        use_rust: Whether to use Rust backend
        strict_rust: Whether to strictly enforce Rust compatibility
        
    Returns:
        List of winners (colors)
        
    Raises:
        When use_rust=True, will propagate all exceptions from the Rust backend
        rather than catching them.
    """
    import traceback
    from catanatron_experimental.engine_interface import create_game
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialize accumulators that need to be initialized once per batch
    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.before_all()
    
    winners = []
    for i in range(num_games):
        # When using Rust, we want to propagate exceptions to enforce strict behavior
        # When using Python, we catch exceptions to continue with other games
        if use_rust:
            # Reset player state if needed
            for player in players:
                if hasattr(player, 'reset_state'):
                    player.reset_state()
            
            # For Rust backend, use map_type parameter
            game_params = {
                'discard_limit': game_config.discard_limit,
                'vps_to_win': game_config.vps_to_win,
                'map_type': game_config.map_instance,  # Pass the map type string directly
                'seed': None,  # Explicit None to match Rust expectation
            }
            
            # Create the game with appropriate parameters - don't catch exceptions
            game = create_game(
                players=players, 
                use_rust=use_rust,
                strict_rust=strict_rust,  # Pass strict_rust directly
                **game_params
            )
            
            # Initialize accumulators
            for accumulator in accumulators:
                accumulator.before(game)
            
            # Play the game
            winner = game.play()
            winners.append(winner)
            
            # Update accumulators
            for accumulator in accumulators:
                accumulator.after(game)
        else:
            # Original code with exception handling for Python backend
            try:
                # Reset player state if needed
                for player in players:
                    if hasattr(player, 'reset_state'):
                        player.reset_state()
                
                # For Python backend, use map_instance parameter with build_map
                from catanatron.models.map_instance import build_map
                game_params = {
                    'discard_limit': game_config.discard_limit,
                    'vps_to_win': game_config.vps_to_win,
                    'map_instance': build_map(game_config.map_instance),
                    'seed': None,
                }
                
                # Create the game with appropriate parameters
                game = create_game(
                    players=players, 
                    use_rust=use_rust,
                    **game_params
                )
                
                # Initialize accumulators
                for accumulator in accumulators:
                    try:
                        accumulator.before(game)
                    except Exception as e:
                        logger.error(f"Error in accumulator.before: {e}")
                
                # Play the game
                winner = game.play()
                winners.append(winner)
                
                # Update accumulators
                for accumulator in accumulators:
                    try:
                        accumulator.after(game)
                    except Exception as e:
                        logger.error(f"Error in accumulator.after: {e}")
                        
            except Exception as e:
                logger.error(f"Error in game {i+1}: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                # Continue with the next game
                continue
    
    # Finalize accumulators
    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.after_all()
            
    return winners


def play_batch(
    num_games,
    players,
    output_options=None,
    game_config=None,
    quiet=False,
    use_rust=False,
    strict_rust=False,
):
    """
    Play multiple games at once.
    
    Args:
        num_games: Number of games to play
        players: List of player objects
        output_options: Output configuration
        game_config: Game configuration
        quiet: Whether to suppress console output
        use_rust: Whether to use the Rust backend
        strict_rust: Whether to strictly enforce Rust compatibility
    """
    if output_options is None:
        output_options = OutputOptions()
    if game_config is None:
        game_config = GameConfigOptions()

    console = Console()

    # ===== Configure Output Options =====
    directory = output_options.output
    if directory is not None:
        ensure_dir(directory)

    # ===== Set up Accumulators (we keep track of winners) =====
    additional_accumulators = []  # can be extended for specific needs
    accumulators = additional_accumulators
    # Always track winners, for CLI output
    winners_by_color = Counter()
    
    # Record start time
    start = time.time()

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        CustomTimeRemainingColumn(),
        console=None if quiet else console,
    ) as progress:
        task = progress.add_task(f"Running {num_games} games...", total=num_games)
        
        # Play games and collect winners
        try:
            winners = play_batch_core(
                num_games, 
                players, 
                game_config, 
                accumulators=accumulators, 
                use_rust=use_rust,
                strict_rust=strict_rust
            )
            progress.update(task, advance=num_games)
        except Exception as e:
            # When using Rust, provide a clearer error message
            if use_rust:
                console.print(f"\n[bold red]Error running games with Rust backend:[/bold red]")
                console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
                console.print("[yellow]Try with --log-level=DEBUG for more details.[/yellow]")
                return []
            else:
                # Re-raise for Python backend
                raise

    # Update the counter with the winners
    for winner in winners:
        winners_by_color[winner] += 1

    # Calculate elapsed time
    elapsed = time.time() - start
    rate = num_games / elapsed if elapsed > 0 else float("inf")
    
    if not quiet:
        console.print("\n=========== RESULTS ===========")
        console.print(
            f"Ran {num_games} games in {elapsed:.3f} secs\nRate: {rate:.2f} games/sec\n"
        )

        console.print("=========== WINNERS ===========")
        
        # Print winners accounting for possible integer colors from Rust
        if len(winners_by_color) > 0:
            for color, count in winners_by_color.most_common():
                console.print(
                    f"{rich_color(color)}: {count} ({count/num_games*100:.1f}%)",
                )
        else:
            console.print("No winners recorded.")

    return winners


if __name__ == "__main__":
    simulate()
