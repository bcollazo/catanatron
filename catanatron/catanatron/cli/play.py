import os
import importlib.util
from dataclasses import dataclass
from typing import Literal, Union

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich import box
from rich.console import Console
from rich.theme import Theme
from rich.text import Text

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.map import build_map
from catanatron.state_functions import get_actual_victory_points

# try to suppress TF output before any potentially tf-importing modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from catanatron.utils import ensure_dir, format_secs
from catanatron.cli.cli_players import (
    CUSTOM_ACCUMULATORS,
    parse_cli_string,
    player_help_table,
)
from catanatron.cli.accumulators import (
    JsonDataAccumulator,
    StatisticsAccumulator,
    VpDistributionAccumulator,
)
from catanatron.cli.simulation_accumulator import SimulationAccumulator


custom_theme = Theme(
    {
        "progress.remaining": "",
        "progress.percentage": "",
        "bar.complete": "green",
        "bar.finished": "green",
    }
)
console = Console(theme=custom_theme)


class CustomTimeRemainingColumn(TimeRemainingColumn):
    """Renders estimated time remaining according to show_time field."""

    def render(self, task):
        """Show time remaining."""
        show = task.fields.get("show_time", True)
        if not show:
            return Text("")
        return super().render(task)


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
    "--output-format",
    default=None,
    type=click.Choice(["csv", "parquet", "json"], case_sensitive=False),
    help="Format to save game data: csv, parquet, or json.",
)
@click.option(
    "--include-board-tensor",
    default=False,
    is_flag=True,
    help="Wether to generate 3D Tensor of the Board for CNN Learning (slower) when using --csv or --parquet.",
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
    "--step-db",
    default=False,
    is_flag=True,
    help="""
        Save the entire game in PGSQL database.
        Expects docker-compose provided database to be up and running.
        This allows games to be replayed.
        WARNING: this reduces the simulation speed down to 1 game per minute.
        """,
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
    output_format,
    include_board_tensor,
    db,
    step_db,
    config_discard_limit,
    config_vps_to_win,
    config_map,
    quiet,
    help_players,
):
    """
    Catan Bot Simulator.
    Catanatron allows you to simulate millions of games at scale
    and test bot strategies against each other.

    Examples:\n\n
        catanatron-play --players R,R,R,R --num 1000\n
        catanatron-play --players W,W,R,R --num 50000 --output data/ --output-format csv\n
        catanatron-play --players VP,F --num 10 --output data/ --ouput-format json\n
        catanatron-play --players W,F,AB:3 --num 1 --ouput-format csv --db --quiet
    """
    if code:
        abspath = os.path.abspath(code)
        spec = importlib.util.spec_from_file_location("module.name", abspath)
        if spec is not None and spec.loader is not None:
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)

    if help_players:
        return Console().print(player_help_table())
    if output and not output_format:
        return print("--output requires --output-format")

    players = parse_cli_string(players)
    output_options = OutputOptions(
        output, output_format, include_board_tensor, db, step_db
    )
    game_config = GameConfigOptions(config_discard_limit, config_vps_to_win, config_map)
    play_batch(
        num,
        players,
        output_options,
        game_config,
        quiet,
    )


@dataclass(frozen=True)
class OutputOptions:
    """Class to keep track of output CLI flags"""

    output: Union[str, None] = None  # path to store files
    output_format: Union[Literal["csv", "parquet", "json"], None] = None
    include_board_tensor: bool = False
    db: bool = False
    step_db: bool = False


@dataclass(frozen=True)
class GameConfigOptions:
    discard_limit: int = 7
    vps_to_win: int = 10
    catan_map: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"


COLOR_TO_RICH_STYLE = {
    Color.RED: "red",
    Color.BLUE: "blue",
    Color.ORANGE: "yellow",
    Color.WHITE: "white",
}


def rich_player_name(player):
    style = COLOR_TO_RICH_STYLE[player.color]
    return f"[{style}]{player}[/{style}]"


def rich_color(color):
    if color is None:
        return ""
    style = COLOR_TO_RICH_STYLE[color]
    return f"[{style}]{color.value}[/{style}]"


def play_batch_core(num_games, players, game_config, accumulators=[]):
    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.before_all()

    for _ in range(num_games):
        for player in players:
            player.reset_state()
        catan_map = build_map(game_config.catan_map)
        game = Game(
            players,
            discard_limit=game_config.discard_limit,
            vps_to_win=game_config.vps_to_win,
            catan_map=catan_map,
        )
        game.play(accumulators)
        yield game

    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.after_all()


def play_batch(
    num_games,
    players,
    output_options=None,
    game_config=None,
    quiet=False,
):
    output_options = output_options or OutputOptions()
    game_config = game_config or GameConfigOptions()

    statistics_accumulator = StatisticsAccumulator()
    vp_accumulator = VpDistributionAccumulator()
    accumulators = [statistics_accumulator, vp_accumulator]
    if output_options.output:
        ensure_dir(output_options.output)
    if output_options.output:
        if output_options.output_format == "csv":
            # lazy load CsvDataAccumulator since depends on pandas / numpy
            from catanatron.gym.accumulators import CsvDataAccumulator

            accumulators.append(
                CsvDataAccumulator(
                    output_options.output, output_options.include_board_tensor
                )
            )
        elif output_options.output_format == "parquet":
            # lazy load ParquetDataAccumulator since depends on pandas / pyarrow
            from catanatron.gym.accumulators import ParquetDataAccumulator

            accumulators.append(
                ParquetDataAccumulator(
                    output_options.output, output_options.include_board_tensor
                )
            )
        elif output_options.output_format == "json":
            accumulators.append(JsonDataAccumulator(output_options.output))
    if output_options.db:
        # lazy load DatabaseAccumulator since depends on sqlalchemy
        from catanatron.web.database_accumulator import DatabaseAccumulator

        accumulators.append(DatabaseAccumulator())
    if output_options.step_db:
        # lazy load DatabaseAccumulator since depends on sqlalchemy
        from catanatron.web.database_accumulator import StepDatabaseAccumulator

        accumulators.append(StepDatabaseAccumulator())
    for accumulator_class in CUSTOM_ACCUMULATORS:
        accumulators.append(accumulator_class(players=players, game_config=game_config))

    if quiet:
        for _ in play_batch_core(num_games, players, game_config, accumulators):
            pass
        return (
            dict(statistics_accumulator.wins),
            dict(statistics_accumulator.results_by_player),
            statistics_accumulator.games,
        )

    # ===== Game Details
    last_n = 10
    actual_last_n = min(last_n, num_games)
    table = Table(title=f"Last {actual_last_n} Games", box=box.MINIMAL)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("SEATING")
    table.add_column("TURNS", justify="right")
    for player in players:
        table.add_column(f"{player.color.value} VP", justify="right")
    table.add_column("WINNER")
    if output_options.db:
        table.add_column("LINK", overflow="fold")
    if output_options.step_db:
        table.add_column("REPLAY LINK", overflow="fold")

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        CustomTimeRemainingColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(f"Playing {num_games} games...", total=num_games)
        player_tasks = [
            progress.add_task(
                rich_player_name(player), total=num_games, show_time=False
            )
            for player in players
        ]

        for i, game in enumerate(
            play_batch_core(num_games, players, game_config, accumulators)
        ):
            winning_color = game.winning_color()

            if (num_games - last_n) < (i + 1):
                seating = ",".join([rich_color(c) for c in game.state.colors])
                row = [
                    str(i + 1),
                    seating,
                    str(game.state.num_turns),
                ]
                for player in players:  # should be in column order
                    points = get_actual_victory_points(game.state, player.color)
                    row.append(str(points))
                row.append(rich_color(winning_color))

                if output_options.db:
                    from catanatron.web.database_accumulator import DatabaseAccumulator

                    database_accumulator = next(
                        (
                            accumulator
                            for accumulator in accumulators
                            if isinstance(accumulator, DatabaseAccumulator)
                        ),
                        None,
                    )
                    row.append(database_accumulator.link)

                if output_options.step_db:
                    from catanatron.web.database_accumulator import (
                        StepDatabaseAccumulator,
                    )

                    step_database_accumulator = next(
                        (
                            accumulator
                            for accumulator in accumulators
                            if isinstance(accumulator, StepDatabaseAccumulator)
                        ),
                        None,
                    )
                    row.append(step_database_accumulator.link)

                table.add_row(*row)

            progress.update(main_task, advance=1)
            if winning_color is not None:
                winning_index = list(map(lambda p: p.color, players)).index(
                    winning_color
                )
                winner_task = player_tasks[winning_index]
                progress.update(winner_task, advance=1)
        progress.refresh()
    console.print(table)

    # ===== PLAYER SUMMARY
    table = Table(title="Player Summary", box=box.MINIMAL)
    table.add_column("", no_wrap=True)
    table.add_column("WINS", justify="right")
    table.add_column("AVG VP", justify="right")
    table.add_column("AVG SETTLES", justify="right")
    table.add_column("AVG CITIES", justify="right")
    table.add_column("AVG ROAD", justify="right")
    table.add_column("AVG ARMY", justify="right")
    table.add_column("AVG DEV VP", justify="right")
    for player in players:
        vps = statistics_accumulator.results_by_player[player.color]
        avg_vps = sum(vps) / len(vps)
        avg_settlements = vp_accumulator.get_avg_settlements(player.color)
        avg_cities = vp_accumulator.get_avg_cities(player.color)
        avg_largest = vp_accumulator.get_avg_largest(player.color)
        avg_longest = vp_accumulator.get_avg_longest(player.color)
        avg_devvps = vp_accumulator.get_avg_devvps(player.color)
        table.add_row(
            rich_player_name(player),
            str(statistics_accumulator.wins[player.color]),
            f"{avg_vps:.2f}",
            f"{avg_settlements:.2f}",
            f"{avg_cities:.2f}",
            f"{avg_longest:.2f}",
            f"{avg_largest:.2f}",
            f"{avg_devvps:.2f}",
        )
    console.print(table)

    # ===== GAME SUMMARY
    avg_ticks = f"{statistics_accumulator.get_avg_ticks():.2f}"
    avg_turns = f"{statistics_accumulator.get_avg_turns():.2f}"
    avg_duration = format_secs(statistics_accumulator.get_avg_duration())
    table = Table(box=box.MINIMAL, title="Game Summary")
    table.add_column("AVG TICKS", justify="right")
    table.add_column("AVG TURNS", justify="right")
    table.add_column("AVG DURATION", justify="right")
    table.add_row(avg_ticks, avg_turns, avg_duration)
    console.print(table)

    if output_options.output:
        console.print(
            f"{output_options.output_format} files saved at: [green]{output_options.output}[/green]"
        )

    return (
        dict(statistics_accumulator.wins),
        dict(statistics_accumulator.results_by_player),
        statistics_accumulator.games,
    )


if __name__ == "__main__":
    simulate()
