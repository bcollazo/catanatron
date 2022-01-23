import os

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
from catanatron.state_functions import get_actual_victory_points

# try to suppress TF output before any potentially tf-importing modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from catanatron_experimental.utils import ensure_dir, formatSecs
from catanatron_experimental.cli.cli_players import player_help_table, CLI_PLAYERS
from catanatron_experimental.cli.accumulators import (
    JsonDataAccumulator,
    CsvDataAccumulator,
    DatabaseAccumulator,
    StatisticsAccumulator,
    VpDistributionAccumulator,
)


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
    "-p",
    "--players",
    default="R,R,R,R",
    help="""
    Comma-separated players to use. Use ':' to set player-specific params.
    (e.g. --players=R,G:25,AB:2:C,W).\n
    See player legend with '--help-players'.
    """,
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
    "--config-discard-limit",
    default=7,
    help="Sets Discard Limit to use in games.",
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
    num, players, output, json, csv, db, config_discard_limit, quiet, help_players
):
    """
    Catan Bot Simulator.
    Catanatron allows you to simulate millions of games at scale
    and test bot strategies against each other.

    Examples:\n\n
        catanatron-play --players=R,R,R,R --num=1000\n
        catanatron-play --players=W,W,R,R --num=50000 --output=data/ --csv\n
        catanatron-play --players=VP,F --num=10 --output=data/ --json\n
        catanatron-play --players=W,F,AB:3 --num=1 --csv --json --db --quiet
    """
    if help_players:
        return Console().print(player_help_table)
    if output and not (json or csv):
        return print("--output requires either --json or --csv to be set")

    player_keys = players.split(",")
    players = []
    colors = [c for c in Color]
    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        for cli_player in CLI_PLAYERS:
            if cli_player.code == code:
                params = [colors[i]] + parts[1:]
                player = cli_player.import_fn(*params)
                players.append(player)
                break

    play_batch(num, players, output, json, csv, db, config_discard_limit, quiet)


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


def play_batch_core(num_games, players, config_discard_limit=7, accumulators=[]):
    for _ in range(num_games):
        for player in players:
            player.reset_state()
        game = Game(players, discard_limit=config_discard_limit)
        game.play(accumulators)
        yield game


def play_batch(
    num_games,
    players,
    output=None,
    json=False,
    csv=False,
    db=False,
    config_discard_limit=7,
    quiet=False,
):
    statistics_accumulator = StatisticsAccumulator()
    vp_accumulator = VpDistributionAccumulator()
    accumulators = [statistics_accumulator, vp_accumulator]
    if output:
        ensure_dir(output)
    if output and csv:
        accumulators.append(CsvDataAccumulator(output))
    if output and json:
        accumulators.append(JsonDataAccumulator(output))
    if db:
        accumulators.append(DatabaseAccumulator())

    if quiet:
        for _ in play_batch_core(
            num_games, players, config_discard_limit, accumulators
        ):
            pass
        return

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
    if db:
        table.add_column("LINK", overflow="fold")

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
            play_batch_core(num_games, players, config_discard_limit, accumulators)
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
                if db:
                    row.append(accumulators[-1].link)

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
    avg_duration = formatSecs(statistics_accumulator.get_avg_duration())
    table = Table(box=box.MINIMAL, title="Game Summary")
    table.add_column("AVG TICKS", justify="right")
    table.add_column("AVG TURNS", justify="right")
    table.add_column("AVG DURATION", justify="right")
    table.add_row(avg_ticks, avg_turns, avg_duration)
    console.print(table)

    if output and csv:
        console.print(f"GZIP CSVs saved at: [green]{output}[/green]")

    return (
        dict(statistics_accumulator.wins),
        dict(statistics_accumulator.results_by_player),
        statistics_accumulator.games,
    )


if __name__ == "__main__":
    simulate()
