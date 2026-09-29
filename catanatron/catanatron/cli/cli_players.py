"""The ``--help-players`` table.

The registry itself lives in :mod:`catanatron.registry` so that the web
server can share it; this module only holds what is specific to the terminal.
"""

from rich.table import Table

from catanatron.players import register_builtins
from catanatron.registry import REGISTRY, describe

register_builtins()


def _format_params(params):
    return ", ".join(
        f"{p['name']}={p['default']!r}" if p["default"] is not None else p["name"]
        for p in params
    )


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("PARAMS", style="green")
    table.add_column("DESCRIPTION")
    for key in sorted(REGISTRY):
        entry = describe(key, REGISTRY[key])
        table.add_row(
            entry["key"],
            entry["name"],
            _format_params(entry["params"]),
            entry["description"],
        )
    return table
