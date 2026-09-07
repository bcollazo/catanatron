"""The ``--help-players`` table.

The registry itself lives in :mod:`catanatron.registry` so that the web
server can share it; this module only holds what is specific to the terminal.
"""

from rich.table import Table

import catanatron.players  # noqa: F401  (registers the builtin players)
from catanatron.registry import REGISTRY


def _format_params(entry):
    parts = []
    for param in entry.params_schema:
        default = param["default"]
        parts.append(
            f"{param['name']}={default!r}" if default is not None else param["name"]
        )
    return ", ".join(parts)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("PARAMS", style="green")
    table.add_column("DESCRIPTION")
    for entry in REGISTRY.entries():
        description = " ".join((entry.description or "").split())
        table.add_row(entry.key, entry.name, _format_params(entry), description)
    return table
