"""Small, reproducible Python baseline and real-board fixture exporter.

Run from any directory with Python >=3.11 and NetworkX installed.
No changes to the production engine; all generated files live beside this script.
"""
import cProfile
import io
import json
import os
from pathlib import Path
import platform
import pstats
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "catanatron"))
local_deps = ROOT / ".venv" / "planning-deps"
if local_deps.exists():
    sys.path.insert(0, str(local_deps))
from catanatron import Game, RandomPlayer, Color
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import get_edges
from catanatron.models.enums import CITY
from catanatron.json import GameEncoder


def new_game(seed):
    return Game([RandomPlayer(c) for c in Color], seed=seed)


def bench(name, fn, iterations=500, repeats=7):
    fn()  # warmup
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for i in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return dict(name=name, unit="ns/op", iterations=iterations,
                samples=samples, median=statistics.median(samples),
                min=min(samples), max=max(samples))


def main():
    games, full = [], []
    profile = cProfile.Profile()
    for seed in range(8):
        game = new_game(seed)
        ticks = 0
        while game.winning_color() is None and game.state.num_turns < 1000:
            if ticks % 32 == 0:
                games.append(game.copy())
            game.play_tick()
            ticks += 1
        full.append(dict(seed=seed, ticks=ticks, turns=game.state.num_turns,
                         winner=str(game.winning_color())))
    # Preserve phase diversity across the entire corpus, including setup.
    corpus = [games[i * (len(games) - 1) // 127] for i in range(128)]
    edges = sorted(tuple(sorted(e)) for e in get_edges())
    assert len(edges) == 72
    lines = [" ".join(str(n) for e in edges for n in e)]
    road_mismatches = 0
    for game in corpus:
        s, board = game.state, game.state.board
        owners = [0] * 54
        for node, (color, _) in board.buildings.items():
            owners[node] = 1 + s.color_to_index[color]
        roads = [0 if board.roads.get(e) is None else
                 1 + s.color_to_index[board.roads[e]] for e in edges]
        player = s.current_player_index + 1
        incident = {n for e, owner in zip(edges, roads) if owner == player for n in e}
        expected = {e for e, owner in zip(edges, roads) if owner == 0 and any(
            owners[n] == player or (owners[n] == 0 and n in incident) for n in e)}
        actual = {tuple(sorted(e)) for e in board.buildable_edges(s.current_color())}
        road_mismatches += expected != actual
        lines.append(" ".join(map(str, [player, *owners, *roads])))
    (HERE / "fixtures.txt").write_text("\n".join(lines) + "\n")
    mid = min(corpus, key=lambda g: abs(len(g.state.action_records) - 512))
    metrics = []
    late = max(corpus, key=lambda g: len(g.state.action_records))
    for label, game in [("setup", corpus[0]), ("midgame", mid), ("late", late)]:
        metrics.append(bench(f"copy_{label}", game.copy))
        metrics.append(bench(f"generate_warm_{label}", lambda: generate_playable_actions(game.state)))
    metrics.append(bench("render_json_midgame", lambda: json.dumps(mid, cls=GameEncoder), 100))
    def copy_execute():
        child = mid.copy()
        child.execute(child.playable_actions[0], validate_action=False)
    metrics.append(bench("copy_execute_midgame_first_action", copy_execute))
    def eight_games():
        ticks = 0
        for seed in range(8):
            game = new_game(seed)
            game.play()
            ticks += len(game.state.action_records)
        return ticks
    metrics.append(bench("eight_random_games", eight_games, 1, 5))
    profile.enable()
    eight_games()
    profile.disable()
    out = io.StringIO()
    pstats.Stats(profile, stream=out).strip_dirs().sort_stats("tottime").print_stats(30)
    (HERE / "python-profile.txt").write_text(out.getvalue().rstrip() + "\n")
    result = dict(python=sys.version, platform=platform.platform(),
                  processor=os.environ.get("PROCESSOR_IDENTIFIER"),
                  pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                  corpus_size=len(corpus), road_query_mismatches=road_mismatches,
                  corpus_history_lengths=[len(g.state.action_records) for g in corpus],
                  games=full, metrics=metrics)
    (HERE / "python-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"corpus_size":len(corpus), "road_query_mismatches":road_mismatches,
                      "metrics":[(m["name"], round(m["median"])) for m in metrics]}, indent=2))


if __name__ == "__main__":
    main()
