#!/usr/bin/env python3
"""
Aggregate statistics on initial placements from synthetic Catan game JSONs.
Compares resources and numbers that winners vs losers place on.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "my-data-path"


def get_initial_settlements(data):
    """Extract initial BUILD_SETTLEMENT actions (2 per player in snake order).
    Works for 2, 3, or 4 players: expects 4, 6, or 8 settlements before first non-placement action.
    """
    colors = data.get("colors", [])
    num_players = len(colors)
    if num_players < 2:
        return []
    expected_settlements = 2 * num_players  # 4, 6, or 8
    settlements = []  # list of (color, node_id)
    for record in data.get("action_records", []):
        action = record[0]
        if not action or action[1] != "BUILD_SETTLEMENT":
            continue
        color, _, node_id = action
        settlements.append((color, node_id))
        if len(settlements) >= expected_settlements:
            break
    return settlements


def get_resources_and_numbers_for_node(data, node_id):
    """For a node, return list of (resource, number) from adjacent tiles (skip desert)."""
    adj = data.get("adjacent_tiles", {})
    key = str(node_id)
    if key not in adj:
        return []
    result = []
    for tile in adj[key]:
        if tile.get("type") == "DESERT":
            continue
        res = tile.get("resource")
        num = tile.get("number")
        if res and num is not None:
            result.append((res, num))
    return result


def analyze():
    resource_counts_winner = defaultdict(int)  # resource -> count
    resource_counts_loser = defaultdict(int)
    number_counts_winner = defaultdict(int)   # number -> count
    number_counts_loser = defaultdict(int)
    pair_counts_winner = defaultdict(int)    # (resource, number) -> count
    pair_counts_loser = defaultdict(int)

    games_processed = 0
    skip_reasons = defaultdict(list)  # reason -> list of filenames

    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            skip_reasons["load_error"].append(path.name)
            continue

        winning_color = data.get("winning_color")
        if not winning_color:
            skip_reasons["no_winning_color"].append(path.name)
            continue

        colors = data.get("colors", [])
        expected_settlements = 2 * len(colors) if len(colors) >= 2 else 8
        settlements = get_initial_settlements(data)
        if len(settlements) < expected_settlements:
            skip_reasons["incomplete_initial_placements"].append(path.name)
            continue

        for color, node_id in settlements:
            pairs = get_resources_and_numbers_for_node(data, node_id)
            is_winner = color == winning_color
            for res, num in pairs:
                if is_winner:
                    resource_counts_winner[res] += 1
                    number_counts_winner[num] += 1
                    pair_counts_winner[(res, num)] += 1
                else:
                    resource_counts_loser[res] += 1
                    number_counts_loser[num] += 1
                    pair_counts_loser[(res, num)] += 1

        games_processed += 1

    # Totals for normalization
    total_winner_tiles = sum(resource_counts_winner.values())
    total_loser_tiles = sum(resource_counts_loser.values())

    def pct(c, total):
        return 100 * c / total if total else 0

    games_skipped = sum(len(v) for v in skip_reasons.values())

    print("=" * 60)
    print("INITIAL PLACEMENT STATS (synthetic Catan data)")
    print("=" * 60)
    print(f"Games analyzed: {games_processed}  (skipped: {games_skipped})")
    if skip_reasons:
        print("\nSkip reasons:")
        for reason, files in sorted(skip_reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {reason}: {len(files)} games")
            if len(files) <= 5:
                for f in files:
                    print(f"    - {f}")
            else:
                for f in files[:3]:
                    print(f"    - {f}")
                print(f"    ... and {len(files) - 3} more")
    print(f"Winner tile touches: {total_winner_tiles}  |  Loser tile touches: {total_loser_tiles}")
    print()

    print("--- RESOURCES (what winners vs losers place on) ---")
    print(f"{'Resource':<10} {'Winners':>12} {'%':>8}  {'Losers':>12} {'%':>8}")
    print("-" * 52)
    for res in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
        w = resource_counts_winner[res]
        l = resource_counts_loser[res]
        print(f"{res:<10} {w:>12} {pct(w, total_winner_tiles):>7.1f}%  {l:>12} {pct(l, total_loser_tiles):>7.1f}%")
    print()

    print("--- NUMBERS (dice values winners vs losers place on) ---")
    print(f"{'Number':<10} {'Winners':>12} {'%':>8}  {'Losers':>12} {'%':>8}")
    print("-" * 52)
    for num in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]:
        w = number_counts_winner[num]
        l = number_counts_loser[num]
        print(f"{num:<10} {w:>12} {pct(w, total_winner_tiles):>7.1f}%  {l:>12} {pct(l, total_loser_tiles):>7.1f}%")
    print()

    print("--- TOP (resource, number) PAIRS - WINNERS ---")
    for (res, num), c in sorted(pair_counts_winner.items(), key=lambda x: -x[1])[:15]:
        print(f"  {res} on {num}: {c} ({pct(c, total_winner_tiles):.1f}%)")
    print()

    print("--- TOP (resource, number) PAIRS - LOSERS ---")
    for (res, num), c in sorted(pair_counts_loser.items(), key=lambda x: -x[1])[:15]:
        print(f"  {res} on {num}: {c} ({pct(c, total_loser_tiles):.1f}%)")
    print()

    # Average number strength: 6 and 8 are best, 2 and 12 worst
    def avg_number(counts, total):
        if total == 0:
            return 0
        # probability of rolling (2-12)
        prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return sum(counts.get(n, 0) * prob.get(n, 0) for n in [2,3,4,5,6,8,9,10,11,12]) / total

    print("--- SUMMARY ---")
    print(f"Winners: most common resources = {sorted(resource_counts_winner, key=resource_counts_winner.get, reverse=True)[:4]}")
    print(f"Losers:  most common resources = {sorted(resource_counts_loser, key=resource_counts_loser.get, reverse=True)[:4]}")
    print(f"Winners: most common numbers   = {sorted(number_counts_winner, key=number_counts_winner.get, reverse=True)[:4]}")
    print(f"Losers:  most common numbers   = {sorted(number_counts_loser, key=number_counts_loser.get, reverse=True)[:4]}")


if __name__ == "__main__":
    analyze()
