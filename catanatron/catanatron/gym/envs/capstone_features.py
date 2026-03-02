from collections import Counter

import numpy as np

from catanatron.models.player import Color
from catanatron.models.map import DICE_PROBAS
from catanatron.models.enums import (
    RESOURCES,
    DEVELOPMENT_CARDS,
    VICTORY_POINT,
    SETTLEMENT,
    CITY,
)
from catanatron.models.board import get_edges
from catanatron.models.decks import (
    starting_devcard_bank,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
)
from catanatron.state_functions import get_player_buildings

CLAIM_ARMY_SIZE = 3


def get_hex_features(game) -> list:
    hex_features = []
    robber_coord = game.state.board.robber_coordinate
    for coord, tile in game.state.board.map.land_tiles.items():
        tile_prob = DICE_PROBAS[tile.number]
        resource_pips = [
            tile_prob if resource == tile.resource else 0.0
            for resource in RESOURCES
        ]
        has_robber = 1 if coord == robber_coord else 0
        hex_features.extend(resource_pips + [has_robber])
    return hex_features


def get_vertex_features(game, self_color: Color, opp_color: Color) -> list:
    vertex_features = []
    buildings = game.state.board.buildings
    resource_to_port_nodes = game.state.board.map.port_nodes
    port_node_to_resource = {
        node_id: resource
        for resource, id_set in resource_to_port_nodes.items()
        for node_id in id_set
    }
    robber_coord = game.state.board.robber_coordinate
    tile_id_to_coord = {
        t.id: c for c, t in game.state.board.map.land_tiles.items()
    }

    for node_id, pip_counts in game.state.board.map.node_production.items():
        # settlement / city status
        if node_id in buildings:
            building_color, building_type = buildings[node_id]
            settlement_status = 0.5 if building_type == SETTLEMENT else 1.0
            if building_color == opp_color:
                settlement_status *= -1
        else:
            settlement_status = 0.0

        resource_pips = [
            pip_counts[r] if r in pip_counts else 0.0 for r in RESOURCES
        ]
        total_pips = sum(pip_counts.values())

        # port trade bonus
        if node_id not in port_node_to_resource:
            port_trade = [0] * 5
        else:
            port_type = port_node_to_resource[node_id]
            if port_type is None:
                port_trade = [1] * 5
            else:
                port_trade = [
                    2.0 if port_type == r else 0.0 for r in RESOURCES
                ]

        # robber impact
        impacted_by_robber = 0
        for tile in game.state.board.map.adjacent_tiles[node_id]:
            if (
                tile.resource is not None
                and tile_id_to_coord.get(tile.id) == robber_coord
            ):
                impacted_by_robber = 1

        is_buildable = (
            1.0 if node_id in game.state.board.board_buildable_ids else 0.0
        )

        vertex_features.extend(
            [settlement_status]
            + resource_pips
            + [total_pips]
            + port_trade
            + [impacted_by_robber, is_buildable]
        )
    return vertex_features


def get_edge_features(game, self_color: Color, opp_color: Color) -> list:
    edge_features = []
    self_buildable = game.state.board.buildable_edges(self_color)
    opp_buildable = game.state.board.buildable_edges(opp_color)

    for edge in get_edges():
        road_color = game.state.board.get_edge_color(edge)
        if road_color is None:
            road_status = 0
        elif road_color == self_color:
            road_status = 1
        else:
            road_status = -1

        self_can_build = edge in self_buildable
        opp_can_build = edge in opp_buildable
        adj_vertex_available = 1.0 if (
            edge[0] in game.state.board.board_buildable_ids
            or edge[1] in game.state.board.board_buildable_ids
        ) else 0.0

        edge_features.extend([
            road_status, self_can_build, opp_can_build, adj_vertex_available,
        ])
    return edge_features


def get_hand_features(game, self_color: Color, opp_color: Color) -> list:
    ps = game.state.player_state
    c2i = game.state.color_to_index
    si, oi = c2i[self_color], c2i[opp_color]

    self_res = [ps[f"P{si}_{r}_IN_HAND"] for r in RESOURCES]
    self_total_res = sum(self_res)
    self_over_limit = int(self_total_res > game.state.discard_limit)
    self_devs = [ps[f"P{si}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS]
    self_total_devs = sum(self_devs)
    self_buildings = [
        ps[f"P{si}_{b}_AVAILABLE"] for b in ("ROADS", "SETTLEMENTS", "CITIES")
    ]

    opp_res = [ps[f"P{oi}_{r}_IN_HAND"] for r in RESOURCES]
    opp_total_res = sum(opp_res)
    opp_over_limit = int(opp_total_res > game.state.discard_limit)
    opp_devs = [ps[f"P{oi}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS]
    opp_total_devs = sum(opp_devs)
    opp_buildings = [
        ps[f"P{oi}_{b}_AVAILABLE"] for b in ("ROADS", "SETTLEMENTS", "CITIES")
    ]

    return [
        *self_res, self_total_res, self_over_limit,
        *self_devs, self_total_devs,
        *self_buildings,
        *opp_res, opp_total_res, opp_over_limit,
        opp_total_devs,
        *opp_buildings,
    ]


# ── helpers for strategic features ──────────────────────────────

def _player_production(game, color, consider_robber=False):
    board = game.state.board
    robber_coord = board.robber_coordinate
    tile_id_to_coord = {
        t.id: c for c, t in board.map.land_tiles.items()
    }
    productions = []
    for resource in RESOURCES:
        prod = 0.0
        for node_id in get_player_buildings(game.state, color, SETTLEMENT):
            for t in board.map.adjacent_tiles[node_id]:
                if t.resource == resource and (
                    not consider_robber
                    or tile_id_to_coord.get(t.id) != robber_coord
                ):
                    prod += DICE_PROBAS[t.number]
        for node_id in get_player_buildings(game.state, color, CITY):
            for t in board.map.adjacent_tiles[node_id]:
                if t.resource == resource and (
                    not consider_robber
                    or tile_id_to_coord.get(t.id) != robber_coord
                ):
                    prod += 2 * DICE_PROBAS[t.number]
        productions.append(prod)
    return productions


def _income_diversity(pip_production):
    pips = np.array(pip_production, dtype=float)
    total = pips.sum()
    if total == 0:
        return 0.0
    probs = pips / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return entropy / np.log(len(pips))


def _player_numbers(game, color):
    board = game.state.board
    buildings = game.state.buildings_by_color[color]
    coverage = {n: 0 for n in range(2, 13) if n != 7}
    for btype, node_ids in buildings.items():
        if btype == "ROAD":
            continue
        mult = 2 if btype == CITY else 1
        for node_id in node_ids:
            for tile in board.map.adjacent_tiles[node_id]:
                if tile.number is not None:
                    coverage[tile.number] += mult
    return coverage


def _roll_diversity(game, color):
    counts = _player_numbers(game, color)
    vals = np.array(list(counts.values()), dtype=float)
    covered = vals[vals > 0]
    if len(covered) == 0:
        return 0.0
    total = covered.sum()
    probs = covered / total
    entropy = -np.sum(probs * np.log(probs))
    return entropy / np.log(len(vals))


# ────────────────────────────────────────────────────────────────

def get_strategic_features(game, self_color: Color, opp_color: Color) -> list:
    ps = game.state.player_state
    c2i = game.state.color_to_index
    si, oi = c2i[self_color], c2i[opp_color]

    # victory points
    self_vp = ps[f"P{si}_VICTORY_POINTS"]
    opp_vp = ps[f"P{oi}_VICTORY_POINTS"]
    self_vp_to_win = game.vps_to_win - self_vp
    opp_vp_to_win = game.vps_to_win - opp_vp
    vp_diff = self_vp - opp_vp

    # roads
    self_road_len = game.state.board.road_lengths[self_color]
    opp_road_len = game.state.board.road_lengths[opp_color]
    who_has_road = ps[f"P{si}_HAS_ROAD"] - ps[f"P{oi}_HAS_ROAD"]
    self_dist_road = game.state.board.road_length - self_road_len
    opp_dist_road = game.state.board.road_length - opp_road_len
    road_diff = self_road_len - opp_road_len

    # army
    self_army = ps[f"P{si}_PLAYED_KNIGHT"]
    opp_army = ps[f"P{oi}_PLAYED_KNIGHT"]
    who_has_army = ps[f"P{si}_HAS_ARMY"] - ps[f"P{oi}_HAS_ARMY"]
    army_diff = self_army - opp_army
    if who_has_army == 0:
        self_dist_army = CLAIM_ARMY_SIZE - self_army
    elif who_has_army == 1:
        self_dist_army = 0
    else:
        self_dist_army = -(army_diff - 1)
    if who_has_army == 0:
        opp_dist_army = CLAIM_ARMY_SIZE - opp_army
    elif who_has_army == -1:
        opp_dist_army = 0
    else:
        opp_dist_army = army_diff + 1

    # production
    self_pips = _player_production(game, self_color, consider_robber=False)
    self_total_pips = sum(self_pips)
    opp_pips = _player_production(game, opp_color, consider_robber=False)
    opp_total_pips = sum(opp_pips)
    pip_diff = [self_pips[i] - opp_pips[i] for i in range(len(self_pips))]

    # robber harm
    self_robbed = _player_production(game, self_color, consider_robber=True)
    opp_robbed = _player_production(game, opp_color, consider_robber=True)
    self_harm = sum(self_pips[i] - self_robbed[i] for i in range(len(self_pips)))
    opp_harm = sum(opp_pips[i] - opp_robbed[i] for i in range(len(opp_pips)))

    # trade rates
    self_ports = game.state.board.get_player_port_resources(self_color)
    self_3to1 = None in self_ports
    self_rates = [
        1 / 2 if r in self_ports else 1 / 3 if self_3to1 else 1 / 4
        for r in RESOURCES
    ]
    opp_ports = game.state.board.get_player_port_resources(opp_color)
    opp_3to1 = None in opp_ports
    opp_rates = [
        1 / 2 if r in opp_ports else 1 / 3 if opp_3to1 else 1 / 4
        for r in RESOURCES
    ]

    # diversity
    self_pip_div = _income_diversity(self_pips)
    opp_pip_div = _income_diversity(opp_pips)
    self_roll_div = _roll_diversity(game, self_color)
    opp_roll_div = _roll_diversity(game, opp_color)

    return [
        self_vp_to_win, opp_vp_to_win, vp_diff,
        self_dist_road, opp_dist_road, road_diff, who_has_road,
        self_dist_army, opp_dist_army, army_diff, who_has_army,
        *self_pips, self_total_pips,
        *opp_pips, opp_total_pips,
        *pip_diff,
        self_harm, opp_harm,
        *self_rates, *opp_rates,
        self_pip_div, opp_pip_div,
        self_roll_div, opp_roll_div,
    ]


def _item_affordability(resources, costs):
    can_afford = True
    total_cost = sum(costs)
    needed = 0
    for amt, cost in zip(resources, costs):
        if amt < cost:
            can_afford = False
        needed += max(0, cost - amt)
    return can_afford, (total_cost - needed) / total_cost


def _get_dev_card_features(game):
    deck = starting_devcard_bank()
    deck_counts = Counter(deck)
    remaining = len(game.state.development_listdeck)
    ps = game.state.player_state
    played = {}
    for dc in DEVELOPMENT_CARDS:
        if dc == VICTORY_POINT:
            continue
        total_played = sum(
            ps[f"P{i}_PLAYED_{dc}"] for i in range(len(game.state.colors))
        )
        played[dc] = total_played
    pct_remaining = remaining / len(deck)
    pct_played = [
        played[dc] / deck_counts[dc]
        for dc in DEVELOPMENT_CARDS
        if dc != VICTORY_POINT
    ]
    return [pct_remaining, *pct_played]


def get_game_features(game, self_color: Color, opp_color: Color) -> list:
    ps = game.state.player_state
    c2i = game.state.color_to_index
    si, oi = c2i[self_color], c2i[opp_color]

    turn_num = game.state.num_turns
    dev_feats = _get_dev_card_features(game)

    self_res = [ps[f"P{si}_{r}_IN_HAND"] for r in RESOURCES]
    self_can_place_sett = float(
        len(game.state.board.buildable_node_ids(self_color)) > 0
    )
    self_can_place_road = float(
        len(game.state.board.buildable_edges(self_color)) > 0
    )
    self_can_place_city = float(
        len(game.state.buildings_by_color[self_color][SETTLEMENT]) > 0
    )
    sa_road, sp_road = _item_affordability(self_res, ROAD_COST_FREQDECK)
    sa_sett, sp_sett = _item_affordability(self_res, SETTLEMENT_COST_FREQDECK)
    sa_city, sp_city = _item_affordability(self_res, CITY_COST_FREQDECK)
    sa_dev, sp_dev = _item_affordability(self_res, DEVELOPMENT_CARD_COST_FREQDECK)

    opp_res = [ps[f"P{oi}_{r}_IN_HAND"] for r in RESOURCES]
    opp_can_place_sett = float(
        len(game.state.board.buildable_node_ids(opp_color)) > 0
    )
    opp_can_place_road = float(
        len(game.state.board.buildable_edges(opp_color)) > 0
    )
    opp_can_place_city = float(
        len(game.state.buildings_by_color[opp_color][SETTLEMENT]) > 0
    )
    oa_road, op_road = _item_affordability(opp_res, ROAD_COST_FREQDECK)
    oa_sett, op_sett = _item_affordability(opp_res, SETTLEMENT_COST_FREQDECK)
    oa_city, op_city = _item_affordability(opp_res, CITY_COST_FREQDECK)
    oa_dev, op_dev = _item_affordability(opp_res, DEVELOPMENT_CARD_COST_FREQDECK)

    return [
        turn_num / 100,
        *dev_feats,
        self_can_place_sett, self_can_place_road, self_can_place_city,
        float(sa_road), float(sa_sett), float(sa_city), float(sa_dev),
        sp_road, sp_sett, sp_city, sp_dev,
        opp_can_place_sett, opp_can_place_road, opp_can_place_city,
        float(oa_road), float(oa_sett), float(oa_city), float(oa_dev),
        op_road, op_sett, op_city, op_dev,
    ]


# ── master observation builder ──────────────────────────────────

def get_capstone_observation(game, self_color: Color, opp_color: Color) -> list:
    return (
        get_hex_features(game)
        + get_vertex_features(game, self_color, opp_color)
        + get_edge_features(game, self_color, opp_color)
        + get_hand_features(game, self_color, opp_color)
        + get_strategic_features(game, self_color, opp_color)
        + get_game_features(game, self_color, opp_color)
    )
