"""Static heuristics for opening settlement and road placement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.map import CatanMap, NUM_NODES


# Strategic resource weights. Ore and wheat drive cities and development cards,
# while wood and brick are kept strong for early expansion.
RESOURCE_WEIGHTS = {
    "ORE": 1.25,
    "WHEAT": 1.25,
    "SHEEP": 0.95,
    "WOOD": 1.0,
    "BRICK": 1.0,
}

EDGE_ORDER = [tuple(sorted(edge)) for edge in get_edges()]
EDGE_TO_INDEX = {edge: idx for idx, edge in enumerate(EDGE_ORDER)}


@dataclass(frozen=True)
class HeuristicConfig:
    """Tunables for the static placement heuristic."""

    resource_weights: dict[str, float] = field(
        default_factory=lambda: RESOURCE_WEIGHTS.copy()
    )
    starting_resource_weights: dict[str, float] = field(
        default_factory=lambda: RESOURCE_WEIGHTS.copy()
    )
    variety_weight: float = 0.06
    complementarity_weight: float = 0.10
    critical_missing_weight: float = 0.06
    generic_port_bonus: float = 0.04
    specific_port_multiplier: float = 0.4
    specific_port_floor: float = 0.02
    number_diversity_weight: float = 0.025
    starting_resource_weight: float = 0.0
    opponent_denial_weight: float = 0.0
    road_new_resource_weight: float = 0.05
    blocked_road_lookahead_discount: float = 0.5
    blocked_road_floor: float = 0.01


DEFAULT_HEURISTIC_CONFIG = HeuristicConfig()


def _node_production(catan_map: CatanMap, node_id: int) -> Counter:
    return catan_map.node_production.get(node_id, Counter())


def _get_port_type(catan_map: CatanMap, node_id: int):
    """Return a resource for 2:1, None for 3:1, or False for no port."""

    for resource, node_set in catan_map.port_nodes.items():
        if node_id in node_set:
            return resource
    return False


def _is_node_buildable(node_id: int, all_settlement_nodes: set[int]) -> bool:
    """Check the Catan distance rule against already occupied nodes."""

    if node_id < 0 or node_id >= NUM_NODES:
        return False
    if node_id in all_settlement_nodes:
        return False
    return all(neighbor not in all_settlement_nodes for neighbor in STATIC_GRAPH.neighbors(node_id))


def _quick_node_score(
    catan_map: CatanMap,
    node_id: int,
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
) -> float:
    """Simplified settlement score for road expansion targets."""

    production = _node_production(catan_map, node_id)
    weighted = sum(
        config.resource_weights.get(resource, 1.0) * pips
        for resource, pips in production.items()
    )
    variety = len(production) * config.variety_weight
    return weighted + variety


def score_settlement(
    catan_map: CatanMap,
    node_id: int,
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
    include_denial: bool = True,
) -> float:
    """Score a candidate settlement node. Higher is better."""

    production = _node_production(catan_map, node_id)
    resources_here = set(production.keys())

    weighted_prod = sum(
        config.resource_weights.get(resource, 1.0) * probability
        for resource, probability in production.items()
    )
    variety_bonus = len(resources_here) * config.variety_weight

    complementarity_bonus = 0.0
    starting_resource_bonus = 0.0
    if my_settlement_nodes:
        existing_resources = set()
        for node in my_settlement_nodes:
            existing_resources.update(_node_production(catan_map, node).keys())

        new_resources = resources_here - existing_resources
        complementarity_bonus = len(new_resources) * config.complementarity_weight
        for critical in ("ORE", "WHEAT"):
            if critical not in existing_resources and critical in resources_here:
                complementarity_bonus += config.critical_missing_weight

        starting_resource_bonus = config.starting_resource_weight * sum(
            config.starting_resource_weights.get(tile.resource, 1.0)
            for tile in catan_map.adjacent_tiles[node_id]
            if tile.resource is not None
        )

    port_bonus = 0.0
    port_type = _get_port_type(catan_map, node_id)
    if port_type is not False:
        if port_type is None:
            port_bonus = config.generic_port_bonus
        else:
            total_prod_of_resource = production.get(port_type, 0.0)
            for node in my_settlement_nodes:
                total_prod_of_resource += _node_production(catan_map, node).get(port_type, 0.0)
            port_bonus = max(
                total_prod_of_resource * config.specific_port_multiplier,
                config.specific_port_floor,
            )

    number_bonus = 0.0
    if my_settlement_nodes:
        existing_numbers = set()
        for node in my_settlement_nodes:
            for tile in catan_map.adjacent_tiles[node]:
                if tile.number is not None:
                    existing_numbers.add(tile.number)

        new_numbers = set()
        for tile in catan_map.adjacent_tiles[node_id]:
            if tile.number is not None and tile.number not in existing_numbers:
                new_numbers.add(tile.number)
        number_bonus = len(new_numbers) * config.number_diversity_weight

    score = (
        weighted_prod
        + variety_bonus
        + complementarity_bonus
        + starting_resource_bonus
        + port_bonus
        + number_bonus
    )

    if include_denial and config.opponent_denial_weight > 0:
        score -= config.opponent_denial_weight * _best_opponent_settlement_score_after(
            catan_map,
            node_id,
            my_settlement_nodes,
            opp_settlement_nodes,
            config,
        )

    return score


def _best_opponent_settlement_score_after(
    catan_map: CatanMap,
    candidate_node_id: int,
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
    config: HeuristicConfig,
) -> float:
    all_after = set(my_settlement_nodes) | set(opp_settlement_nodes) | {candidate_node_id}
    best = 0.0
    for opp_node_id in range(NUM_NODES):
        if not _is_node_buildable(opp_node_id, all_after):
            continue
        score = score_settlement(
            catan_map,
            opp_node_id,
            opp_settlement_nodes,
            [*my_settlement_nodes, candidate_node_id],
            config,
            include_denial=False,
        )
        best = max(best, score)
    return best


def score_road(
    catan_map: CatanMap,
    edge_idx: int,
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
) -> float:
    """Score an opening road by the future settlement prospects it points at."""

    node_a, node_b = EDGE_ORDER[edge_idx]
    all_settlements = set(my_settlement_nodes) | set(opp_settlement_nodes)
    my_set = set(my_settlement_nodes)

    if node_a in my_set and node_b not in my_set:
        far_end = node_b
    elif node_b in my_set and node_a not in my_set:
        far_end = node_a
    else:
        return max(
            _quick_node_score(catan_map, node_a, config),
            _quick_node_score(catan_map, node_b, config),
        )

    if _is_node_buildable(far_end, all_settlements):
        base = _quick_node_score(catan_map, far_end, config)

        existing_resources = set()
        for node in my_settlement_nodes:
            existing_resources.update(_node_production(catan_map, node).keys())
        far_resources = set(_node_production(catan_map, far_end).keys())
        return base + len(far_resources - existing_resources) * config.road_new_resource_weight

    further_scores = []
    for next_node in STATIC_GRAPH.neighbors(far_end):
        if next_node in my_set:
            continue
        if _is_node_buildable(next_node, all_settlements):
            further_scores.append(
                _quick_node_score(catan_map, next_node, config)
                * config.blocked_road_lookahead_discount
            )
    return max(further_scores) if further_scores else config.blocked_road_floor


def score_legal_settlements(
    catan_map: CatanMap,
    legal_node_ids: list[int],
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
) -> dict[int, float]:
    """Score every legal settlement node. Returns ``{node_id: score}``."""

    return {
        node_id: score_settlement(
            catan_map,
            node_id,
            my_settlement_nodes,
            opp_settlement_nodes,
            config,
        )
        for node_id in legal_node_ids
    }


def score_legal_roads(
    catan_map: CatanMap,
    legal_edge_indices: list[int],
    my_settlement_nodes: list[int],
    opp_settlement_nodes: list[int],
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
) -> dict[int, float]:
    """Score every legal road edge. Returns ``{edge_idx: score}``."""

    return {
        edge_idx: score_road(
            catan_map,
            edge_idx,
            my_settlement_nodes,
            opp_settlement_nodes,
            config,
        )
        for edge_idx in legal_edge_indices
    }


def sample_from_scores(scores: dict[int, float], temperature: float) -> tuple[int, float]:
    """Choose from heuristic scores with deterministic argmax or softmax sampling."""

    if not scores:
        raise ValueError("Cannot sample from an empty score table")

    actions = list(scores.keys())
    values = np.asarray([scores[action] for action in actions], dtype=np.float64)

    if temperature <= 1e-8:
        best_idx = int(np.argmax(values))
        return actions[best_idx], 0.0

    logits = values / temperature
    logits -= logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    chosen_idx = int(np.random.choice(len(actions), p=probs))
    log_prob = float(np.log(probs[chosen_idx] + 1e-10))
    return actions[chosen_idx], log_prob
