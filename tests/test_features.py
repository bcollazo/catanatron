from types import SimpleNamespace

import catanatron.features as features
from catanatron.models.enums import WOOD, SETTLEMENT, CITY
from catanatron.models.player import Color


def test_build_production_features_robber_only_blocks_robbed_tile(monkeypatch):
    """
    Regression test for the robber over-blocking bug.

    Scenario:
    - RED has a settlement on a node that touches the robber tile,
      but that same node also touches an unrobbed WOOD tile.
    - Total WOOD production at the node is 5.
    - Effective WOOD production with the robber considered should still be 2,
      not 0.

    Old behavior:
        The whole node is skipped because it appears in robbed_nodes.
    Fixed behavior:
        get_node_production(..., robber_coordinate) decides what portion
        of the node's production is actually blocked.
    """
    target_node = 7
    robber_coordinate = "robber_tile"

    game = SimpleNamespace(
        state=SimpleNamespace(
            colors=(Color.RED, Color.BLUE),
            board=SimpleNamespace(
                robber_coordinate=robber_coordinate,
                map=SimpleNamespace(
                    tiles={
                        robber_coordinate: SimpleNamespace(
                            nodes={0: target_node}
                        )
                    }
                ),
            ),
        )
    )

    def fake_get_player_buildings(state, color, building_type):
        if color == Color.RED and building_type == SETTLEMENT:
            return [target_node]
        if building_type == CITY:
            return []
        return []

    def fake_get_node_production(catan_map, node_id, resource, robber_coordinate=None):
        if node_id != target_node or resource != WOOD:
            return 0

        # Total WOOD production from this node is 5.
        # With the robber on one adjacent tile, only part of that production
        # should be removed, leaving 2.
        return 2 if robber_coordinate is not None else 5

    monkeypatch.setattr(features, "get_player_buildings", fake_get_player_buildings)
    monkeypatch.setattr(features, "get_node_production", fake_get_node_production)

    total = features.build_production_features(False)(game, Color.RED)
    effective = features.build_production_features(True)(game, Color.RED)

    assert total["TOTAL_P0_WOOD_PRODUCTION"] == 5
    assert effective["EFFECTIVE_P0_WOOD_PRODUCTION"] == 2