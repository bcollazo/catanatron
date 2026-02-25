"""
PydanticAI tools for the Catan LLM agent.

Provides 8 tools for game state analysis and decision making:
1. get_game_state - Comprehensive game state
2. get_available_actions - List of valid actions
3. evaluate_action - Evaluate immediate impact of an action
4. simulate_action - Simulate future states
5. get_opponent_info - Opponent analysis
6. get_resource_production - Production probabilities
7. get_winning_probability - Win probability estimate
8. analyze_board - Strategic board analysis
"""

from typing import List, Dict, Any
from pydantic_ai import Agent, RunContext

from catanatron.players.llm.base import CatanDependencies

from catanatron.state_functions import (
    player_key,
    get_actual_victory_points,
    get_visible_victory_points,
    get_player_freqdeck,
    player_num_resource_cards,
    get_dev_cards_in_hand,
    get_played_dev_cards,
    get_longest_road_length,
    get_longest_road_color,
    get_largest_army,
    get_player_buildings,
)
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY, ROAD
from catanatron.models.map import LandTile
from catanatron.players.llm.state_formatter import StateFormatter


def register_tools(agent: Agent) -> None:
    """
    Register all tools with the agent.

    Args:
        agent: The PydanticAI agent to register tools on
    """

    @agent.tool()
    def get_game_and_action_analysis(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
        """
        Combine the analysis of the board into a single analysis.
        """

        full_analysis = {
            "game_state": StateFormatter.format_full_state(ctx.deps.game, ctx.deps.color),
            "available_actions": [StateFormatter.format_action(action, i) for i, action in enumerate(ctx.deps.playable_actions)],
            "strategy_recommendation": ctx.deps.strategy_recommendation if ctx.deps.strategy_recommendation else "No strategy advisor configured",
            "strategy_reasoning": ctx.deps.strategy_reasoning if ctx.deps.strategy_reasoning else "No detailed reasoning available",
        }
        return full_analysis

    # @agent.tool
    # def get_game_state(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get comprehensive current game state including your resources,
    #     buildings, opponents, and board information.

    #     Use this to understand the overall game situation before making decisions.
    #     """
    #     return StateFormatter.format_full_state(ctx.deps.game, ctx.deps.color)

    # @agent.tool
    # def get_available_actions(ctx: RunContext[CatanDependencies]) -> List[Dict[str, Any]]:
    #     """
    #     Get list of all valid actions you can take right now.

    #     Each action has an index, type, and description.
    #     Return the index of your chosen action.
    #     """
    #     actions = []
    #     for i, action in enumerate(ctx.deps.playable_actions):
    #         actions.append(StateFormatter.format_action(action, i))
    #     return actions

    # @agent.tool
    # def get_strategy_recommendation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get the strategy advisor's recommended action and reasoning.

    #     The strategy advisor uses algorithmic analysis (like AlphaBeta search)
    #     to suggest the best action. You can follow or override this recommendation.
    #     """
    #     if ctx.deps.strategy_recommendation is None:
    #         return {
    #             "has_recommendation": False,
    #             "message": "No strategy advisor configured",
    #         }

    #     # Find the index of the recommended action
    #     rec_index = None
    #     for i, action in enumerate(ctx.deps.playable_actions):
    #         if action == ctx.deps.strategy_recommendation:
    #             rec_index = i
    #             break

    #     return {
    #         "has_recommendation": True,
    #         "recommended_action_index": rec_index,
    #         "action": StateFormatter.format_action(ctx.deps.strategy_recommendation, rec_index or 0),
    #         "reasoning": ctx.deps.strategy_reasoning or "No detailed reasoning available",
    #     }

    # @agent.tool
    # def evaluate_action(
    #     ctx: RunContext[CatanDependencies], action_index: int
    # ) -> Dict[str, Any]:
    #     """
    #     Evaluate a specific action's immediate impact.

    #     Args:
    #         action_index: Index of the action to evaluate (from get_available_actions)

    #     Returns:
    #         Dictionary with VP change, resource changes, and other impacts.
    #     """
    #     if action_index < 0 or action_index >= len(ctx.deps.playable_actions):
    #         return {"error": f"Invalid action index: {action_index}"}

    #     action = ctx.deps.playable_actions[action_index]
    #     game = ctx.deps.game
    #     my_color = ctx.deps.color

    #     # Get current state
    #     before_vps = get_actual_victory_points(game.state, my_color)
    #     before_resources = get_player_freqdeck(game.state, my_color)

    #     # Simulate the action
    #     game_copy = game.copy()
    #     try:
    #         game_copy.execute(action)
    #     except Exception as e:
    #         return {"error": f"Failed to simulate action: {str(e)}"}

    #     # Get new state
    #     after_vps = get_actual_victory_points(game_copy.state, my_color)
    #     after_resources = get_player_freqdeck(game_copy.state, my_color)

    #     # Calculate changes
    #     resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    #     resource_changes = {
    #         name: after_resources[i] - before_resources[i]
    #         for i, name in enumerate(resource_names)
    #     }

    #     return {
    #         "action": StateFormatter.format_action(action, action_index),
    #         "vp_change": after_vps - before_vps,
    #         "resource_changes": resource_changes,
    #         "new_resource_totals": {
    #             name: after_resources[i] for i, name in enumerate(resource_names)
    #         },
    #     }

    # @agent.tool
    # def simulate_action(
    #     ctx: RunContext[CatanDependencies], action_index: int
    # ) -> Dict[str, Any]:
    #     """
    #     Simulate taking an action and show the resulting game state.

    #     This helps you see what the board will look like after the action.

    #     Args:
    #         action_index: Index of the action to simulate
    #     """
    #     if action_index < 0 or action_index >= len(ctx.deps.playable_actions):
    #         return {"error": f"Invalid action index: {action_index}"}

    #     action = ctx.deps.playable_actions[action_index]
    #     game = ctx.deps.game

    #     # Simulate the action
    #     game_copy = game.copy()
    #     try:
    #         game_copy.execute(action)
    #     except Exception as e:
    #         return {"error": f"Failed to simulate action: {str(e)}"}

    #     # Return the new state summary
    #     return {
    #         "action_taken": StateFormatter.format_action(action, action_index),
    #         "resulting_state": StateFormatter.format_full_state(game_copy, ctx.deps.color),
    #         "new_available_actions_count": len(game_copy.playable_actions),
    #     }

    # @agent.tool
    # def get_opponent_info(ctx: RunContext[CatanDependencies]) -> List[Dict[str, Any]]:
    #     """
    #     Get detailed information about all opponents.

    #     Includes their visible VPs, card counts, buildings, and achievements.
    #     """
    #     opponents = []
    #     state = ctx.deps.game.state
    #     my_color = ctx.deps.color

    #     for color in state.colors:
    #         if color == my_color:
    #             continue

    #         key = player_key(state, color)
    #         buildings = state.buildings_by_color.get(color, {})

    #         opponent_info = {
    #             "color": color.value,
    #             "visible_victory_points": get_visible_victory_points(state, color),
    #             "resource_card_count": player_num_resource_cards(state, color),
    #             "dev_card_count": get_dev_cards_in_hand(state, color),
    #             "played_knights": get_played_dev_cards(state, color, "KNIGHT"),
    #             "longest_road_length": get_longest_road_length(state, color),
    #             "has_longest_road": state.player_state.get(f"{key}_HAS_ROAD", False),
    #             "has_largest_army": state.player_state.get(f"{key}_HAS_ARMY", False),
    #             "settlements": len(buildings.get(SETTLEMENT, [])),
    #             "cities": len(buildings.get(CITY, [])),
    #             "roads": len(buildings.get(ROAD, [])),
    #             "threat_assessment": _assess_threat(state, color, my_color),
    #         }
    #         opponents.append(opponent_info)

    #     # Sort by VPs descending
    #     opponents.sort(key=lambda x: x["visible_victory_points"], reverse=True)
    #     return opponents

    # @agent.tool
    # def get_resource_production(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Get expected resource production probabilities for all players.

    #     Shows which resources each player is likely to receive on average per turn,
    #     accounting for settlements, cities, and the robber position.
    #     """
    #     from catanatron.features import build_production_features

    #     game = ctx.deps.game
    #     my_color = ctx.deps.color

    #     # Use the production features from the game
    #     prod_fn = build_production_features(consider_robber=True)
    #     production = prod_fn(game, my_color)

    #     # Format the production data
    #     result = {"my_production": {}, "opponent_production": {}}

    #     resource_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

    #     # My production (P0 in the features)
    #     for resource in resource_names:
    #         key = f"EFFECTIVE_P0_{resource}_PRODUCTION"
    #         result["my_production"][resource.lower()] = round(production.get(key, 0), 3)

    #     # Calculate total for me
    #     result["my_production"]["total"] = round(
    #         sum(result["my_production"].values()), 3
    #     )

    #     # Opponent production
    #     for i, color in enumerate(game.state.colors):
    #         if color == my_color:
    #             continue
    #         # Find the player index in the feature output
    #         player_idx = (i - game.state.colors.index(my_color)) % len(game.state.colors)
    #         opp_prod = {}
    #         for resource in resource_names:
    #             key = f"EFFECTIVE_P{player_idx}_{resource}_PRODUCTION"
    #             opp_prod[resource.lower()] = round(production.get(key, 0), 3)
    #         opp_prod["total"] = round(sum(opp_prod.values()), 3)
    #         result["opponent_production"][color.value] = opp_prod

    #     return result

    # @agent.tool
    # def get_winning_probability(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    #     """
    #     Estimate winning probability based on current position.

    #     Uses a heuristic value function to estimate each player's strength.
    #     """
    #     game = ctx.deps.game
    #     state = game.state
    #     my_color = ctx.deps.color

    #     # Calculate a simple score for each player
    #     scores = {}
    #     for color in state.colors:
    #         key = player_key(state, color)

    #         # Base score from VPs
    #         vps = get_visible_victory_points(state, color)
    #         score = vps * 100

    #         # Add for resources (flexibility)
    #         cards = player_num_resource_cards(state, color)
    #         score += min(cards, 10) * 5  # Cap to avoid overvaluing large hands

    #         # Add for dev cards
    #         dev_cards = get_dev_cards_in_hand(state, color)
    #         score += dev_cards * 15

    #         # Add for longest road/army
    #         if state.player_state.get(f"{key}_HAS_ROAD", False):
    #             score += 200
    #         if state.player_state.get(f"{key}_HAS_ARMY", False):
    #             score += 200

    #         # Add for road length progress
    #         road_length = get_longest_road_length(state, color)
    #         if road_length >= 4:
    #             score += (road_length - 4) * 20

    #         # Add for army progress
    #         knights = get_played_dev_cards(state, color, "KNIGHT")
    #         if knights >= 2:
    #             score += (knights - 2) * 20

    #         scores[color.value] = score

    #     # Convert to probabilities
    #     total_score = sum(scores.values())
    #     probabilities = {
    #         color: round(score / total_score * 100, 1) if total_score > 0 else 25.0
    #         for color, score in scores.items()
    #     }

    #     my_vps = get_actual_victory_points(state, my_color)

    #     return {
    #         "my_color": my_color.value,
    #         "my_victory_points": my_vps,
    #         "vps_to_win": game.vps_to_win,
    #         "vps_remaining": game.vps_to_win - my_vps,
    #         "estimated_win_probability": probabilities.get(my_color.value, 0),
    #         "all_probabilities": probabilities,
    #         "ranking": sorted(
    #             probabilities.items(), key=lambda x: x[1], reverse=True
    #         ),
    #     }

    @agent.tool
    def analyze_board(
        ctx: RunContext[CatanDependencies], focus: str
    ) -> Dict[str, Any]:
        """
        Analyze the board for strategic insights.

        Args:
            focus: What to analyze. Options:
                - 'expansion': Where can I build next, best spots
                - 'blocking': How to block opponents' expansion
                - 'ports': Port accessibility and trading options
                - 'robber': Optimal robber placements
        """
        game = ctx.deps.game
        state = game.state
        board = state.board
        my_color = ctx.deps.color

        if focus == "expansion":
            return _analyze_expansion(game, my_color)
        elif focus == "blocking":
            return _analyze_blocking(game, my_color)
        elif focus == "ports":
            return _analyze_ports(game, my_color)
        elif focus == "robber":
            return _analyze_robber(game, my_color)
        else:
            return {
                "error": f"Unknown focus: {focus}",
                "valid_options": ["expansion", "blocking", "ports", "robber"],
            }


def _assess_threat(state, opponent_color, my_color) -> str:
    """Assess how threatening an opponent is."""
    opp_vps = get_visible_victory_points(state, opponent_color)
    my_vps = get_visible_victory_points(state, my_color)

    if opp_vps >= 8:
        return "CRITICAL - Close to winning!"
    elif opp_vps >= 6:
        return "HIGH - Strong position"
    elif opp_vps > my_vps:
        return "MEDIUM - Ahead of you"
    else:
        return "LOW - Behind or equal"


def _analyze_expansion(game, my_color) -> Dict[str, Any]:
    """Analyze expansion opportunities."""
    board = game.state.board

    buildable_nodes = list(board.buildable_node_ids(my_color))
    buildable_edges = [list(e) for e in board.buildable_edges(my_color)]

    # Assess each buildable node
    node_assessments = []
    for node_id in buildable_nodes[:5]:  # Limit to top 5
        # Get production at this node
        tiles = board.map.adjacent_tiles.get(node_id, [])
        resources = []
        total_prob = 0
        for tile in tiles:
            if isinstance(tile, LandTile) and tile.resource:
                from catanatron.models.map import number_probability

                prob = number_probability(tile.number) if tile.number else 0
                resources.append(
                    {"resource": tile.resource, "number": tile.number, "probability": prob}
                )
                total_prob += prob

        node_assessments.append(
            {
                "node_id": node_id,
                "resources": resources,
                "total_production_probability": round(total_prob, 3),
            }
        )

    # Sort by production
    node_assessments.sort(
        key=lambda x: x["total_production_probability"], reverse=True
    )

    return {
        "buildable_settlement_locations": len(buildable_nodes),
        "buildable_road_locations": len(buildable_edges),
        "best_settlement_spots": node_assessments,
        "expansion_advice": (
            "Focus on high-probability numbers (6, 8, 5, 9) and resource diversity"
            if buildable_nodes
            else "No settlement spots available - build roads to expand"
        ),
    }


def _analyze_blocking(game, my_color) -> Dict[str, Any]:
    """Analyze blocking opportunities."""
    state = game.state
    board = state.board

    blocking_opportunities = []
    for color in state.colors:
        if color == my_color:
            continue

        # Find nodes that would block this opponent
        their_buildable = list(board.buildable_node_ids(color, initial_build_phase=False))
        my_buildable = set(board.buildable_node_ids(my_color))

        overlap = [n for n in their_buildable if n in my_buildable]
        if overlap:
            blocking_opportunities.append(
                {
                    "opponent": color.value,
                    "blockable_nodes": overlap[:3],
                    "their_vps": get_visible_victory_points(state, color),
                }
            )

    return {
        "blocking_opportunities": blocking_opportunities,
        "advice": (
            "Consider blocking the leading player's expansion"
            if blocking_opportunities
            else "No immediate blocking opportunities"
        ),
    }


def _analyze_ports(game, my_color) -> Dict[str, Any]:
    """Analyze port access and trading."""
    board = game.state.board
    my_ports = list(board.get_player_port_resources(my_color))

    port_info = {
        "current_ports": my_ports,
        "has_3_to_1_port": None in my_ports,
        "specialized_ports": [p for p in my_ports if p is not None],
    }

    # Determine trading rates
    rates = {"wood": 4, "brick": 4, "sheep": 4, "wheat": 4, "ore": 4}
    if None in my_ports:
        rates = {r: 3 for r in rates}
    for port_resource in my_ports:
        if port_resource:
            rates[port_resource.lower()] = 2

    port_info["trading_rates"] = rates
    port_info["advice"] = (
        "Good port access - consider using maritime trades"
        if len(my_ports) > 0
        else "No ports yet - consider building towards coastal settlements"
    )

    return port_info


def _analyze_robber(game, my_color) -> Dict[str, Any]:
    """Analyze robber placement options."""
    state = game.state
    board = state.board

    placements = []
    for coord, tile in board.map.tiles.items():
        # Skip non-land tiles (Water, Port) and desert tiles
        if not isinstance(tile, LandTile) or tile.resource is None:
            continue
        if coord == board.robber_coordinate:
            continue

        # Who would be affected?
        affected = {}
        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id)
            if building:
                color, btype = building
                if color != my_color:
                    if color.value not in affected:
                        affected[color.value] = 0
                    affected[color.value] += 2 if btype == CITY else 1

        if affected:
            from catanatron.models.map import number_probability

            prob = number_probability(tile.number) if tile.number else 0
            placements.append(
                {
                    "coordinate": coord,
                    "resource": tile.resource,
                    "number": tile.number,
                    "probability": prob,
                    "affected_players": affected,
                    "total_impact": sum(affected.values()) * prob,
                }
            )

    # Sort by impact
    placements.sort(key=lambda x: x["total_impact"], reverse=True)

    return {
        "best_robber_placements": placements[:5],
        "current_robber_location": board.robber_coordinate,
        "advice": "Target the leading player's highest-production tile",
    }
