"""
State formatting utilities for LLM-readable game state.

Converts Catanatron game state into structured dictionaries and
natural language descriptions suitable for LLM consumption.
"""

from typing import Dict, List, Any, Optional

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import (
    Action,
    ActionType,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
)
from catanatron.state_functions import (
    player_key,
    get_player_freqdeck,
    player_num_resource_cards,
    get_dev_cards_in_hand,
    get_played_dev_cards,
    get_actual_victory_points,
    get_visible_victory_points,
    get_longest_road_length,
    get_longest_road_color,
    get_largest_army,
    player_has_rolled,
)


RESOURCE_NAMES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


class StateFormatter:
    """
    Utility class for formatting game state for LLM consumption.

    Provides methods to convert game state to:
    - Structured dictionaries (for tool outputs)
    - Natural language summaries (for prompts)
    - Action descriptions
    """

    @staticmethod
    def format_full_state(game: Game, my_color: Color) -> Dict[str, Any]:
        """
        Convert complete game state to a structured dictionary.

        This is the main state representation used by tools.
        """
        state = game.state
        my_key = player_key(state, my_color)

        return {
            "turn": state.num_turns,
            "phase": state.current_prompt.value,
            "my_color": my_color.value,
            "is_initial_build_phase": state.is_initial_build_phase,
            "my_state": StateFormatter._format_player_state(state, my_color, my_key, is_self=True),
            "opponents": StateFormatter._format_opponents(state, my_color),
            "board": StateFormatter._format_board_state(state, my_color),
            "bank": StateFormatter._format_bank(state),
            "special_achievements": StateFormatter._format_achievements(state),
            "trade_state": StateFormatter._format_trade_state(state),
        }

    @staticmethod
    def _format_player_state(
        state, color: Color, key: str, is_self: bool = False
    ) -> Dict[str, Any]:
        """Format a single player's state."""
        result = {
            "color": color.value,
            "visible_victory_points": get_visible_victory_points(state, color),
            "num_resource_cards": player_num_resource_cards(state, color),
            "num_dev_cards": get_dev_cards_in_hand(state, color),
            "played_knights": get_played_dev_cards(state, color, "KNIGHT"),
            "longest_road_length": get_longest_road_length(state, color),
            "has_longest_road": state.player_state.get(f"{key}_HAS_ROAD", False),
            "has_largest_army": state.player_state.get(f"{key}_HAS_ARMY", False),
            "settlements_available": state.player_state.get(f"{key}_SETTLEMENTS_AVAILABLE", 0),
            "cities_available": state.player_state.get(f"{key}_CITIES_AVAILABLE", 0),
            "roads_available": state.player_state.get(f"{key}_ROADS_AVAILABLE", 0),
        }

        # Add private information only for self
        if is_self:
            result["actual_victory_points"] = get_actual_victory_points(state, color)
            result["resources"] = {
                "wood": state.player_state.get(f"{key}_WOOD_IN_HAND", 0),
                "brick": state.player_state.get(f"{key}_BRICK_IN_HAND", 0),
                "sheep": state.player_state.get(f"{key}_SHEEP_IN_HAND", 0),
                "wheat": state.player_state.get(f"{key}_WHEAT_IN_HAND", 0),
                "ore": state.player_state.get(f"{key}_ORE_IN_HAND", 0),
            }
            result["development_cards"] = {
                "knight": state.player_state.get(f"{key}_KNIGHT_IN_HAND", 0),
                "year_of_plenty": state.player_state.get(f"{key}_YEAR_OF_PLENTY_IN_HAND", 0),
                "monopoly": state.player_state.get(f"{key}_MONOPOLY_IN_HAND", 0),
                "road_building": state.player_state.get(f"{key}_ROAD_BUILDING_IN_HAND", 0),
                "victory_point": state.player_state.get(f"{key}_VICTORY_POINT_IN_HAND", 0),
            }
            result["has_rolled"] = player_has_rolled(state, color)

        # Buildings
        buildings = state.buildings_by_color.get(color, {})
        result["settlements"] = buildings.get(SETTLEMENT, [])
        result["cities"] = buildings.get(CITY, [])
        result["num_roads"] = len(buildings.get(ROAD, []))

        return result

    @staticmethod
    def _format_opponents(state, my_color: Color) -> List[Dict[str, Any]]:
        """Format information about opponents."""
        opponents = []
        for color in state.colors:
            if color == my_color:
                continue
            key = player_key(state, color)
            opponents.append(
                StateFormatter._format_player_state(state, color, key, is_self=False)
            )
        return opponents

    @staticmethod
    def _format_board_state(state, my_color: Color) -> Dict[str, Any]:
        """Format board-related information."""
        board = state.board

        return {
            "robber_coordinate": board.robber_coordinate,
            "buildable_settlement_nodes": list(board.buildable_node_ids(my_color)),
            "buildable_road_edges": [
                list(edge) for edge in board.buildable_edges(my_color)
            ],
            "my_ports": list(board.get_player_port_resources(my_color)),
        }

    @staticmethod
    def _format_bank(state) -> Dict[str, Any]:
        """Format bank resource availability."""
        freqdeck = state.resource_freqdeck
        return {
            "resources": {
                "wood": freqdeck[0],
                "brick": freqdeck[1],
                "sheep": freqdeck[2],
                "wheat": freqdeck[3],
                "ore": freqdeck[4],
            },
            "dev_cards_remaining": len(state.development_listdeck),
        }

    @staticmethod
    def _format_achievements(state) -> Dict[str, Any]:
        """Format special achievements (longest road, largest army)."""
        longest_road_color = get_longest_road_color(state)
        largest_army_color, largest_army_size = get_largest_army(state)

        return {
            "longest_road_holder": longest_road_color.value if longest_road_color else None,
            "largest_army_holder": largest_army_color.value if largest_army_color else None,
            "largest_army_size": largest_army_size,
        }

    @staticmethod
    def _format_trade_state(state) -> Dict[str, Any]:
        """Format current trade state if any."""
        if not state.is_resolving_trade:
            return {"is_trading": False}

        current_trade = state.current_trade
        return {
            "is_trading": True,
            "offering": {
                "wood": current_trade[0],
                "brick": current_trade[1],
                "sheep": current_trade[2],
                "wheat": current_trade[3],
                "ore": current_trade[4],
            },
            "asking": {
                "wood": current_trade[5],
                "brick": current_trade[6],
                "sheep": current_trade[7],
                "wheat": current_trade[8],
                "ore": current_trade[9],
            },
            "acceptees": list(state.acceptees) if state.acceptees else [],
        }

    @staticmethod
    def format_for_prompt(game: Game, my_color: Color) -> str:
        """
        Create a natural language summary suitable for LLM prompts.

        This is a concise overview rather than full state dump.
        """
        state = game.state
        my_key = player_key(state, my_color)

        lines = []
        lines.append(f"=== GAME STATE (Turn {state.num_turns}) ===")
        lines.append(f"You are: {my_color.value}")
        lines.append(f"Phase: {state.current_prompt.value}")
        lines.append("")

        # My state
        lines.append("YOUR STATUS:")
        vps = get_actual_victory_points(state, my_color)
        lines.append(f"  Victory Points: {vps}")

        # Resources
        wood = state.player_state.get(f"{my_key}_WOOD_IN_HAND", 0)
        brick = state.player_state.get(f"{my_key}_BRICK_IN_HAND", 0)
        sheep = state.player_state.get(f"{my_key}_SHEEP_IN_HAND", 0)
        wheat = state.player_state.get(f"{my_key}_WHEAT_IN_HAND", 0)
        ore = state.player_state.get(f"{my_key}_ORE_IN_HAND", 0)
        lines.append(f"  Resources: Wood({wood}), Brick({brick}), Sheep({sheep}), Wheat({wheat}), Ore({ore})")

        # Dev cards
        knights = state.player_state.get(f"{my_key}_KNIGHT_IN_HAND", 0)
        yop = state.player_state.get(f"{my_key}_YEAR_OF_PLENTY_IN_HAND", 0)
        monopoly = state.player_state.get(f"{my_key}_MONOPOLY_IN_HAND", 0)
        rb = state.player_state.get(f"{my_key}_ROAD_BUILDING_IN_HAND", 0)
        vp_cards = state.player_state.get(f"{my_key}_VICTORY_POINT_IN_HAND", 0)
        if any([knights, yop, monopoly, rb, vp_cards]):
            lines.append(f"  Dev Cards: Knight({knights}), YoP({yop}), Monopoly({monopoly}), RoadBuilding({rb}), VP({vp_cards})")

        # Buildings
        buildings = state.buildings_by_color.get(my_color, {})
        settlements = len(buildings.get(SETTLEMENT, []))
        cities = len(buildings.get(CITY, []))
        roads = len(buildings.get(ROAD, []))
        lines.append(f"  Buildings: {settlements} settlements, {cities} cities, {roads} roads")
        lines.append("")

        # Opponents
        lines.append("OPPONENTS:")
        for color in state.colors:
            if color == my_color:
                continue
            other_key = player_key(state, color)
            other_vps = get_visible_victory_points(state, color)
            other_cards = player_num_resource_cards(state, color)
            other_dev = get_dev_cards_in_hand(state, color)
            lines.append(f"  {color.value}: {other_vps} VPs, {other_cards} resource cards, {other_dev} dev cards")

        lines.append("")

        # Trade state if applicable
        if state.is_resolving_trade:
            offer = state.current_trade[:5]
            ask = state.current_trade[5:10]
            lines.append(f"ACTIVE TRADE: Offering {offer}, Asking {ask}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_action(action: Action, index: int) -> Dict[str, Any]:
        """Format a single action for display to LLM."""
        action_type = action.action_type.value
        value = action.value

        result = {
            "index": index,
            "type": action_type,
            "description": StateFormatter._describe_action(action),
        }

        if value is not None:
            result["value"] = StateFormatter._serialize_value(value)

        return result

    @staticmethod
    def _describe_action(action: Action) -> str:
        """Generate human-readable description of an action."""
        action_type = action.action_type
        value = action.value

        descriptions = {
            ActionType.ROLL: "Roll the dice",
            ActionType.END_TURN: "End your turn",
            ActionType.BUY_DEVELOPMENT_CARD: "Buy a development card (costs: 1 sheep, 1 wheat, 1 ore)",
            ActionType.PLAY_KNIGHT_CARD: "Play Knight card (move robber)",
            ActionType.PLAY_ROAD_BUILDING: "Play Road Building card (build 2 free roads)",
            ActionType.CANCEL_TRADE: "Cancel the trade offer",
        }

        if action_type in descriptions:
            return descriptions[action_type]

        if action_type == ActionType.BUILD_SETTLEMENT:
            return f"Build settlement at node {value}"

        if action_type == ActionType.BUILD_CITY:
            return f"Upgrade settlement to city at node {value}"

        if action_type == ActionType.BUILD_ROAD:
            return f"Build road between nodes {value[0]} and {value[1]}"

        if action_type == ActionType.MOVE_ROBBER:
            coord, victim = value
            if victim:
                return f"Move robber to {coord} and steal from {victim.value}"
            return f"Move robber to {coord} (no one to steal from)"

        if action_type == ActionType.DISCARD:
            if value:
                return f"Discard resources: {value}"
            return "Discard resources (random selection)"

        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return f"Play Year of Plenty: take {value[0]} and {value[1]}"

        if action_type == ActionType.PLAY_MONOPOLY:
            return f"Play Monopoly: take all {value} from opponents"

        if action_type == ActionType.MARITIME_TRADE:
            give = [r for r in value[:4] if r is not None]
            get = value[4]
            return f"Maritime trade: give {give} for {get}"

        if action_type == ActionType.OFFER_TRADE:
            offer = value[:5]
            ask = value[5:]
            return f"Offer trade: give {StateFormatter._freqdeck_to_str(offer)} for {StateFormatter._freqdeck_to_str(ask)}"

        if action_type == ActionType.ACCEPT_TRADE:
            return "Accept the trade offer"

        if action_type == ActionType.REJECT_TRADE:
            return "Reject the trade offer"

        if action_type == ActionType.CONFIRM_TRADE:
            partner = value[10] if len(value) > 10 else "unknown"
            return f"Confirm trade with {partner}"

        return f"{action_type.value}"

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize action value for JSON output."""
        if value is None:
            return None
        if isinstance(value, (int, str, bool, float)):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        if hasattr(value, "value"):  # Enum
            return value.value
        return str(value)

    @staticmethod
    def _freqdeck_to_str(freqdeck: List[int]) -> str:
        """Convert a freqdeck to readable string."""
        parts = []
        for i, count in enumerate(freqdeck):
            if count > 0:
                parts.append(f"{count} {RESOURCE_NAMES[i].lower()}")
        return ", ".join(parts) if parts else "nothing"

    @staticmethod
    def format_strategy_insight(
        recommendation: Optional[Action],
        reasoning: Optional[str],
        strategy_name: str = "Strategy Advisor",
    ) -> str:
        """Format parent player's recommendation for the LLM."""
        if recommendation is None:
            return ""

        desc = StateFormatter._describe_action(recommendation)
        return f"""
STRATEGY ADVISOR RECOMMENDATION:
The {strategy_name} suggests: {desc}
Reasoning: {reasoning or "No detailed reasoning available"}

You may follow this recommendation or choose differently based on your analysis.
"""
