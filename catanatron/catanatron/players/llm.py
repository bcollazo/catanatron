"""LLM-based player that uses Ollama for decision making."""
import re
import os
from pathlib import Path

from catanatron.models.player import Player
from catanatron.models.enums import Resource


class LLMPlayer(Player):
    """
    Player that uses a local Ollama instance to make decisions via LLM.

    This player sends the game state and available actions to an LLM
    and parses the response to select an action.
    """

    def __init__(self, color, model_name="llama3.2", ollama_base_url="http://localhost:11434", is_bot=True):
        """Initialize the LLM player.

        Args:
            color (Color): The player's color
            model_name (str): Name of the Ollama model to use (default: llama3.2)
            ollama_base_url (str): Base URL for Ollama API (default: http://localhost:11434)
            is_bot (bool): Whether this is a bot player
        """
        super().__init__(color, is_bot)
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self._prompt_template = None
        self._llm = None

    def _get_prompt_template(self):
        """Load the prompt template from markdown file."""
        if self._prompt_template is None:
            try:
                from langchain_core.prompts import PromptTemplate

                template_path = Path(__file__).parent / "llm_player_prompt.md"
                with open(template_path, "r") as f:
                    template_content = f.read()

                self._prompt_template = PromptTemplate.from_template(template_content)
            except ImportError:
                raise ImportError(
                    "langchain is required for LLMPlayer. "
                    "Install it with: pip install langchain langchain-ollama"
                )
        return self._prompt_template

    def _get_llm(self):
        """Get or create the Ollama LLM instance."""
        if self._llm is None:
            try:
                from langchain_ollama import OllamaLLM

                self._llm = OllamaLLM(
                    model=self.model_name,
                    base_url=self.ollama_base_url,
                    temperature=0.7,
                )
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for LLMPlayer. "
                    "Install it with: pip install langchain-ollama"
                )
        return self._llm

    def _describe_node_location(self, board, node_id):
        """Describe a node location by its adjacent tiles.

        Args:
            board: The game board
            node_id: The node ID to describe

        Returns:
            str: Description like "3 WHEAT - 5 ORE - 6 BRICK"
        """
        adjacent_tiles = board.map.adjacent_tiles.get(node_id, [])

        # Build descriptions for each adjacent tile
        tile_descriptions = []
        for tile in adjacent_tiles:
            if tile.resource is None:  # Desert tile
                tile_descriptions.append(f"{tile.number or 0} DESERT")
            else:
                tile_descriptions.append(f"{tile.number or 0} {tile.resource}")

        # Sort for consistency
        tile_descriptions.sort()

        return " - ".join(tile_descriptions) if tile_descriptions else "Unknown"

    def _extract_state_info(self, game):
        """Extract simplified state information from the game.

        Args:
            game: The game instance

        Returns:
            dict: Simplified state information
        """
        state = game.state
        color = self.color

        # Get player state
        player_key_prefix = f"P{state.colors.index(color)}_"

        def get_player_stat(stat_name):
            return state.player_state.get(f"{player_key_prefix}{stat_name}", 0)

        # Build resource hand as a list
        resource_hand = []
        for resource, stat_name in [
            ("WOOD", "WOOD_IN_HAND"),
            ("BRICK", "BRICK_IN_HAND"),
            ("SHEEP", "SHEEP_IN_HAND"),
            ("WHEAT", "WHEAT_IN_HAND"),
            ("ORE", "ORE_IN_HAND"),
        ]:
            count = get_player_stat(stat_name)
            resource_hand.extend([resource] * count)

        # Build unused dev cards list
        unused_dev_cards = []
        for card, stat_name in [
            ("KNIGHT", "KNIGHT_IN_HAND"),
            ("YEAR_OF_PLENTY", "YEAR_OF_PLENTY_IN_HAND"),
            ("MONOPOLY", "MONOPOLY_IN_HAND"),
            ("ROAD_BUILDING", "ROAD_BUILDING_IN_HAND"),
            ("VICTORY_POINT", "VICTORY_POINT_IN_HAND"),
        ]:
            count = get_player_stat(stat_name)
            unused_dev_cards.extend([card] * count)

        # Build used dev cards list (played cards)
        used_dev_cards = []
        for card, stat_name in [
            ("KNIGHT", "PLAYED_KNIGHT"),
            ("YEAR_OF_PLENTY", "PLAYED_YEAR_OF_PLENTY"),
            ("MONOPOLY", "PLAYED_MONOPOLY"),
            ("ROAD_BUILDING", "PLAYED_ROAD_BUILDING"),
        ]:
            count = get_player_stat(stat_name)
            used_dev_cards.extend([card] * count)

        # Get buildings with locations
        buildings_info = []
        for node_id, (building_color, building_type) in state.board.buildings.items():
            if building_color == color:
                location_desc = self._describe_node_location(state.board, node_id)
                building_name = "Settlement" if building_type == 1 else "City"
                buildings_info.append(f"- {building_name} at {location_desc}")

        # Extract own stats
        info = {
            "color": color.value,
            "your_vp": get_player_stat("VICTORY_POINTS"),
            "your_hand": str(resource_hand),
            "your_unused_dev_cards": str(unused_dev_cards),
            "your_used_dev_cards": str(used_dev_cards),
            "your_buildings": "\n".join(buildings_info) if buildings_info else "None",
            "your_roads_available": get_player_stat("ROADS_AVAILABLE"),
            "your_settlements_available": get_player_stat("SETTLEMENTS_AVAILABLE"),
            "your_cities_available": get_player_stat("CITIES_AVAILABLE"),
            "has_longest_road": "Yes" if get_player_stat("HAS_ROAD") else "No",
            "has_largest_army": "Yes" if get_player_stat("HAS_ARMY") else "No",
            "current_turn": state.num_turns,
            "robber_active": "Yes" if hasattr(state.board, 'robber_coordinate') else "No",
        }

        # Extract opponent stats (only public information)
        opponents_info = []
        for i, opponent_color in enumerate(state.colors):
            if opponent_color == color:
                continue

            opp_prefix = f"P{i}_"
            opp_vp = state.player_state.get(f"{opp_prefix}VICTORY_POINTS", 0)

            # Count total resource cards (public info - you can see hand size)
            opp_resource_cards = sum([
                state.player_state.get(f"{opp_prefix}WOOD_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}BRICK_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}SHEEP_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}WHEAT_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}ORE_IN_HAND", 0),
            ])

            # Count total dev cards (public info)
            opp_dev_cards = sum([
                state.player_state.get(f"{opp_prefix}KNIGHT_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}YEAR_OF_PLENTY_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}MONOPOLY_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}ROAD_BUILDING_IN_HAND", 0),
                state.player_state.get(f"{opp_prefix}VICTORY_POINT_IN_HAND", 0),
            ])

            opp_has_longest = "Yes" if state.player_state.get(f"{opp_prefix}HAS_ROAD", False) else "No"
            opp_has_largest = "Yes" if state.player_state.get(f"{opp_prefix}HAS_ARMY", False) else "No"

            opponents_info.append(
                f"- {opponent_color.value}: {opp_vp} VP, {opp_resource_cards} resource cards, "
                f"{opp_dev_cards} dev cards, Longest Road: {opp_has_longest}, Largest Army: {opp_has_largest}"
            )

        info["opponents_stats"] = "\n".join(opponents_info)

        return info

    def _format_actions(self, playable_actions):
        """Format playable actions as a numbered list.

        Args:
            playable_actions: List of Action objects

        Returns:
            str: Formatted actions list
        """
        actions_list = []
        for i, action in enumerate(playable_actions):
            # Simplify action display
            action_desc = f"{action.action_type.value}"
            if action.value is not None and action.value != ():
                action_desc += f" (value: {action.value})"
            actions_list.append(f"{i}. {action_desc}")

        return "\n".join(actions_list)

    def _parse_llm_response(self, response_text, num_actions):
        """Parse the LLM response to extract action index.

        Args:
            response_text (str): Raw LLM response
            num_actions (int): Number of available actions

        Returns:
            int: Parsed action index, or 0 if parsing fails
        """
        # Try to extract a number from the response
        # Look for standalone numbers or numbers at the start/end of lines
        numbers = re.findall(r'\b(\d+)\b', response_text.strip())

        if numbers:
            # Take the first number found
            action_idx = int(numbers[0])

            # Validate it's within range
            if 0 <= action_idx < num_actions:
                return action_idx

        # Fallback: return first action if parsing fails
        print(f"Warning: Could not parse LLM response '{response_text}'. Defaulting to action 0.")
        return 0

    def decide(self, game, playable_actions):
        """Use LLM to decide which action to take.

        Args:
            game: The game instance
            playable_actions: List of available actions

        Returns:
            Action: The chosen action
        """
        try:
            # Extract state information
            state_info = self._extract_state_info(game)

            # Format actions
            state_info["actions_list"] = self._format_actions(playable_actions)
            state_info["max_action_index"] = len(playable_actions) - 1

            # Get prompt template and LLM
            prompt_template = self._get_prompt_template()
            llm = self._get_llm()

            # Format prompt
            formatted_prompt = prompt_template.format(**state_info)

            # Call LLM
            response = llm.invoke(formatted_prompt)

            # Parse response
            action_idx = self._parse_llm_response(response, len(playable_actions))

            return playable_actions[action_idx]

        except Exception as e:
            print(f"Error in LLMPlayer.decide: {e}")
            print(f"Falling back to first action")
            return playable_actions[0]
