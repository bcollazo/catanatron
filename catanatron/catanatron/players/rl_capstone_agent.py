from catanatron import Player, Color, Game, Action
from collections.abc import Iterable
from catanatron.cli import register_cli_player
import numpy as np
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
capstone_agent_path = os.path.join(current_script_dir, "../../../capstone_agent")
normalized_path = os.path.normpath(capstone_agent_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

if root_dir not in sys.path:
    sys.path.append(root_dir)

if normalized_path not in sys.path:
    sys.path.append(normalized_path)
from capstone_agent.CapstoneAgent import CapstoneAgent # main gameplay agent
from capstone_agent.MainPlayAgent import MainPlayAgent
from capstone_agent.Placement.PlacementAgent import PlacementAgent

from capstone_agent.CONSTANTS import ACTION_SPACE_SIZE

gym_env_path = os.path.join(current_script_dir, "../gym/envs")
normalized_path = os.path.normpath(gym_env_path)
if normalized_path not in sys.path:
    sys.path.append(normalized_path)
from catanatron.gym.envs.capstone_features import get_capstone_observation
from catanatron.gym.envs.action_translator import capstone_to_action, batch_catanatron_to_capstone
from catanatron.gym.envs.catanatron_env import (
    to_action_space as to_catanatron_action_space,
)

class RLCapstonePlayer(Player):

    def __init__(self, 
                 color: Color,
                 settlement_play_load_file: str = None, 
                 main_play_load_file: str = None):

        self.color = color
        
        # load settlement agent
        self.settlement_play_agent = PlacementAgent()
        if settlement_play_load_file is not None:
            self.settlement_play_agent.load(settlement_play_load_file)
        
        # load main play agent
        self.main_play_agent = MainPlayAgent()
        if main_play_load_file is not None:
            self.main_play_agent.load(main_play_load_file)

        # create router (full play) agent
        self.full_agent = CapstoneAgent(self.settlement_play_agent,
                                        self.main_play_agent)
    
    def get_valid_actions(self, playable_actions: Iterable[Action]):
        """
        Returns:
            List[int]: valid actions in capstone action-space indices
        """
        catanatron_indices = list(
            map(to_catanatron_action_space, playable_actions)
        )
        return batch_catanatron_to_capstone(catanatron_indices)

    def get_action_mask(self, playable_actions: Iterable[Action]) -> np.ndarray:
        """Binary mask of shape (ACTION_SPACE_SIZE,). 1 = valid, 0 = invalid."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        for idx in self.get_valid_actions(playable_actions):
            mask[idx] = 1.0
        return mask
    
    def decide(self, game: Game, playable_actions: Iterable[Action]):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # figure out opponent's color
        color_to_index = game.state.color_to_index
        self_index = color_to_index[self.color]
        opp_index = 1 if self_index == 0 else 0
        opp_color = game.state.colors[opp_index]
        
        # get observation and action_mask
        observation = get_capstone_observation(game, self_color=self.color, opp_color = opp_color)
        action_mask = self.get_action_mask(playable_actions)

        # get a capstone indexed action
        capstone_action, _, _ = self.full_agent.select_action(observation, action_mask)

        # convert capstone action into the Catan Game indexed action
        catan_action = capstone_to_action(
                capstone_action, playable_actions
            )

        return catan_action

register_cli_player("Capstone_RL_AGENT", RLCapstonePlayer)