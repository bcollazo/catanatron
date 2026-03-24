from catanatron import Player
from catanatron.cli import register_cli_player

import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
capstone_agent_path = os.path.join(current_script_dir, "../../../capstone_agent")
normalized_path = os.path.normpath(capstone_agent_path)

if normalized_path not in sys.path:
    sys.path.append(normalized_path)
from AgentRouter import AgentRouter
from CapstoneAgent import CapstoneAgent # main gameplay agent
from PlacementAgent import PlacementAgent

gym_env_path = os.path.join(current_script_dir, "../gym/envs")
normalized_path = os.path.normpath(gym_env_path)
if normalized_path not in sys.path:
    sys.path.append(normalized_path)
from capstone_env import CapstoneCatanatronEnv

print(sys.path)

class RLCapstonePlayer(Player):

    def __init__(self, capstone_env: CapstoneCatanatronEnv, 
                 settlement_play_load_file: str = None, 
                 main_play_load_file: str = None):
        
        # take in the environment
        # TODO -> THIS NEEDS TO CHANGE
        self.env = capstone_env

        # load settlement agent
        self.settlement_play_agent = PlacementAgent()
        if settlement_play_load_file is not None:
            self.settlement_play_agent.load(settlement_play_load_file)
        
        # load main play agent
        self.main_play_agent = CapstoneAgent()
        if main_play_load_file is not None:
            self.main_play_agent.load(main_play_load_file)

        # create router (full play) agent
        self.router_agent = AgentRouter(self.settlement_play_agent,
                                        self.main_play_agent,
                                        self.env)
    
    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # use the environment 
        observation = self.env._get_obseravations()
        # valid_actions = self.env.get_valid_actions()
        action_mask = self.env.get_action_mask()

        action, _, _ = self.router_agent.select_action(observation, action_mask)

        return action

register_cli_player("Capstone_RL_AGENT", RLCapstonePlayer)