import torch
import torch.nn as nn

class CapstoneModel(torch.nn.Module):

    VALUE_SIZE = 1

    ROBBER_ACTION_SIZE = 19
    VERTEX_ACTION_SIZE = 54
    EDGE_ACTION_SIZE = 72
    PLAY_DEV_ACTION_SIZE = 5 # Buy Dev + 4 dev cards to play
    TRADING_ACTION_SIZE = 20
    NUM_RESOURCES = 5
    NUM_RESOURCE_PAIRS = 15
    TURN_MANAGEMENT_ACTION_SIZE = 3 # Roll, End Turn, Discard


    def __init__(self, obs_size, hidden_size):
        super(CapstoneModel, self).__init__()

        # ReLU activation
        # Softmax and Tanh final activations for policy and value functions

        self.linear_compressor = torch.nn.Linear(obs_size, hidden_size)

        self.residual1 = torch.nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU()
                    )

        self.residual2 = torch.nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU()
                    )
        
        self.residual3 = torch.nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU()
                    )
        
        self.residual4 = torch.nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU()
                    )
        
        # Value head
        self.value_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.VALUE_SIZE),
                        nn.Tanh()
                    )

        # Policy Heads
        self.full_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size)
                    )

        # Each action type gets a couple of layers to learn
        self.robber_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.ROBBER_ACTION_SIZE)
                    )
        self.settlement_vertex_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.VERTEX_ACTION_SIZE)
                    )
        self.city_vertex_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.VERTEX_ACTION_SIZE)
                    )
        
        self.edge_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.EDGE_ACTION_SIZE)
                    )
        
        self.dev_card_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.PLAY_DEV_ACTION_SIZE)
                    )
        self.trading_policy_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.TRADING_ACTION_SIZE)
                    )
        self.monopoly_resource_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.NUM_RESOURCES)
                    )
        self.yop_resource_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.NUM_RESOURCE_PAIRS)
                    )
        self.turn_management_head = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, self.TURN_MANAGEMENT_ACTION_SIZE)
                    )
        

    def forward(self, x, mask: torch.Tensor):

        x = self.linear_compressor(x)

        # Residual Block 1
        result1 = self.residual1(x)
        x = nn.ReLU()(x + result1)

        # Residual Block 2
        result2 = self.residual2(x)
        x = nn.ReLU()(x + result2)

        # Residual Block 3
        result3 = self.residual3(x)
        x = nn.ReLU()(x + result3)

        # Residual Block 4
        result4 = self.residual4(x)
        x = nn.ReLU()(x + result4)


        # Value Head
        state_value = self.value_head(x)

        # Policy head
        x = self.full_policy_head(x)

        robber = self.robber_policy_head(x)
        settlement= self.settlement_vertex_policy_head(x)
        city = self.city_vertex_policy_head(x)
        road = self.edge_policy_head(x)
        dev_card = self.dev_card_policy_head(x)
        trading = self.trading_policy_head(x)
        monopoly_resource = self.monopoly_resource_head(x)
        yop_resource = self.yop_resource_head(x)
        turn_management = self.turn_management_head(x)

        policy_logits = torch.cat([
            robber, settlement, city, road, dev_card, 
            trading, monopoly_resource, yop_resource, turn_management
        ], dim=-1)

        # ensure mask is on right device
        mask_tensor = torch.as_tensor(mask, device=policy_logits.device)

        masked_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        return probs, state_value