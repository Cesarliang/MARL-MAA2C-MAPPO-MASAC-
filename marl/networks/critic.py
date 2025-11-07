"""
Critic Networks for Value Function Approximation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Critic(nn.Module):
    """
    Critic network for estimating state value or Q-value.
    """
    
    def __init__(self, state_dim: int, action_dim: Optional[int] = None,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (None for state-value, int for Q-value)
            hidden_dims: Hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        self.action_dim = action_dim
        input_dim = state_dim + (action_dim if action_dim is not None else 0)
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: State tensor (shape: [batch_size, state_dim])
            action: Action tensor (shape: [batch_size, action_dim]), required if action_dim is not None
        
        Returns:
            Value estimate (shape: [batch_size, 1])
        """
        if self.action_dim is not None:
            if action is None:
                raise ValueError("Action must be provided for Q-value network")
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        
        features = self.network(x)
        value = self.value_head(features)
        return value


class CentralizedCritic(nn.Module):
    """
    Centralized critic that takes global state information.
    Used in CTDE (Centralized Training with Decentralized Execution) paradigm.
    """
    
    def __init__(self, global_state_dim: int, n_agents: int,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        Args:
            global_state_dim: Dimension of global state
            n_agents: Number of agents
            hidden_dims: Hidden layer dimensions
        """
        super(CentralizedCritic, self).__init__()
        
        self.n_agents = n_agents
        
        layers = []
        input_dim = global_state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        # Output value for each agent
        self.value_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_agents)
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, global_state: torch.Tensor, agent_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through centralized critic.
        
        Args:
            global_state: Global state tensor (shape: [batch_size, global_state_dim])
            agent_id: Specific agent ID to get value for (None for all agents)
        
        Returns:
            Value estimate for specified agent or all agents
        """
        features = self.network(global_state)
        
        if agent_id is not None:
            return self.value_heads[agent_id](features)
        else:
            # Return values for all agents
            values = torch.cat([head(features) for head in self.value_heads], dim=-1)
            return values


from typing import Optional

