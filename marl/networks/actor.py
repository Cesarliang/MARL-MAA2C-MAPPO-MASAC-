"""
Actor Networks for Policy Representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Actor(nn.Module):
    """
    Actor network for discrete action spaces.
    Outputs probability distribution over actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            state: State tensor (shape: [batch_size, state_dim])
        
        Returns:
            Action logits (shape: [batch_size, action_dim])
        """
        features = self.network(state)
        logits = self.action_head(features)
        return logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution.
        
        Args:
            state: State tensor
        
        Returns:
            Action probabilities
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, select greedy action
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1))
        
        return action, log_prob


class ContinuousActor(nn.Module):
    """
    Actor network for continuous action spaces.
    Outputs Gaussian policy (mean and log_std).
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of continuous action space
            hidden_dims: Hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(ContinuousActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            state: State tensor (shape: [batch_size, state_dim])
        
        Returns:
            mean: Action mean
            log_std: Log standard deviation
        """
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from Gaussian policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            # Apply tanh squashing
            action = torch.tanh(action)
            # Correct log_prob for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob
