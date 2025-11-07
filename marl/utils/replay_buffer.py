"""
Replay Buffer for Experience Storage
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque


class ReplayBuffer:
    """
    Simple replay buffer for single-agent RL.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Batch of (states, actions, rewards, next_states, dones)
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent RL with support for global and local observations.
    """
    
    def __init__(self, capacity: int, n_agents: int):
        """
        Args:
            capacity: Maximum buffer size
            n_agents: Number of agents
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs: List[np.ndarray], actions: List[np.ndarray],
             rewards: List[float], next_obs: List[np.ndarray],
             dones: List[bool], global_state: Optional[np.ndarray] = None,
             next_global_state: Optional[np.ndarray] = None):
        """
        Add multi-agent experience to buffer.
        
        Args:
            obs: List of observations for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_obs: List of next observations for each agent
            dones: List of done flags for each agent
            global_state: Optional global state
            next_global_state: Optional next global state
        """
        experience = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'global_state': global_state,
            'next_global_state': next_global_state
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch of multi-agent experiences.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Dictionary of batched experiences
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Stack experiences for each agent
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []
        
        for agent_id in range(self.n_agents):
            obs_batch.append(torch.FloatTensor([exp['obs'][agent_id] for exp in batch]))
            actions_batch.append(torch.FloatTensor([exp['actions'][agent_id] for exp in batch]))
            rewards_batch.append(torch.FloatTensor([exp['rewards'][agent_id] for exp in batch]).unsqueeze(1))
            next_obs_batch.append(torch.FloatTensor([exp['next_obs'][agent_id] for exp in batch]))
            dones_batch.append(torch.FloatTensor([exp['dones'][agent_id] for exp in batch]).unsqueeze(1))
        
        result = {
            'obs': obs_batch,
            'actions': actions_batch,
            'rewards': rewards_batch,
            'next_obs': next_obs_batch,
            'dones': dones_batch
        }
        
        # Add global state if available
        if batch[0]['global_state'] is not None:
            result['global_state'] = torch.FloatTensor([exp['global_state'] for exp in batch])
            result['next_global_state'] = torch.FloatTensor([exp['next_global_state'] for exp in batch])
        
        return result
    
    def __len__(self) -> int:
        return len(self.buffer)


class EpisodeBuffer:
    """
    Buffer for storing complete episodes (useful for on-policy algorithms).
    """
    
    def __init__(self):
        """Initialize episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             value: float, log_prob: float, done: bool):
        """
        Add transition to episode buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Episode termination flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all transitions as tensors.
        
        Returns:
            Tuple of (states, actions, rewards, values, log_probs, dones)
        """
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.dones))
        )
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        return len(self.states)
