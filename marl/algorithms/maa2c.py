"""
Multi-Agent Advantage Actor-Critic (MAA2C) with Game Equilibrium Guidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple

from marl.networks.actor import Actor
from marl.networks.critic import Critic, CentralizedCritic
from marl.equilibrium.nash_solver import NashEquilibriumSolver


class MAA2C:
    """
    Multi-Agent Advantage Actor-Critic with Nash Equilibrium Guidance.
    
    This algorithm combines the A2C approach with game-theoretic equilibrium
    concepts to guide multi-agent learning towards stable solutions.
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_equilibrium_guidance: bool = True,
                 equilibrium_weight: float = 0.3,
                 device: str = 'cpu'):
        """
        Args:
            n_agents: Number of agents
            state_dim: Dimension of state space for each agent
            action_dim: Number of discrete actions for each agent
            hidden_dims: Hidden layer dimensions
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_equilibrium_guidance: Whether to use Nash equilibrium guidance
            equilibrium_weight: Weight for equilibrium guidance (0-1)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_equilibrium_guidance = use_equilibrium_guidance
        self.equilibrium_weight = equilibrium_weight
        self.device = torch.device(device)
        
        # Create actor and critic networks for each agent
        self.actors = [
            Actor(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        self.critics = [
            Critic(state_dim, hidden_dims=hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic)
            for critic in self.critics
        ]
        
        # Nash equilibrium solver
        if self.use_equilibrium_guidance:
            self.nash_solver = NashEquilibriumSolver()
        
        # Statistics
        self.train_step = 0
    
    def select_action(self, states: List[np.ndarray], 
                     deterministic: bool = False) -> Tuple[List[int], List[float]]:
        """
        Select actions for all agents.
        
        Args:
            states: List of states for each agent
            deterministic: If True, select greedy actions
        
        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
        """
        actions = []
        log_probs = []
        
        for agent_id in range(self.n_agents):
            state = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.actors[agent_id].sample_action(state, deterministic)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
        
        return actions, log_probs
    
    def compute_returns(self, rewards: List[float], dones: List[bool],
                       next_state: np.ndarray, agent_id: int) -> float:
        """
        Compute returns for an agent.
        
        Args:
            rewards: List of rewards
            dones: List of done flags
            next_state: Next state
            agent_id: Agent index
        
        Returns:
            Discounted return
        """
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            next_value = self.critics[agent_id](next_state_tensor).item()
        
        returns = []
        R = next_value if not dones[-1] else 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        return returns
    
    def update(self, batch: dict) -> dict:
        """
        Update all agents using collected experiences.
        
        Args:
            batch: Dictionary containing batch of experiences
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0
        }
        
        for agent_id in range(self.n_agents):
            # Extract data for this agent
            states = batch['states'][agent_id]
            actions = batch['actions'][agent_id]
            returns = batch['returns'][agent_id]
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Compute values and advantages
            values = self.critics[agent_id](states).squeeze()
            advantages = returns - values.detach()
            
            # Compute actor loss
            logits = self.actors[agent_id](states)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze()
            
            # Policy gradient loss
            policy_loss = -(action_log_probs * advantages).mean()
            
            # Entropy regularization
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Total actor loss
            actor_loss = policy_loss - self.entropy_coef * entropy
            
            # Apply equilibrium guidance if enabled
            if self.use_equilibrium_guidance and 'equilibrium_strategy' in batch:
                eq_strategy = batch['equilibrium_strategy'][agent_id]
                eq_strategy_tensor = torch.FloatTensor(eq_strategy).to(self.device)
                
                # KL divergence between current policy and equilibrium strategy
                kl_loss = (probs * (torch.log(probs + 1e-8) - 
                                   torch.log(eq_strategy_tensor + 1e-8))).sum(dim=-1).mean()
                
                actor_loss += self.equilibrium_weight * kl_loss
            
            # Critic loss
            critic_loss = nn.MSELoss()(values, returns)
            
            # Update actor
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()
            
            # Update critic
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critic_optimizers[agent_id].step()
            
            # Update metrics
            metrics['actor_loss'] += actor_loss.item()
            metrics['critic_loss'] += critic_loss.item()
            metrics['entropy'] += entropy.item()
        
        # Average metrics across agents
        for key in metrics:
            metrics[key] /= self.n_agents
        
        self.train_step += 1
        return metrics
    
    def compute_equilibrium_guidance(self, q_values_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute Nash equilibrium strategies from Q-values.
        
        Args:
            q_values_list: List of Q-values for each agent
        
        Returns:
            List of equilibrium strategies
        """
        if not self.use_equilibrium_guidance:
            return [None] * self.n_agents
        
        # For two-player games, compute Nash equilibrium
        if self.n_agents == 2:
            # Construct payoff matrix from Q-values
            payoff_matrix = q_values_list[0]
            strategy, _ = self.nash_solver.solve_two_player_zero_sum(payoff_matrix)
            
            # Return strategies for both players
            return [strategy, None]  # Second player's strategy is implicit
        else:
            # For n-player games, use fictitious play
            payoff_matrices = q_values_list
            strategies = self.nash_solver.fictitious_play(payoff_matrices)
            return strategies
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        
        for i, opt in enumerate(self.critic_optimizers):
            opt.load_state_dict(checkpoint['critic_optimizers'][i])
