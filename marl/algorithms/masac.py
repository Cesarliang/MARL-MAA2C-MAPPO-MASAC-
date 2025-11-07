"""
Multi-Agent Soft Actor-Critic (MASAC) with Game Equilibrium Guidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

from marl.networks.actor import ContinuousActor
from marl.networks.critic import Critic
from marl.equilibrium.nash_solver import NashEquilibriumSolver


class MASAC:
    """
    Multi-Agent Soft Actor-Critic with Nash Equilibrium Guidance.
    
    SAC is an off-policy algorithm that optimizes a stochastic policy in continuous
    action spaces, maximizing both expected return and entropy.
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_entropy_tuning: bool = True,
                 use_equilibrium_guidance: bool = True,
                 equilibrium_weight: float = 0.3,
                 device: str = 'cpu'):
        """
        Args:
            n_agents: Number of agents
            state_dim: Dimension of state space
            action_dim: Dimension of continuous action space
            hidden_dims: Hidden layer dimensions
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            lr_alpha: Learning rate for temperature parameter
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter (if not auto-tuning)
            auto_entropy_tuning: Whether to automatically tune entropy
            use_equilibrium_guidance: Whether to use equilibrium guidance
            equilibrium_weight: Weight for equilibrium guidance
            device: Device to run on
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning
        self.use_equilibrium_guidance = use_equilibrium_guidance
        self.equilibrium_weight = equilibrium_weight
        self.device = torch.device(device)
        
        # Create actor and critics for each agent
        self.actors = [
            ContinuousActor(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        # Two Q-networks per agent for stability
        self.q1_networks = [
            Critic(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        self.q2_networks = [
            Critic(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        # Target networks
        self.q1_targets = [
            Critic(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        self.q2_targets = [
            Critic(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        # Initialize target networks
        for i in range(n_agents):
            self.q1_targets[i].load_state_dict(self.q1_networks[i].state_dict())
            self.q2_targets[i].load_state_dict(self.q2_networks[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        
        self.q1_optimizers = [
            optim.Adam(q1.parameters(), lr=lr_critic)
            for q1 in self.q1_networks
        ]
        
        self.q2_optimizers = [
            optim.Adam(q2.parameters(), lr=lr_critic)
            for q2 in self.q2_networks
        ]
        
        # Temperature parameter
        if self.auto_entropy_tuning:
            self.target_entropy = [-action_dim for _ in range(n_agents)]
            self.log_alphas = [
                torch.zeros(1, requires_grad=True, device=self.device)
                for _ in range(n_agents)
            ]
            self.alpha_optimizers = [
                optim.Adam([log_alpha], lr=lr_alpha)
                for log_alpha in self.log_alphas
            ]
            self.alphas = [log_alpha.exp() for log_alpha in self.log_alphas]
        else:
            self.alphas = [torch.tensor(alpha, device=self.device) for _ in range(n_agents)]
        
        # Nash equilibrium solver
        if self.use_equilibrium_guidance:
            self.nash_solver = NashEquilibriumSolver()
        
        self.train_step = 0
    
    def select_action(self, states: List[np.ndarray],
                     deterministic: bool = False) -> List[np.ndarray]:
        """
        Select actions for all agents.
        
        Args:
            states: List of states for each agent
            deterministic: If True, select mean actions
        
        Returns:
            actions: List of actions for each agent
        """
        actions = []
        
        for agent_id in range(self.n_agents):
            state = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _ = self.actors[agent_id].sample_action(state, deterministic)
            
            actions.append(action.cpu().numpy()[0])
        
        return actions
    
    def update(self, batch: dict) -> dict:
        """
        Update all agents using SAC.
        
        Args:
            batch: Dictionary containing batch of experiences
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'actor_loss': 0.0,
            'q1_loss': 0.0,
            'q2_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': 0.0
        }
        
        for agent_id in range(self.n_agents):
            # Extract data for this agent
            states = torch.FloatTensor(batch['states'][agent_id]).to(self.device)
            actions = torch.FloatTensor(batch['actions'][agent_id]).to(self.device)
            rewards = torch.FloatTensor(batch['rewards'][agent_id]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch['next_states'][agent_id]).to(self.device)
            dones = torch.FloatTensor(batch['dones'][agent_id]).unsqueeze(1).to(self.device)
            
            # Update Q-functions
            with torch.no_grad():
                next_actions, next_log_probs = self.actors[agent_id].sample_action(next_states)
                
                q1_next = self.q1_targets[agent_id](next_states, next_actions)
                q2_next = self.q2_targets[agent_id](next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
                
                alpha = self.alphas[agent_id].detach()
                target_q = rewards + self.gamma * (1 - dones) * (q_next - alpha * next_log_probs)
            
            # Q1 loss
            q1_pred = self.q1_networks[agent_id](states, actions)
            q1_loss = nn.MSELoss()(q1_pred, target_q)
            
            self.q1_optimizers[agent_id].zero_grad()
            q1_loss.backward()
            self.q1_optimizers[agent_id].step()
            
            # Q2 loss
            q2_pred = self.q2_networks[agent_id](states, actions)
            q2_loss = nn.MSELoss()(q2_pred, target_q)
            
            self.q2_optimizers[agent_id].zero_grad()
            q2_loss.backward()
            self.q2_optimizers[agent_id].step()
            
            # Update actor
            new_actions, log_probs = self.actors[agent_id].sample_action(states)
            
            q1_new = self.q1_networks[agent_id](states, new_actions)
            q2_new = self.q2_networks[agent_id](states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            alpha = self.alphas[agent_id].detach()
            actor_loss = (alpha * log_probs - q_new).mean()
            
            # Apply equilibrium guidance
            if self.use_equilibrium_guidance and 'equilibrium_actions' in batch:
                eq_actions = torch.FloatTensor(batch['equilibrium_actions'][agent_id]).to(self.device)
                eq_loss = nn.MSELoss()(new_actions, eq_actions)
                actor_loss += self.equilibrium_weight * eq_loss
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()
            
            # Update alpha (temperature)
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alphas[agent_id] * 
                              (log_probs + self.target_entropy[agent_id]).detach()).mean()
                
                self.alpha_optimizers[agent_id].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizers[agent_id].step()
                
                self.alphas[agent_id] = self.log_alphas[agent_id].exp()
                metrics['alpha_loss'] += alpha_loss.item()
            
            # Soft update target networks
            self._soft_update(self.q1_networks[agent_id], self.q1_targets[agent_id])
            self._soft_update(self.q2_networks[agent_id], self.q2_targets[agent_id])
            
            # Update metrics
            metrics['actor_loss'] += actor_loss.item()
            metrics['q1_loss'] += q1_loss.item()
            metrics['q2_loss'] += q2_loss.item()
            metrics['alpha'] += self.alphas[agent_id].item()
        
        # Average metrics across agents
        for key in metrics:
            metrics[key] /= self.n_agents
        
        self.train_step += 1
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update target network parameters.
        
        Args:
            source: Source network
            target: Target network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'q1_networks': [q1.state_dict() for q1 in self.q1_networks],
            'q2_networks': [q2.state_dict() for q2 in self.q2_networks],
            'q1_targets': [q1.state_dict() for q1 in self.q1_targets],
            'q2_targets': [q2.state_dict() for q2 in self.q2_targets],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'q1_optimizers': [opt.state_dict() for opt in self.q1_optimizers],
            'q2_optimizers': [opt.state_dict() for opt in self.q2_optimizers],
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        
        for i in range(self.n_agents):
            self.q1_networks[i].load_state_dict(checkpoint['q1_networks'][i])
            self.q2_networks[i].load_state_dict(checkpoint['q2_networks'][i])
            self.q1_targets[i].load_state_dict(checkpoint['q1_targets'][i])
            self.q2_targets[i].load_state_dict(checkpoint['q2_targets'][i])
        
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        
        for i in range(self.n_agents):
            self.q1_optimizers[i].load_state_dict(checkpoint['q1_optimizers'][i])
            self.q2_optimizers[i].load_state_dict(checkpoint['q2_optimizers'][i])
