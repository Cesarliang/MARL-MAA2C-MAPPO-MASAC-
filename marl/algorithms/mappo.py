"""
Multi-Agent Proximal Policy Optimization (MAPPO) with Game Equilibrium Guidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

from marl.networks.actor import Actor
from marl.networks.critic import CentralizedCritic
from marl.equilibrium.correlated_solver import CorrelatedEquilibriumSolver


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization with Correlated Equilibrium Guidance.
    
    Uses centralized training with decentralized execution (CTDE) paradigm.
    Incorporates correlated equilibrium concepts for better coordination.
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 action_dim: int,
                 global_state_dim: Optional[int] = None,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 use_equilibrium_guidance: bool = True,
                 equilibrium_weight: float = 0.3,
                 device: str = 'cpu'):
        """
        Args:
            n_agents: Number of agents
            state_dim: Dimension of local state space
            action_dim: Number of discrete actions
            global_state_dim: Dimension of global state (if None, uses n_agents * state_dim)
            hidden_dims: Hidden layer dimensions
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: Number of PPO update epochs
            mini_batch_size: Mini-batch size for PPO updates
            use_equilibrium_guidance: Whether to use equilibrium guidance
            equilibrium_weight: Weight for equilibrium guidance
            device: Device to run on
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim or (n_agents * state_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.use_equilibrium_guidance = use_equilibrium_guidance
        self.equilibrium_weight = equilibrium_weight
        self.device = torch.device(device)
        
        # Create decentralized actors for each agent
        self.actors = [
            Actor(state_dim, action_dim, hidden_dims).to(self.device)
            for _ in range(n_agents)
        ]
        
        # Centralized critic
        self.critic = CentralizedCritic(
            self.global_state_dim, n_agents, hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Correlated equilibrium solver
        if self.use_equilibrium_guidance:
            self.eq_solver = CorrelatedEquilibriumSolver()
        
        self.train_step = 0
    
    def select_action(self, states: List[np.ndarray],
                     deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
        """
        Select actions for all agents.
        
        Args:
            states: List of local states for each agent
            deterministic: If True, select greedy actions
        
        Returns:
            actions: List of actions
            log_probs: List of log probabilities
            values: List of value estimates
        """
        actions = []
        log_probs = []
        values = []
        
        # Construct global state
        global_state = torch.FloatTensor(np.concatenate(states)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get values from centralized critic
            value_estimates = self.critic(global_state)
        
        for agent_id in range(self.n_agents):
            state = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.actors[agent_id].sample_action(state, deterministic)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value_estimates[0, agent_id].item())
        
        return actions, log_probs, values
    
    def compute_gae(self, rewards: List[List[float]], values: List[List[float]],
                   dones: List[List[bool]], next_values: List[float]) -> List[List[float]]:
        """
        Compute Generalized Advantage Estimation for all agents.
        
        Args:
            rewards: Rewards for each agent over time
            values: Value estimates for each agent over time
            dones: Done flags for each agent over time
            next_values: Next state values for each agent
        
        Returns:
            advantages: GAE advantages for each agent
        """
        advantages = []
        
        for agent_id in range(self.n_agents):
            agent_rewards = rewards[agent_id]
            agent_values = values[agent_id]
            agent_dones = dones[agent_id]
            
            gae = 0
            agent_advantages = []
            
            for t in reversed(range(len(agent_rewards))):
                if t == len(agent_rewards) - 1:
                    next_value = next_values[agent_id] if not agent_dones[t] else 0
                else:
                    next_value = agent_values[t + 1]
                
                delta = agent_rewards[t] + self.gamma * next_value * (1 - agent_dones[t]) - agent_values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - agent_dones[t]) * gae
                agent_advantages.insert(0, gae)
            
            advantages.append(agent_advantages)
        
        return advantages
    
    def update(self, batch: dict) -> dict:
        """
        Update all agents using PPO.
        
        Args:
            batch: Dictionary containing batch of experiences
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0
        }
        
        # Extract data
        states = [torch.FloatTensor(batch['states'][i]).to(self.device) 
                 for i in range(self.n_agents)]
        actions = [torch.LongTensor(batch['actions'][i]).to(self.device)
                  for i in range(self.n_agents)]
        old_log_probs = [torch.FloatTensor(batch['log_probs'][i]).to(self.device)
                        for i in range(self.n_agents)]
        advantages = [torch.FloatTensor(batch['advantages'][i]).to(self.device)
                     for i in range(self.n_agents)]
        returns = [torch.FloatTensor(batch['returns'][i]).to(self.device)
                  for i in range(self.n_agents)]
        global_states = torch.FloatTensor(batch['global_states']).to(self.device)
        
        # Normalize advantages
        for i in range(self.n_agents):
            advantages[i] = (advantages[i] - advantages[i].mean()) / (advantages[i].std() + 1e-8)
        
        # PPO update
        batch_size = states[0].shape[0]
        
        for _ in range(self.ppo_epochs):
            # Shuffle indices for mini-batch updates
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                # Update each agent's actor
                for agent_id in range(self.n_agents):
                    mb_states = states[agent_id][mb_indices]
                    mb_actions = actions[agent_id][mb_indices]
                    mb_old_log_probs = old_log_probs[agent_id][mb_indices]
                    mb_advantages = advantages[agent_id][mb_indices]
                    
                    # Compute new log probs
                    logits = self.actors[agent_id](mb_states)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(-1, mb_actions.unsqueeze(-1)).squeeze()
                    
                    # Compute ratio
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    # Clipped surrogate loss
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy bonus
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    
                    # Total actor loss
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    
                    # Apply equilibrium guidance
                    if self.use_equilibrium_guidance and 'equilibrium_dist' in batch:
                        eq_dist = batch['equilibrium_dist']
                        eq_strategy = torch.FloatTensor(eq_dist.sum(axis=1-agent_id)).to(self.device)
                        eq_strategy = eq_strategy / eq_strategy.sum()
                        
                        kl_loss = (probs * (torch.log(probs + 1e-8) - 
                                           torch.log(eq_strategy + 1e-8))).sum(dim=-1).mean()
                        actor_loss += self.equilibrium_weight * kl_loss
                    
                    # Update actor
                    self.actor_optimizers[agent_id].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                    self.actor_optimizers[agent_id].step()
                    
                    # Track metrics
                    metrics['actor_loss'] += actor_loss.item()
                    metrics['entropy'] += entropy.item()
                    metrics['clip_fraction'] += ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                
                # Update centralized critic
                mb_global_states = global_states[mb_indices]
                mb_returns = torch.stack([returns[i][mb_indices] for i in range(self.n_agents)], dim=1)
                
                values = self.critic(mb_global_states)
                critic_loss = nn.MSELoss()(values, mb_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                metrics['critic_loss'] += critic_loss.item()
        
        # Average metrics
        num_updates = self.ppo_epochs * (batch_size // self.mini_batch_size)
        for key in metrics:
            if key != 'clip_fraction':
                metrics[key] /= (num_updates * self.n_agents)
            else:
                metrics[key] /= num_updates
        
        self.train_step += 1
        return metrics
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        
        self.critic.load_state_dict(checkpoint['critic'])
        
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
