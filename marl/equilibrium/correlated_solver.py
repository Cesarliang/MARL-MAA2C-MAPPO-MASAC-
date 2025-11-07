"""
Correlated Equilibrium Solver for Multi-Agent Games
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linprog


class CorrelatedEquilibriumSolver:
    """
    Solver for computing Correlated Equilibrium in multi-agent games.
    Correlated equilibrium is a generalization of Nash equilibrium that can achieve
    better social welfare by allowing coordination through public signals.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Convergence threshold
        """
        self.epsilon = epsilon
    
    def solve_two_player(self, payoff_matrices: List[np.ndarray], 
                        objective: str = 'welfare') -> np.ndarray:
        """
        Solve Correlated Equilibrium for two-player game using Linear Programming.
        
        Args:
            payoff_matrices: List of payoff matrices for each player
            objective: Optimization objective ('welfare' or 'fairness')
        
        Returns:
            joint_distribution: Joint probability distribution over action profiles
        """
        payoff_1, payoff_2 = payoff_matrices[0], payoff_matrices[1]
        n_actions_1, n_actions_2 = payoff_1.shape
        n_vars = n_actions_1 * n_actions_2
        
        # Objective: maximize social welfare (sum of payoffs)
        if objective == 'welfare':
            c = -(payoff_1 + payoff_2).flatten()
        else:
            c = -payoff_1.flatten()  # Maximize player 1's payoff
        
        # Inequality constraints for incentive compatibility
        constraints_ub = []
        
        # Player 1's incentive constraints
        for i in range(n_actions_1):
            for j in range(n_actions_1):
                if i != j:
                    # Sum over player 2's actions: p(i,k) * (u1(i,k) - u1(j,k)) >= 0
                    constraint = np.zeros(n_vars)
                    for k in range(n_actions_2):
                        idx = i * n_actions_2 + k
                        constraint[idx] = payoff_1[i, k] - payoff_1[j, k]
                    constraints_ub.append(-constraint)  # Convert to <= form
        
        # Player 2's incentive constraints
        for k in range(n_actions_2):
            for l in range(n_actions_2):
                if k != l:
                    # Sum over player 1's actions: p(i,k) * (u2(i,k) - u2(i,l)) >= 0
                    constraint = np.zeros(n_vars)
                    for i in range(n_actions_1):
                        idx = i * n_actions_2 + k
                        constraint[idx] = payoff_2[i, k] - payoff_2[i, l]
                    constraints_ub.append(-constraint)
        
        A_ub = np.array(constraints_ub) if constraints_ub else None
        b_ub = np.zeros(len(constraints_ub)) if constraints_ub else None
        
        # Equality constraint: probabilities sum to 1
        A_eq = np.ones((1, n_vars))
        b_eq = np.array([1])
        
        # Bounds: probabilities are non-negative
        bounds = [(0, None) for _ in range(n_vars)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if result.success:
            joint_dist = result.x.reshape(n_actions_1, n_actions_2)
            return joint_dist
        else:
            # Fallback to uniform distribution
            return np.ones((n_actions_1, n_actions_2)) / (n_actions_1 * n_actions_2)
    
    def sample_action(self, joint_distribution: np.ndarray) -> Tuple[int, int]:
        """
        Sample joint action from correlated equilibrium distribution.
        
        Args:
            joint_distribution: Joint probability distribution
        
        Returns:
            Tuple of actions for each player
        """
        flat_dist = joint_distribution.flatten()
        flat_dist = flat_dist / flat_dist.sum()  # Normalize
        
        n_actions_1, n_actions_2 = joint_distribution.shape
        
        # Sample from joint distribution
        idx = np.random.choice(len(flat_dist), p=flat_dist)
        action_1 = idx // n_actions_2
        action_2 = idx % n_actions_2
        
        return action_1, action_2
    
    def get_conditional_strategy(self, joint_distribution: np.ndarray, 
                                player: int, signal: int) -> np.ndarray:
        """
        Get conditional strategy for a player given a signal (recommended action).
        
        Args:
            joint_distribution: Joint probability distribution
            player: Player index (0 or 1)
            signal: Recommended action for this player
        
        Returns:
            Conditional probability distribution for the other player
        """
        if player == 0:
            # Player 1's strategy given signal
            marginal = joint_distribution[signal, :].sum()
            if marginal > 0:
                return joint_distribution[signal, :] / marginal
            else:
                return np.ones(joint_distribution.shape[1]) / joint_distribution.shape[1]
        else:
            # Player 2's strategy given signal
            marginal = joint_distribution[:, signal].sum()
            if marginal > 0:
                return joint_distribution[:, signal] / marginal
            else:
                return np.ones(joint_distribution.shape[0]) / joint_distribution.shape[0]
    
    def compute_social_welfare(self, joint_distribution: np.ndarray,
                              payoff_matrices: List[np.ndarray]) -> float:
        """
        Compute expected social welfare under the correlated equilibrium.
        
        Args:
            joint_distribution: Joint probability distribution
            payoff_matrices: List of payoff matrices for each player
        
        Returns:
            Expected social welfare
        """
        welfare = 0.0
        for payoff in payoff_matrices:
            welfare += np.sum(joint_distribution * payoff)
        return welfare
    
    def guide_policy(self, q_values: torch.Tensor, 
                    equilibrium_distribution: np.ndarray,
                    agent_id: int,
                    temperature: float = 1.0,
                    alpha: float = 0.6) -> torch.Tensor:
        """
        Guide policy towards correlated equilibrium strategy.
        
        Args:
            q_values: Q-values for each action (shape: [batch_size, n_actions])
            equilibrium_distribution: Joint probability distribution
            agent_id: Agent index
            temperature: Temperature parameter for soft guidance
            alpha: Weight for Q-learning (0-1)
        
        Returns:
            Guided action probabilities
        """
        # Compute marginal distribution for this agent
        if agent_id == 0:
            marginal = equilibrium_distribution.sum(axis=1)
        else:
            marginal = equilibrium_distribution.sum(axis=0)
        
        eq_strategy = torch.from_numpy(marginal).float().to(q_values.device)
        eq_strategy = eq_strategy / eq_strategy.sum()  # Normalize
        
        # Compute softmax of Q-values
        q_probs = torch.softmax(q_values / temperature, dim=-1)
        
        # Blend Q-value policy with equilibrium strategy
        guided_probs = alpha * q_probs + (1 - alpha) * eq_strategy.unsqueeze(0)
        
        return guided_probs
