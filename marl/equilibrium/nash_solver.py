"""
Nash Equilibrium Solver for Multi-Agent Games
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linprog


class NashEquilibriumSolver:
    """
    Solver for computing Nash Equilibrium in multi-agent games.
    Supports both pure and mixed strategy equilibria.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Convergence threshold for iterative methods
        """
        self.epsilon = epsilon
    
    def solve_two_player_zero_sum(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve Nash Equilibrium for two-player zero-sum game using Linear Programming.
        
        Args:
            payoff_matrix: Payoff matrix for player 1 (shape: [n_actions_p1, n_actions_p2])
        
        Returns:
            strategy: Mixed strategy for player 1
            value: Value of the game
        """
        n_actions = payoff_matrix.shape[0]
        m_actions = payoff_matrix.shape[1]
        
        # Player 1's strategy (maximizer)
        # max v s.t. A^T * x >= v * 1, sum(x) = 1, x >= 0
        c = np.zeros(n_actions + 1)
        c[-1] = -1  # maximize v
        
        # Constraints: -A^T * x + v * 1 <= 0
        A_ub = np.hstack([-payoff_matrix.T, np.ones((m_actions, 1))])
        b_ub = np.zeros(m_actions)
        
        # Equality constraint: sum(x) = 1
        A_eq = np.zeros((1, n_actions + 1))
        A_eq[0, :n_actions] = 1
        b_eq = np.array([1])
        
        # Bounds
        bounds = [(0, None) for _ in range(n_actions)] + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            strategy = result.x[:n_actions]
            value = result.x[-1]
            return strategy, value
        else:
            # Fallback to uniform strategy
            return np.ones(n_actions) / n_actions, 0.0
    
    def fictitious_play(self, payoff_matrices: List[np.ndarray], 
                       max_iterations: int = 10000) -> List[np.ndarray]:
        """
        Compute approximate Nash Equilibrium using Fictitious Play.
        
        Args:
            payoff_matrices: List of payoff matrices for each agent
            max_iterations: Maximum number of iterations
        
        Returns:
            strategies: List of mixed strategies for each agent
        """
        n_agents = len(payoff_matrices)
        n_actions = [pm.shape[0] for pm in payoff_matrices]
        
        # Initialize with uniform strategies
        strategies = [np.ones(n) / n for n in n_actions]
        belief_counts = [np.ones(n) for n in n_actions]
        
        for iteration in range(max_iterations):
            old_strategies = [s.copy() for s in strategies]
            
            for agent_id in range(n_agents):
                # Compute best response to current beliefs
                payoff = payoff_matrices[agent_id]
                
                # For simplicity, assume 2-player games
                if n_agents == 2:
                    other_id = 1 - agent_id
                    expected_payoffs = payoff @ strategies[other_id]
                    best_action = np.argmax(expected_payoffs)
                    
                    # Update belief counts
                    belief_counts[agent_id][best_action] += 1
                    strategies[agent_id] = belief_counts[agent_id] / belief_counts[agent_id].sum()
            
            # Check convergence
            converged = all(
                np.linalg.norm(strategies[i] - old_strategies[i]) < self.epsilon
                for i in range(n_agents)
            )
            
            if converged:
                break
        
        return strategies
    
    def support_enumeration(self, payoff_matrices: List[np.ndarray]) -> List[np.ndarray]:
        """
        Find Nash Equilibrium using support enumeration (for small games).
        
        Args:
            payoff_matrices: List of payoff matrices for each agent
        
        Returns:
            strategies: List of mixed strategies for each agent
        """
        # For simplicity, return uniform strategies
        # Full implementation would enumerate all possible supports
        n_actions = [pm.shape[0] for pm in payoff_matrices]
        return [np.ones(n) / n for n in n_actions]
    
    def guide_policy(self, q_values: torch.Tensor, equilibrium_strategy: np.ndarray,
                    temperature: float = 1.0) -> torch.Tensor:
        """
        Guide policy towards equilibrium strategy while maintaining exploration.
        
        Args:
            q_values: Q-values for each action (shape: [batch_size, n_actions])
            equilibrium_strategy: Nash equilibrium mixed strategy
            temperature: Temperature parameter for soft guidance
        
        Returns:
            Guided action probabilities
        """
        # Convert equilibrium strategy to torch tensor
        eq_strategy = torch.from_numpy(equilibrium_strategy).float().to(q_values.device)
        
        # Compute softmax of Q-values
        q_probs = torch.softmax(q_values / temperature, dim=-1)
        
        # Blend Q-value policy with equilibrium strategy
        alpha = 0.5  # Balance between Q-learning and equilibrium
        guided_probs = alpha * q_probs + (1 - alpha) * eq_strategy.unsqueeze(0)
        
        return guided_probs
