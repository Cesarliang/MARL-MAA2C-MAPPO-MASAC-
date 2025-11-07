"""
Matrix Game Environment for Multi-Agent RL
"""

import numpy as np
from typing import List, Tuple, Optional


class MatrixGame:
    """
    Simple matrix game environment for testing multi-agent algorithms.
    Supports various classic games like Prisoner's Dilemma, Coordination Game, etc.
    """
    
    PRISONER_DILEMMA = np.array([
        [[-1, -1], [-3, 0]],
        [[0, -3], [-2, -2]]
    ])
    
    COORDINATION_GAME = np.array([
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ])
    
    MATCHING_PENNIES = np.array([
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]]
    ])
    
    def __init__(self, payoff_matrices: Optional[np.ndarray] = None, 
                 game_type: str = 'prisoner_dilemma',
                 max_steps: int = 1):
        """
        Args:
            payoff_matrices: Custom payoff matrices [n_agents, n_actions_1, n_actions_2]
            game_type: Type of predefined game ('prisoner_dilemma', 'coordination', 'matching_pennies')
            max_steps: Maximum steps per episode (1 for one-shot games)
        """
        self.max_steps = max_steps
        self.current_step = 0
        
        if payoff_matrices is not None:
            self.payoff_matrices = payoff_matrices
        else:
            if game_type == 'prisoner_dilemma':
                self.payoff_matrices = self.PRISONER_DILEMMA
            elif game_type == 'coordination':
                self.payoff_matrices = self.COORDINATION_GAME
            elif game_type == 'matching_pennies':
                self.payoff_matrices = self.MATCHING_PENNIES
            else:
                raise ValueError(f"Unknown game type: {game_type}")
        
        self.n_agents = self.payoff_matrices.shape[0]
        self.n_actions = [self.payoff_matrices.shape[1], self.payoff_matrices.shape[2]]
        
        # State is just a placeholder for matrix games
        self.state_dim = 2  # Minimal state representation
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset environment.
        
        Returns:
            Initial states for each agent
        """
        self.current_step = 0
        # Return dummy states (matrix games don't have meaningful states)
        return [np.array([1.0, 0.0]) for _ in range(self.n_agents)]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent
        
        Returns:
            next_states: Next states for each agent
            rewards: Rewards for each agent
            dones: Done flags for each agent
            info: Additional information
        """
        self.current_step += 1
        
        # Compute rewards from payoff matrices
        rewards = []
        for agent_id in range(self.n_agents):
            if self.n_agents == 2:
                reward = self.payoff_matrices[agent_id, actions[0], actions[1]]
            else:
                # For more than 2 agents, use simplified payoff
                reward = self.payoff_matrices[agent_id].flatten()[
                    sum([actions[i] * (self.n_actions[i] ** i) for i in range(self.n_agents)])
                ]
            rewards.append(float(reward))
        
        # Check if episode is done
        dones = [self.current_step >= self.max_steps] * self.n_agents
        
        # Next states (same as current for one-shot games)
        next_states = [np.array([1.0, 0.0]) for _ in range(self.n_agents)]
        
        info = {'step': self.current_step}
        
        return next_states, rewards, dones, info
    
    def get_payoff_matrices(self) -> np.ndarray:
        """Get payoff matrices for equilibrium computation."""
        return self.payoff_matrices
    
    def render(self):
        """Render the environment (not implemented for matrix games)."""
        pass


class RepeatedMatrixGame(MatrixGame):
    """
    Repeated matrix game with history-dependent states.
    """
    
    def __init__(self, payoff_matrices: Optional[np.ndarray] = None,
                 game_type: str = 'prisoner_dilemma',
                 max_steps: int = 100,
                 history_length: int = 5):
        """
        Args:
            payoff_matrices: Custom payoff matrices
            game_type: Type of predefined game
            max_steps: Maximum steps per episode
            history_length: Length of action history to include in state
        """
        super().__init__(payoff_matrices, game_type, max_steps)
        self.history_length = history_length
        self.state_dim = 2 + history_length * self.n_agents  # Current step + action history
        self.action_history = []
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment with history."""
        self.current_step = 0
        self.action_history = []
        
        # Initial state includes step info and empty history
        states = []
        for _ in range(self.n_agents):
            state = np.zeros(self.state_dim)
            state[0] = 1.0  # Reset signal
            states.append(state)
        
        return states
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], dict]:
        """Execute step with history tracking."""
        # Store action history
        self.action_history.append(actions.copy())
        if len(self.action_history) > self.history_length:
            self.action_history.pop(0)
        
        # Get rewards using parent class method
        _, rewards, dones, info = super().step(actions)
        
        # Construct states with history
        next_states = []
        for agent_id in range(self.n_agents):
            state = np.zeros(self.state_dim)
            state[0] = 0.0  # Not a reset
            state[1] = self.current_step / self.max_steps  # Normalized step
            
            # Add action history
            for i, hist_actions in enumerate(self.action_history):
                offset = 2 + i * self.n_agents
                for j, action in enumerate(hist_actions):
                    if offset + j < self.state_dim:
                        state[offset + j] = action
            
            next_states.append(state)
        
        return next_states, rewards, dones, info
