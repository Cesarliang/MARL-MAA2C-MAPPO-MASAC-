"""
Training script for MAA2C on Matrix Games
"""

import numpy as np
import torch
from marl.algorithms.maa2c import MAA2C
from marl.envs.matrix_game import MatrixGame, RepeatedMatrixGame
from marl.utils.logger import Logger


def train_maa2c(game_type='prisoner_dilemma', 
                n_episodes=5000,
                use_equilibrium=True):
    """
    Train MAA2C on a matrix game.
    
    Args:
        game_type: Type of matrix game
        n_episodes: Number of training episodes
        use_equilibrium: Whether to use equilibrium guidance
    """
    # Create environment
    env = RepeatedMatrixGame(game_type=game_type, max_steps=20)
    
    # Create algorithm
    agent = MAA2C(
        n_agents=env.n_agents,
        state_dim=env.state_dim,
        action_dim=env.n_actions[0],  # Assuming symmetric actions
        use_equilibrium_guidance=use_equilibrium,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create logger
    logger = Logger(log_dir=f'./logs/maa2c_{game_type}')
    
    print(f"Training MAA2C on {game_type}")
    print(f"Equilibrium guidance: {use_equilibrium}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    # Training loop
    for episode in range(n_episodes):
        states = env.reset()
        episode_rewards = [0.0] * env.n_agents
        episode_length = 0
        
        # Collect episode data
        episode_data = {
            'states': [[] for _ in range(env.n_agents)],
            'actions': [[] for _ in range(env.n_agents)],
            'rewards': [[] for _ in range(env.n_agents)],
            'returns': [[] for _ in range(env.n_agents)],
        }
        
        done = False
        while not done:
            # Select actions
            actions, log_probs = agent.select_action(states)
            
            # Execute actions
            next_states, rewards, dones, info = env.step(actions)
            
            # Store transition
            for i in range(env.n_agents):
                episode_data['states'][i].append(states[i])
                episode_data['actions'][i].append(actions[i])
                episode_data['rewards'][i].append(rewards[i])
                episode_rewards[i] += rewards[i]
            
            states = next_states
            episode_length += 1
            done = dones[0]
        
        # Compute returns
        for i in range(env.n_agents):
            returns = agent.compute_returns(
                episode_data['rewards'][i],
                [False] * (episode_length - 1) + [True],
                states[i],
                i
            )
            episode_data['returns'][i] = returns
        
        # Compute equilibrium guidance if enabled
        if use_equilibrium:
            # Get approximate Q-values from critic
            q_values_list = []
            for i in range(env.n_agents):
                # Simple approximation: use payoff matrix
                payoff_matrices = env.get_payoff_matrices()
                q_values_list.append(payoff_matrices[i])
            
            equilibrium_strategies = agent.compute_equilibrium_guidance(q_values_list)
            episode_data['equilibrium_strategy'] = equilibrium_strategies
        
        # Update agent
        metrics = agent.update(episode_data)
        
        # Log metrics
        logger.log_episode(
            episode_reward=np.mean(episode_rewards),
            episode_length=episode_length,
            additional_metrics=metrics
        )
        
        # Print progress
        if (episode + 1) % 100 == 0:
            logger.print_summary(window=100)
    
    # Save model
    agent.save(f'./models/maa2c_{game_type}.pt')
    logger.save()
    
    print("\nTraining completed!")
    print(f"Model saved to: ./models/maa2c_{game_type}.pt")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MAA2C on matrix games')
    parser.add_argument('--game', type=str, default='prisoner_dilemma',
                       choices=['prisoner_dilemma', 'coordination', 'matching_pennies'],
                       help='Type of matrix game')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--no-equilibrium', action='store_true',
                       help='Disable equilibrium guidance')
    
    args = parser.parse_args()
    
    train_maa2c(
        game_type=args.game,
        n_episodes=args.episodes,
        use_equilibrium=not args.no_equilibrium
    )
