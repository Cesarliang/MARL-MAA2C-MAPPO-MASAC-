"""
Training script for MAPPO on Matrix Games
"""

import numpy as np
import torch
from marl.algorithms.mappo import MAPPO
from marl.envs.matrix_game import RepeatedMatrixGame
from marl.utils.logger import Logger


def train_mappo(game_type='coordination', 
                n_episodes=3000,
                use_equilibrium=True):
    """
    Train MAPPO on a matrix game.
    
    Args:
        game_type: Type of matrix game
        n_episodes: Number of training episodes
        use_equilibrium: Whether to use equilibrium guidance
    """
    # Create environment
    env = RepeatedMatrixGame(game_type=game_type, max_steps=20)
    
    # Create algorithm
    agent = MAPPO(
        n_agents=env.n_agents,
        state_dim=env.state_dim,
        action_dim=env.n_actions[0],
        global_state_dim=env.state_dim * env.n_agents,
        use_equilibrium_guidance=use_equilibrium,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create logger
    logger = Logger(log_dir=f'./logs/mappo_{game_type}')
    
    print(f"Training MAPPO on {game_type}")
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
            'log_probs': [[] for _ in range(env.n_agents)],
            'values': [[] for _ in range(env.n_agents)],
            'global_states': []
        }
        
        done = False
        while not done:
            # Select actions
            actions, log_probs, values = agent.select_action(states)
            
            # Execute actions
            next_states, rewards, dones, info = env.step(actions)
            
            # Store transition
            global_state = np.concatenate(states)
            episode_data['global_states'].append(global_state)
            
            for i in range(env.n_agents):
                episode_data['states'][i].append(states[i])
                episode_data['actions'][i].append(actions[i])
                episode_data['rewards'][i].append(rewards[i])
                episode_data['log_probs'][i].append(log_probs[i])
                episode_data['values'][i].append(values[i])
                episode_rewards[i] += rewards[i]
            
            states = next_states
            episode_length += 1
            done = dones[0]
        
        # Compute next values
        _, _, next_values = agent.select_action(states)
        
        # Compute advantages using GAE
        advantages = agent.compute_gae(
            rewards=[[episode_data['rewards'][i][t] for t in range(episode_length)] 
                    for i in range(env.n_agents)],
            values=[[episode_data['values'][i][t] for t in range(episode_length)]
                   for i in range(env.n_agents)],
            dones=[[False] * (episode_length - 1) + [True]] * env.n_agents,
            next_values=next_values
        )
        
        episode_data['advantages'] = advantages
        
        # Compute returns
        returns = []
        for i in range(env.n_agents):
            agent_returns = [advantages[i][t] + episode_data['values'][i][t] 
                           for t in range(episode_length)]
            returns.append(agent_returns)
        episode_data['returns'] = returns
        
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
    agent.save(f'./models/mappo_{game_type}.pt')
    logger.save()
    
    print("\nTraining completed!")
    print(f"Model saved to: ./models/mappo_{game_type}.pt")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MAPPO on matrix games')
    parser.add_argument('--game', type=str, default='coordination',
                       choices=['prisoner_dilemma', 'coordination', 'matching_pennies'],
                       help='Type of matrix game')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes')
    parser.add_argument('--no-equilibrium', action='store_true',
                       help='Disable equilibrium guidance')
    
    args = parser.parse_args()
    
    train_mappo(
        game_type=args.game,
        n_episodes=args.episodes,
        use_equilibrium=not args.no_equilibrium
    )
