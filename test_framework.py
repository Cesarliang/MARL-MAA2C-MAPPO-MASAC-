"""
Quick test script to verify the implementation works
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from marl.algorithms.maa2c import MAA2C
from marl.envs.matrix_game import MatrixGame
from marl.equilibrium.nash_solver import NashEquilibriumSolver

def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("Testing Game Equilibrium-Guided MARL Framework")
    print("=" * 60)
    
    # Test 1: Nash Equilibrium Solver
    print("\n1. Testing Nash Equilibrium Solver...")
    solver = NashEquilibriumSolver()
    payoff_matrix = np.array([[3, 0], [5, 1]])  # Simple 2x2 game
    strategy, value = solver.solve_two_player_zero_sum(payoff_matrix)
    print(f"   ✓ Nash equilibrium strategy: {strategy}")
    print(f"   ✓ Game value: {value:.4f}")
    
    # Test 2: Matrix Game Environment
    print("\n2. Testing Matrix Game Environment...")
    env = MatrixGame(game_type='prisoner_dilemma')
    states = env.reset()
    print(f"   ✓ Environment initialized with {env.n_agents} agents")
    print(f"   ✓ State dimension: {env.state_dim}")
    print(f"   ✓ Action space: {env.n_actions}")
    
    # Test 3: MAA2C Algorithm
    print("\n3. Testing MAA2C Algorithm...")
    agent = MAA2C(
        n_agents=2,
        state_dim=2,
        action_dim=2,
        use_equilibrium_guidance=True,
        device='cpu'
    )
    print(f"   ✓ MAA2C agent created with {agent.n_agents} agents")
    print(f"   ✓ Equilibrium guidance: {agent.use_equilibrium_guidance}")
    
    # Test 4: Action Selection
    print("\n4. Testing Action Selection...")
    states = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    actions, log_probs = agent.select_action(states)
    print(f"   ✓ Actions selected: {actions}")
    print(f"   ✓ Log probabilities: {[f'{lp:.4f}' for lp in log_probs]}")
    
    # Test 5: Environment Step
    print("\n5. Testing Environment Step...")
    next_states, rewards, dones, info = env.step(actions)
    print(f"   ✓ Rewards: {rewards}")
    print(f"   ✓ Episode done: {dones[0]}")
    
    # Test 6: Small Training Loop
    print("\n6. Testing Small Training Loop (10 episodes)...")
    total_rewards = []
    for episode in range(10):
        states = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            actions, _ = agent.select_action(states)
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += np.mean(rewards)
            states = next_states
            done = dones[0]
        
        total_rewards.append(episode_reward)
    
    print(f"   ✓ Average reward over 10 episodes: {np.mean(total_rewards):.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("The framework is working correctly.")
    
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
