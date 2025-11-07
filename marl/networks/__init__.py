"""
Neural Network Architectures for MARL
"""

from marl.networks.actor import Actor, ContinuousActor
from marl.networks.critic import Critic, CentralizedCritic

__all__ = ['Actor', 'ContinuousActor', 'Critic', 'CentralizedCritic']
