"""
Multi-Agent Reinforcement Learning with Game Equilibrium Guidance
"""

__version__ = "0.1.0"

from marl.algorithms.maa2c import MAA2C
from marl.algorithms.mappo import MAPPO
from marl.algorithms.masac import MASAC

__all__ = ['MAA2C', 'MAPPO', 'MASAC']
