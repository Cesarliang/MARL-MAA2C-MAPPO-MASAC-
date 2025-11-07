"""
Logger for Training Metrics
"""

import os
import json
from typing import Dict, Any, Optional
from collections import defaultdict


class Logger:
    """
    Simple logger for tracking training metrics.
    """
    
    def __init__(self, log_dir: str = './logs'):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.episode_count = 0
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if step is not None:
            metrics['step'] = step
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def log_episode(self, episode_reward: float, episode_length: int,
                   additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log episode-level metrics.
        
        Args:
            episode_reward: Total episode reward
            episode_length: Episode length
            additional_metrics: Additional metrics to log
        """
        self.episode_count += 1
        self.metrics['episode_reward'].append(episode_reward)
        self.metrics['episode_length'].append(episode_length)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.metrics[key].append(value)
    
    def print_summary(self, window: int = 100):
        """
        Print summary of recent episodes.
        
        Args:
            window: Number of recent episodes to average
        """
        if len(self.metrics['episode_reward']) == 0:
            return
        
        recent_rewards = self.metrics['episode_reward'][-window:]
        recent_lengths = self.metrics['episode_length'][-window:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_length = sum(recent_lengths) / len(recent_lengths)
        
        print(f"Episode {self.episode_count} | "
              f"Avg Reward (last {window}): {avg_reward:.2f} | "
              f"Avg Length: {avg_length:.2f}")
    
    def save(self, filename: str = 'metrics.json'):
        """
        Save metrics to file.
        
        Args:
            filename: Name of file to save metrics
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def get_metric(self, name: str) -> list:
        """
        Get specific metric values.
        
        Args:
            name: Metric name
        
        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])
