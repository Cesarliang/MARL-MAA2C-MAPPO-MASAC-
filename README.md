# 博弈均衡引导的多智能体强化学习 (Game Equilibrium-Guided Multi-Agent Reinforcement Learning)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [中文](#中文)

## 中文

### 简介

这是一个基于PyTorch实现的多智能体强化学习(MARL)框架，融合了博弈论均衡概念来指导多智能体学习。该框架实现了三种主流的MARL算法，每种算法都集成了博弈均衡引导机制：

- **MAA2C**: 多智能体优势演员-评论家算法，结合Nash均衡引导
- **MAPPO**: 多智能体近端策略优化算法，结合相关均衡(Correlated Equilibrium)引导
- **MASAC**: 多智能体软演员-评论家算法，支持连续动作空间的Nash均衡引导

### 核心特性

1. **博弈均衡计算模块**
   - Nash均衡求解器（支持两人零和博弈、虚拟博弈等）
   - 相关均衡求解器（提升社会福利和协调性）
   - 均衡策略与强化学习策略的混合机制

2. **三种MARL算法实现**
   - MAA2C: 基于策略梯度的多智能体算法
   - MAPPO: 采用集中训练分散执行(CTDE)范式
   - MASAC: 支持连续动作空间的软Q学习

3. **灵活的神经网络架构**
   - 离散和连续动作空间的Actor网络
   - 去中心化和中心化的Critic网络
   - 可配置的隐藏层维度

4. **完整的实验环境**
   - 矩阵博弈环境（囚徒困境、协调博弈、匹配硬币等）
   - 重复博弈环境（支持历史信息）
   - 易于扩展的环境接口

### 安装

```bash
git clone https://github.com/Cesarliang/MARL-MAA2C-MAPPO-MASAC-.git
cd MARL-MAA2C-MAPPO-MASAC-
pip install -r requirements.txt
```

### 快速开始

#### 训练MAA2C算法

```bash
python examples/train_maa2c.py --game prisoner_dilemma --episodes 5000
```

#### 训练MAPPO算法

```bash
python examples/train_mappo.py --game coordination --episodes 3000
```

#### 禁用均衡引导

```bash
python examples/train_maa2c.py --game coordination --no-equilibrium
```

### 项目结构

```
MARL-MAA2C-MAPPO-MASAC-/
├── marl/
│   ├── algorithms/          # MARL算法实现
│   │   ├── maa2c.py        # MAA2C算法
│   │   ├── mappo.py        # MAPPO算法
│   │   └── masac.py        # MASAC算法
│   ├── equilibrium/         # 博弈均衡模块
│   │   ├── nash_solver.py  # Nash均衡求解器
│   │   └── correlated_solver.py  # 相关均衡求解器
│   ├── networks/            # 神经网络架构
│   │   ├── actor.py        # Actor网络
│   │   └── critic.py       # Critic网络
│   ├── envs/                # 多智能体环境
│   │   └── matrix_game.py  # 矩阵博弈环境
│   └── utils/               # 工具模块
│       ├── replay_buffer.py # 经验回放缓冲区
│       └── logger.py        # 日志记录器
├── examples/                # 训练示例
│   ├── train_maa2c.py
│   └── train_mappo.py
└── requirements.txt         # 依赖包
```

### 算法说明

#### MAA2C (Multi-Agent Advantage Actor-Critic)

MAA2C是A2C算法的多智能体扩展，结合Nash均衡引导：

- 每个智能体维护独立的Actor和Critic网络
- 通过Nash均衡求解器计算均衡策略
- 使用KL散度将策略引导向均衡方向

#### MAPPO (Multi-Agent Proximal Policy Optimization)

MAPPO采用集中训练分散执行范式：

- 去中心化Actor用于决策执行
- 中心化Critic利用全局信息
- 结合相关均衡提升协调能力
- PPO的clip机制保证训练稳定性

#### MASAC (Multi-Agent Soft Actor-Critic)

MASAC支持连续动作空间：

- 最大化熵正则化的期望回报
- 双Q网络减少价值估计偏差
- 自动调节温度参数
- Nash均衡引导连续策略

### 博弈均衡引导机制

#### Nash均衡

- 适用于竞争性场景
- 提供稳定的策略基准
- 避免被对手利用

#### 相关均衡

- 适用于合作性场景  
- 通过公共信号协调智能体
- 可实现更高的社会福利

### 参数配置

主要超参数：

- `use_equilibrium_guidance`: 是否启用均衡引导（默认True）
- `equilibrium_weight`: 均衡引导的权重（0-1之间，默认0.3）
- `lr_actor`: Actor学习率（默认3e-4）
- `lr_critic`: Critic学习率（默认1e-3）
- `gamma`: 折扣因子（默认0.99）

### 实验结果

在经典矩阵博弈中的表现：

- **囚徒困境**: Nash均衡引导帮助智能体收敛到稳定策略
- **协调博弈**: 相关均衡引导提升了协调效率
- **匹配硬币**: 混合策略均衡指导智能体避免被预测

---

## English

### Introduction

A PyTorch-based Multi-Agent Reinforcement Learning (MARL) framework that integrates game-theoretic equilibrium concepts to guide multi-agent learning. The framework implements three mainstream MARL algorithms, each integrated with game equilibrium guidance mechanisms:

- **MAA2C**: Multi-Agent Advantage Actor-Critic with Nash Equilibrium guidance
- **MAPPO**: Multi-Agent Proximal Policy Optimization with Correlated Equilibrium guidance
- **MASAC**: Multi-Agent Soft Actor-Critic with Nash Equilibrium guidance for continuous actions

### Key Features

1. **Game Equilibrium Computation Module**
   - Nash Equilibrium solver (supports two-player zero-sum games, fictitious play, etc.)
   - Correlated Equilibrium solver (improves social welfare and coordination)
   - Mechanism for blending equilibrium strategies with RL policies

2. **Three MARL Algorithm Implementations**
   - MAA2C: Policy gradient-based multi-agent algorithm
   - MAPPO: Uses Centralized Training with Decentralized Execution (CTDE) paradigm
   - MASAC: Soft Q-learning for continuous action spaces

3. **Flexible Neural Network Architectures**
   - Actor networks for discrete and continuous action spaces
   - Decentralized and centralized Critic networks
   - Configurable hidden layer dimensions

4. **Complete Experimental Environments**
   - Matrix game environments (Prisoner's Dilemma, Coordination Game, Matching Pennies, etc.)
   - Repeated game environments (with history information)
   - Easy-to-extend environment interface

### Installation

```bash
git clone https://github.com/Cesarliang/MARL-MAA2C-MAPPO-MASAC-.git
cd MARL-MAA2C-MAPPO-MASAC-
pip install -r requirements.txt
```

### Quick Start

#### Train MAA2C

```bash
python examples/train_maa2c.py --game prisoner_dilemma --episodes 5000
```

#### Train MAPPO

```bash
python examples/train_mappo.py --game coordination --episodes 3000
```

#### Disable Equilibrium Guidance

```bash
python examples/train_maa2c.py --game coordination --no-equilibrium
```

### Project Structure

```
MARL-MAA2C-MAPPO-MASAC-/
├── marl/
│   ├── algorithms/          # MARL algorithm implementations
│   │   ├── maa2c.py        # MAA2C algorithm
│   │   ├── mappo.py        # MAPPO algorithm
│   │   └── masac.py        # MASAC algorithm
│   ├── equilibrium/         # Game equilibrium module
│   │   ├── nash_solver.py  # Nash equilibrium solver
│   │   └── correlated_solver.py  # Correlated equilibrium solver
│   ├── networks/            # Neural network architectures
│   │   ├── actor.py        # Actor networks
│   │   └── critic.py       # Critic networks
│   ├── envs/                # Multi-agent environments
│   │   └── matrix_game.py  # Matrix game environment
│   └── utils/               # Utility modules
│       ├── replay_buffer.py # Experience replay buffer
│       └── logger.py        # Logger
├── examples/                # Training examples
│   ├── train_maa2c.py
│   └── train_mappo.py
└── requirements.txt         # Dependencies
```

### Algorithm Details

#### MAA2C (Multi-Agent Advantage Actor-Critic)

MAA2C extends A2C to multi-agent settings with Nash equilibrium guidance:

- Each agent maintains independent Actor and Critic networks
- Computes equilibrium strategies using Nash equilibrium solver
- Guides policies toward equilibrium using KL divergence

#### MAPPO (Multi-Agent Proximal Policy Optimization)

MAPPO uses the Centralized Training with Decentralized Execution paradigm:

- Decentralized Actors for decision-making
- Centralized Critic leverages global information
- Correlated equilibrium improves coordination
- PPO's clipping mechanism ensures training stability

#### MASAC (Multi-Agent Soft Actor-Critic)

MASAC supports continuous action spaces:

- Maximizes entropy-regularized expected return
- Twin Q-networks reduce value estimation bias
- Automatic temperature tuning
- Nash equilibrium guides continuous policies

### Game Equilibrium Guidance

#### Nash Equilibrium

- Suitable for competitive scenarios
- Provides stable strategy baseline
- Prevents exploitation by opponents

#### Correlated Equilibrium

- Suitable for cooperative scenarios
- Coordinates agents through public signals
- Achieves higher social welfare

### Hyperparameter Configuration

Main hyperparameters:

- `use_equilibrium_guidance`: Enable equilibrium guidance (default True)
- `equilibrium_weight`: Weight for equilibrium guidance (0-1, default 0.3)
- `lr_actor`: Actor learning rate (default 3e-4)
- `lr_critic`: Critic learning rate (default 1e-3)
- `gamma`: Discount factor (default 0.99)

### Experimental Results

Performance on classic matrix games:

- **Prisoner's Dilemma**: Nash equilibrium guidance helps agents converge to stable strategies
- **Coordination Game**: Correlated equilibrium guidance improves coordination efficiency
- **Matching Pennies**: Mixed strategy equilibrium guides agents to avoid being predicted

### Citation

If you use this code in your research, please cite:

```bibtex
@software{marl_equilibrium_2024,
  author = {Cesarliang},
  title = {Game Equilibrium-Guided Multi-Agent Reinforcement Learning},
  year = {2024},
  url = {https://github.com/Cesarliang/MARL-MAA2C-MAPPO-MASAC-}
}
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.
