import copy

import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import configparser

config_dir = 'configs/configs_sac.ini'  # 修改为SAC相关的配置文件
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

from typing import Tuple, Union
from torch.optim import adam, rmsprop
import torch.optim as optim
import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import ReplayMemory  # 修改为ReplayMemory，
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder
from collections import OrderedDict
from Model_common import PolicyNet, QValueNet

class MASAC:
    """
    An multi-agent learned with SAC
    """
    def __init__(self, env, state_dim, action_dim, hidden_dim, n_agents,
                 memory_capacity=20000, max_steps=None, actor_hidden_dim=256, critic_hidden_dim=256,
                 roll_out_n_steps=1, reward_gamma=0.99, reward_scale=20,
                 actor_lr=1e-3, critic_lr=1e-3, test_seeds="", alpha=1e-2,discount=0.99,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=256, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R", target_entropy=-1, tau=0.005, gamma=0.95):

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]

        # 初始化基本参数
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state, self.action_mask = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.memory = ReplayMemory(memory_capacity)   # SAC需要使用ReplayMemory
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha = alpha  # SAC中的熵系数
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.test_seeds = test_seeds.strip(',')
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.roll_out_n_steps = roll_out_n_steps
        self.entropy_reg = entropy_reg
        self.device = th.device("cuda" if th.cuda.is_available() and self.use_cuda else "cpu")
        self.n_agents = n_agents

        # 初始化actor网络及其目标网络和优化器
        self.actors = [PolicyNet(self.state_dim, self.actor_hidden_dim, self.action_dim).to(self.device) for _ in range(n_agents)]
        self.actor_targets = [deepcopy(actor).to(self.device) for actor in self.actors]
        if self.optimizer_type == "adam":
            self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizers = [torch.optim.RMSprop(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # 初始化两个critic网络及其目标网络和优化器
        self.critics_1 = [QValueNet(self.state_dim, self.action_dim, self.critic_hidden_dim).to(self.device) for _ in
                          range(self.n_agents)]
        self.critics_2 = [QValueNet(self.state_dim, self.action_dim, self.critic_hidden_dim).to(self.device) for _ in
                          range(self.n_agents)]
        self.critic_targets_1 = [deepcopy(critic).to(self.device) for critic in self.critics_1]
        self.critic_targets_2 = [deepcopy(critic).to(self.device) for critic in self.critics_2]
        if self.optimizer_type == "adam":
            self.critic_optimizers_1 = [torch.optim.Adam(critic.parameters(), lr=self.critic_lr) for critic in
                                        self.critics_1]
            self.critic_optimizers_2 = [torch.optim.Adam(critic.parameters(), lr=self.critic_lr) for critic in
                                        self.critics_2]
        elif self.optimizer_type == "rmsprop":
            self.critic_optimizers_1 = [torch.optim.RMSprop(critic.parameters(), lr=self.critic_lr) for critic in
                                        self.critics_1]
            self.critic_optimizers_2 = [torch.optim.RMSprop(critic.parameters(), lr=self.critic_lr) for critic in
                                        self.critics_2]
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # 将每个智能体的critic网络的参数复制给对应的目标网络
        for i in range(self.n_agents):
            self.critic_targets_1[i].load_state_dict(self.critics_1[i].state_dict())
            self.critic_targets_2[i].load_state_dict(self.critics_2[i].state_dict())

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha)

        if self.use_cuda:
            self.actors = [actor.to(self.device) for actor in self.actors]
            self.critics_1 = [critic.to(self.device) for critic in self.critics_1]
            self.critics_2 = [critic.to(self.device) for critic in self.critics_2]
            self.actor_targets = [actor_target.to(self.device) for actor_target in self.actor_targets]
            self.critic_targets_1 = [critic_target.to(self.device) for critic_target in self.critic_targets_1]
            self.critic_targets_2 = [critic_target.to(self.device) for critic_target in self.critic_targets_2]

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]
        self.headway_distance = [0]

    def alpha_loss(self, states_var):
        alpha = self.log_alpha.exp()
        alpha_loss = 0
        for agent_id in range(self.n_agents):
            mean, log_std, std = self.actors[agent_id](states_var[:, agent_id, :])
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
            alpha_loss += -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return alpha_loss

    def _softmax_action(self, state, n_agents):
        state_var = th.tensor(state, dtype=th.float32)
        softmax_actions = []

        if n_agents > len(self.actors):
            raise ValueError(
                f"Number of agents ({n_agents}) exceeds the number of actor networks ({len(self.actors)})."
            )

        for agent_id in range(n_agents):
            softmax_action_var = th.exp(self.actors[agent_id](state_var[:, agent_id, :]))
            softmax_actions.append(softmax_action_var)

        return softmax_actions

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, _ = self.env.reset()
            self.n_steps = 0

        states = []
        actions = []
        rewards = []
        done = True
        average_speed = 0
        cursh_reward = 0
        headway_diss = 0
        self.cursh = 0
        self.n_agents = len(self.env.controlled_vehicles)

        # 进行n步
        for _ in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state, self.n_agents)
            next_state, global_reward, done, info = self.env.step(action)
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            rewards.append(reward)
            average_speed += info["average_speed"]
            final_state = next_state
            self.env_state = next_state

            cursh_reward += sum(info["cursh_reward"]) + cursh_reward

            headway_distance = info["headway_distance"]
            headway_diss += np.mean(headway_distance)

            self.n_steps += 1

            if done:
                self.env_state, _ = self.env.reset()
                break

        if cursh_reward != 0:
            print("发生撞车")
            self.cursh = 1
            # discount   reward

        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.episode_rewards.append(0)
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.headway_distance[-1] = headway_diss / self.epoch_steps[-1]
            self.average_speed.append(0)
            self.headway_distance.append(0)
            self.epoch_steps.append(0)
        else:
            self.episode_done = False
            final_action = self.action(self.env_state, self.n_agents)
            final_value = self.value(self.env_state, final_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
        self.memory.push(states, actions, rewards)

    def compute_grad_actor(self, model, states_var, actions_var, alpha, v=None):
        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()

        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                if grad is not None:
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})
                else:
                    # 如果 grad 为 None，可以选择跳过或使用原始参数
                    dummy_model_params_1.update({layer_name: param})
                    dummy_model_params_2.update({layer_name: param})

        model.load_state_dict(dummy_model_params_1, strict=False)
        loss_1 = self.actor_loss(model, states_var, actions_var, model, alpha)
        grads_1 = torch.autograd.grad(loss_1, model.parameters(), allow_unused=True)

        model.load_state_dict(dummy_model_params_2, strict=False)
        loss_2 = self.actor_loss(model, states_var, actions_var, model, alpha)
        grads_2 = torch.autograd.grad(loss_2, model.parameters(), allow_unused=True)

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                if g1 is not None and g2 is not None:
                    grads.append((g1 - g2) / (2 * delta))
                else:
                    grads.append(None)
        return grads

    def compute_grad_critic(self, model, states_var, actions_var, rewards_var, next_states_var, dones_var, alpha,
                            v=None):
        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        loss_1 = self.critic_loss(model, states_var, actions_var, rewards_var, next_states_var, dones_var, alpha)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        loss_2 = self.critic_loss(model, states_var, actions_var, rewards_var, next_states_var, dones_var, alpha)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads

    def actor_loss(self, actor, states_var, actions_var, actor_target, alpha):
        if states_var.dim() == 3:
            states_var = states_var.view(-1, states_var.size(-1))
        if actions_var.dim() == 3:
            actions_var = actions_var.view(-1, actions_var.size(-1))
        action_mean, action_log_std, action_std = actor(states_var)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = action_dist.log_prob(actions_var).sum(dim=-1, keepdim=True)

        target_action_mean, target_action_log_std, target_action_std = actor_target(states_var)
        target_action_dist = torch.distributions.Normal(target_action_mean, target_action_std)
        target_action = target_action_dist.rsample()
        target_log_prob = target_action_dist.log_prob(target_action).sum(dim=-1, keepdim=True)

        q_values1 = self.critics_1[0](states_var, actions_var)
        q_values2 = self.critics_2[1](states_var, actions_var)
        q_values = torch.min(q_values1, q_values2)

        actor_loss = (alpha * log_prob - q_values).mean() + (alpha * target_log_prob).mean()
        return actor_loss

    def critic_loss(self, critic, states_var, actions_var, rewards_var, next_states_var, dones_var, alpha):
        with torch.no_grad():
            next_q_values_list = []

            if next_states_var.dim() == 2:
                batch_size = next_states_var.size(0)
                state_dim = next_states_var.size(1) // self.n_agents
                next_states_var = next_states_var.view(batch_size, self.n_agents, state_dim)


            # 遍历每个智能体，计算Q值
            for agent_id in range(self.n_agents):
                next_action_probs = self.actors[agent_id](next_states_var[:, agent_id, :])
                if isinstance(next_action_probs, tuple):
                    next_action_probs = next_action_probs[0]

                next_log_action_probs = torch.log(next_action_probs + 1e-10)

                next_q_values1 = self.critic_targets_1[0](next_states_var[:, agent_id, :], next_action_probs)
                next_q_values2 = self.critic_targets_2[1](next_states_var[:, agent_id, :], next_action_probs)
                next_q_values = torch.min(next_q_values1, next_q_values2)

                next_q_values_list.append(next_q_values - alpha * next_log_action_probs)

            # 堆叠所有 agent 的 Q 值，第二维为智能体数量 (n_agents)
            next_q_values = torch.stack(next_q_values_list, dim=1)

        total_critic_loss = 0

        for agent_id in range(self.n_agents):
            reward = rewards_var[:, agent_id].unsqueeze(-1)

            if dones_var.shape[1] == 1:
                done = dones_var
            else:
                done = dones_var[:, agent_id].unsqueeze(-1)

            # 计算目标 Q 值
            target_q_values = reward + (1 - done) * self.reward_gamma * next_q_values[:, agent_id].view(reward.size(0),
                                                                                                        -1)

            # 计算当前 Q 值
            q_values = critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])

            # 计算 MSE 损失并累加
            critic_loss = nn.MSELoss()(q_values, target_q_values.detach())
            total_critic_loss += critic_loss

        return total_critic_loss / self.n_agents

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        batch = self.memory.sample(self.batch_size)
        batch_size = len(batch.states)  # 获取实际的批次大小

        states_var = to_tensor_var(batch.states, self.use_cuda).view(batch_size, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(batch_size, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(batch_size, self.n_agents, 1)

        # 处理 next_states
        # 处理 next_states
        if isinstance(batch.next_states, tuple):
            # 将tuple转换为numpy数组，同时过滤掉None值
            next_states = np.array([s for s in batch.next_states if s is not None])
        else:
            next_states = batch.next_states

        if len(next_states) == 0:
            if not hasattr(self, '_warning_printed'):
                print("Warning: All next_states are None. Handling this gracefully.")
                self._warning_printed = True
            next_states = np.zeros((batch_size, self.n_agents, self.state_dim))

        if len(next_states.shape) == 2:
            if next_states.shape[1] % self.state_dim != 0:
                raise ValueError(
                    f"next_states shape {next_states.shape} is not compatible with state_dim {self.state_dim}")
            next_states_var = to_tensor_var(next_states, self.use_cuda).view(batch_size, self.n_agents, self.state_dim)
        elif len(next_states.shape) == 3:
            next_states_var = to_tensor_var(next_states, self.use_cuda)
        else:
            raise ValueError(f"Unexpected shape for next_states: {next_states.shape}")

        dones = np.array(batch.dones)[:batch_size]
        if len(dones.shape) == 1:
            dones_var = to_tensor_var(dones, self.use_cuda).view(batch_size, 1).expand(-1, self.n_agents)
        else:
            dones_var = to_tensor_var(dones, self.use_cuda).view(batch_size, self.n_agents)

        # 更新 critic 网络
        for agent_id in range(self.n_agents):
            self.critic_optimizers_1[agent_id].zero_grad()
            critic_loss_1 = self.critic_loss(self.critics_1[agent_id], states_var[:, agent_id],
                                             actions_var[:, agent_id],
                                             rewards_var[:, agent_id], next_states_var[:, agent_id],
                                             dones_var[:, agent_id], self.alpha)
            critic_loss_1.backward()
            self.critic_optimizers_1[agent_id].step()

            self.critic_optimizers_2[agent_id].zero_grad()
            critic_loss_2 = self.critic_loss(self.critics_2[agent_id], states_var[:, agent_id],
                                             actions_var[:, agent_id],
                                             rewards_var[:, agent_id], next_states_var[:, agent_id],
                                             dones_var[:, agent_id], self.alpha)
            critic_loss_2.backward()
            self.critic_optimizers_2[agent_id].step()

        # 存储每个代理的actor网络梯度
        actor_gradients = [[] for _ in range(self.n_agents)]
        for agent_id in range(self.n_agents):
            if agent_id < len(self.actor_optimizers):
                self.actor_optimizers[agent_id].zero_grad()
                actor_loss = self.actor_loss(
                    self.actors[agent_id],
                    states_var[:, agent_id, :],
                    actions_var[:, agent_id, :],
                    self.actor_targets[agent_id],
                    self.alpha
                )
                actor_loss.backward()
                actor_gradients[agent_id] = [param.grad.clone() if param.grad is not None else None for param in
                                             self.actors[agent_id].parameters()]
            else:
                print(f"Warning: No optimizer found for agent {agent_id}")

        global_actor_gradients = []
        for agent_gradients in zip(*actor_gradients):
            valid_gradients = [grad for grad in agent_gradients if grad is not None]
            if valid_gradients:
                avg_gradient = sum(valid_gradients) / len(valid_gradients)
            else:
                avg_gradient = None
            global_actor_gradients.append(avg_gradient)

        for agent_id in range(self.n_agents):
            for param, global_grad in zip(self.actors[agent_id].parameters(), global_actor_gradients):
                if global_grad is not None:
                    param.grad = global_grad.clone()
            self.actor_optimizers[agent_id].step()

        # 软更新目标网络
        for agent_id in range(self.n_agents):
            for target_param, param in zip(self.critic_targets_1[agent_id].parameters(),
                                           self.critics_1[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_targets_2[agent_id].parameters(),
                                           self.critics_2[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 遍历每个智能体的 Actor 网络，并分别对其参数进行更新
        for actor_target, actor in zip(self.actor_targets, self.actors):
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # 将生成器对象转换为列表
        # actor_parameters = list(self.actor.parameters())
        # actor_avg_gradients = [None] * len(actor_parameters)
        actor_gradients1 = []
        actor_gradients2 = []
        actor_gradients3 = []
        actor_gradients0 = []
        critic_gradients1 = []
        critic_gradients2 = []
        critic_gradients3 = []
        critic_gradients0 = []

        # 初始化梯度存储
        #actor_gradients = [[] for _ in range(self.n_agents)]
        #critic_gradients = [[] for _ in range(self.n_agents)]
        #self.actor_optimizers = [Adam(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        #self.critic_optimizers = [Adam(critic.parameters(), lr=self.critic_lr) for critic in self.critics]

        a = 5e-10
        b = 5e-11

        for agent_id in range(self.n_agents):
            self.actor_optimizers[agent_id].zero_grad()
            self.critic_optimizers_1[agent_id].zero_grad()
            self.critic_optimizers_2[agent_id].zero_grad()

            # 计算 actor 损失并进行反向传播
            # 确保传递 actions_var 和 alpha 参数
            actor_loss = self.actor_loss(
                self.actors[agent_id],
                states_var[:, agent_id, :],
                actions_var[:, agent_id, :],
                self.actor_targets[agent_id],
                self.alpha
            )
            actor_loss.backward(retain_graph=True)

        for agent_id in range(self.n_agents):
            for target_param, param in zip(self.critic_targets_1[agent_id].parameters(),
                                           self.critics_1[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_targets_2[agent_id].parameters(),
                                           self.critics_2[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for critic_target, critic in zip(self.critic_targets_1, self.critics_1):
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for critic_target, critic in zip(self.critic_targets_2, self.critics_2):
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


            "////meta"
            # print(self.actors[agent_id].parameters())
            new_params = []

            temp_actor = copy.deepcopy(self.actors[agent_id])
            # temp_actor_target = copy.deepcopy(self.actor_targets[agent_id])

            temp_actor_loss = self.actor_loss(
                temp_actor,
                states_var[:, agent_id, :],
                actions_var[:, agent_id, :],
                self.actor_targets[agent_id],
                self.alpha
            )
            temp_actor_loss.backward(retain_graph=True)
            # for param in temp_actor.parameters():
            #     # print(param)
            #     if param.grad is not None:
            #         print(param.grad)
            grads1 = torch.autograd.grad(temp_actor_loss, temp_actor.parameters(), allow_unused=True)
            # grads1_tar = torch.autograd.grad(temp_actor_loss, temp_actor_target.parameters(), allow_unused=True)

            for param, grad in zip(temp_actor.parameters(), grads1):
                if grad is not None:
                    param.data.sub_(a * grad)
            # for param,grad in zip(temp_actor_target.parameters(), grads1_tar):
            #     param.data.sub_(0.01 * grad)
            # actor_loss_1st = self.actor_loss(temp_actor,states_var[:, agent_id, :],actions_var[:, agent_id, :],self.actor_targets[agent_id],advantages)
            actor_loss_1st = self.actor_loss(
                temp_actor,
                states_var[:, agent_id, :],
                actions_var[:, agent_id, :],
                self.actor_targets[agent_id],
                self.alpha
            )
            grads_1st = torch.autograd.grad(actor_loss_1st, temp_actor.parameters(), allow_unused=True)
            # grads_1st_tar = torch.autograd.grad(actor_loss_1st, temp_actor_target.parameters())

            grads_2nd = self.compute_grad_actor(
                self.actors[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :], self.alpha, v=grads_1st)
            # grads_2nd = self.compute_grad_actor(
            #     self.actors[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
            #     self.actor_targets[agent_id], advantages, v=grads_1st)
            # 获取第三个数据批次 data_batch_3，并使用临时模型、第二次梯度 grads_1st 以及参数 v 和 second_order_grads 执行二阶梯度计算，得到梯度 grads_2nd。
            # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
            for param, grad1, grad2 in zip(self.actors[agent_id].parameters(), grads_1st, grads_2nd):
                if grad1 is not None and grad2 is not None:
                    param.data.sub_(b * grad1 - b * a * grad2)
            #self._soft_update_target(self.actor_targets[agent_id], self.actors[agent_id], tau=self.tau)

            '///'
            #
            # for param in self.actor.parameters():
            #     if param.grad is not None:
            #         print(param.grad,agent_id)
            if agent_id ==0:

            # print(self.actor.parameters())
                for param in self.actors[agent_id].parameters():
                    if param.grad is not None:
                        # print(param.grad)
                        actor_gradients0.append(param.grad.clone())

                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        actor_gradients0.append(None)
            elif agent_id ==1:
                for param in self.actors[agent_id].parameters():
                    if param.grad is not None:

                        actor_gradients1.append(param.grad.clone())

                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        actor_gradients1.append(None)
            elif agent_id ==2:
                for param in self.actors[agent_id].parameters():
                    if param.grad is not None:

                        actor_gradients2.append(param.grad.clone())
                        # actor_gradients.append(agent_id)
                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        actor_gradients2.append(None)
            elif agent_id ==3:
                for param in self.actors[agent_id].parameters():
                    if param.grad is not None:

                        actor_gradients3.append(param.grad.clone())
                        # actor_gradients.append(agent_id)
                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        actor_gradients3.append(None)
            # for i, param_gradient in enumerate(actor_gradients):
            #     if param_gradient is not None:
            #         if actor_avg_gradients[i] is None:
            #             actor_avg_gradients[i] = param_gradient.clone()
            #         else:
            #             actor_avg_gradients[i] += param_gradient
            # print(actor_gradients,"梯度")
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            # self.actor_optimizer.step()

            # update critic network
            # self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values1 = self.critics_1[agent_id](states_var[:, agent_id, :], actions_var[:, agent_id, :])
            values2 = self.critics_2[agent_id](states_var[:, agent_id, :], actions_var[:, agent_id, :])
            values = torch.min(values1, values2)
            # print(target_values,1)
            # print(type(target_values),type(values),values)
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()

            "////meta"
            # print(self.actors[agent_id].parameters())
            new_params = []

            temp_critic1 = copy.deepcopy(self.critics_1[agent_id])
            temp_critic1_loss = self.critic_loss(temp_critic1, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                            rewards_var[:, agent_id].view(-1, 1),
                                            next_states_var[:, agent_id, :],
                                            dones_var[:, agent_id].view(-1, 1), self.alpha)
            temp_critic1_loss.backward(retain_graph=True)
            # for param in temp_actor.parameters():
            #     # print(param)
            #     if param.grad is not None:
            #         print(param.grad)
            grads1_critic1 = torch.autograd.grad(temp_critic1_loss, temp_critic1.parameters(), allow_unused=True)
            temp_critic2 = copy.deepcopy(self.critics_2[agent_id])
            temp_critic2_loss = self.critic_loss(temp_critic2, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                                 rewards_var[:, agent_id].view(-1, 1),
                                                 next_states_var[:, agent_id, :],
                                                 dones_var[:, agent_id].view(-1, 1), self.alpha)
            temp_critic2_loss.backward(retain_graph=True)
            grads1_critic2 = torch.autograd.grad(temp_critic2_loss, temp_critic2.parameters(), allow_unused=True)

            for param, grad in zip(temp_critic1.parameters(), grads1_critic1):
                param.data.sub_(self.alpha * grad)

            for param, grad in zip(temp_critic2.parameters(), grads1_critic2):
                param.data.sub_(self.alpha * grad)

            critic1_loss_1st = self.critic_loss(temp_critic1, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                                rewards_var[:, agent_id].view(-1, 1), next_states_var[:, agent_id, :],
                                                dones_var[:, agent_id].view(-1, 1), self.alpha)

            grads_1st_critic1 = torch.autograd.grad(critic1_loss_1st, temp_critic1.parameters())

            critic2_loss_1st = self.critic_loss(temp_critic2, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                                rewards_var[:, agent_id].view(-1, 1), next_states_var[:, agent_id, :],
                                                dones_var[:, agent_id].view(-1, 1), self.alpha)
            grads_1st_critic2 = torch.autograd.grad(critic2_loss_1st, temp_critic2.parameters())

            grads_2nd_critic1 = self.compute_grad_critic(
            self.critics_1[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
                rewards_var[:, agent_id].view(-1, 1), next_states_var[:, agent_id, :],
                dones_var[:, agent_id].view(-1, 1), v=grads_1st_critic1, alpha=self.alpha  # 传递alpha参数
            )

            grads_2nd_critic2 = self.compute_grad_critic(
                self.critics_2[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
                rewards_var[:, agent_id].view(-1, 1), next_states_var[:, agent_id, :],
                dones_var[:, agent_id].view(-1, 1), v=grads_1st_critic2, alpha=self.alpha  # 传递alpha参数
            )
            # 获取第三个数据批次 data_batch_3，并使用临时模型、第二次梯度 grads_1st 以及参数 v 和 second_order_grads 执行二阶梯度计算，得到梯度 grads_2nd。
            # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
            for param, grad1, grad2 in zip(
                    self.critics_1[agent_id].parameters(), grads_1st_critic1, grads_2nd_critic1
            ):
                param.data.sub_(b * grad1 - b * a * grad2)

            for param, grad1, grad2 in zip(
                    self.critics_2[agent_id].parameters(), grads_1st_critic2, grads_2nd_critic2
            ):
                param.data.sub_(b * grad1 - b * a * grad2)
            # self._soft_update_target(self.critic_targets[agent_id], self.critics[agent_id])

            '///'
            #   求critic梯度
            # for param in self.critic.parameters():
            #     if param.grad is not None:
            #         critic_gradients.append(param.grad)
            #     else:
            #         critic_gradients.append(None)
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critics_1[agent_id].parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critics_2[agent_id].parameters(), self.max_grad_norm)
            if agent_id ==0:
                for param in self.critics_1[agent_id].parameters():
                    if param.grad is not None:

                        critic_gradients0.append(param.grad.clone())
                    else:
                        critic_gradients0.append(None)
            elif agent_id ==1:
                for param in self.critics_1[agent_id].parameters():
                    if param.grad is not None:

                        critic_gradients1.append(param.grad.clone())

                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        critic_gradients1.append(None)
            elif agent_id ==2:
                for param in self.critics_1[agent_id].parameters():
                    if param.grad is not None:

                        critic_gradients2.append(param.grad.clone())
                        # critic_gradients.append(agent_id)
                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        critic_gradients2.append(None)
            elif agent_id ==3:
                for param in self.critics_1[agent_id].parameters():
                    if param.grad is not None:

                        critic_gradients3.append(param.grad.clone())
                        # critic_gradients.append(agent_id)
                        # print(param.grad.data,agent_id,"梯度")
                    else:
                        critic_gradients3.append(None)
            # print(critic_loss.item(),self.n_episodes)

        actor_avg_gradients = []
        critic_avg_gradients = []
        min_length = min(len(actor_gradients0), len(actor_gradients1), len(actor_gradients2), len(actor_gradients3))

        if self.n_agents == 3:
            for i in range(min_length):
                actor_avg_gradients.append((actor_gradients0[i] + actor_gradients1[i] + actor_gradients2[i]) / 3)
        elif self.n_agents == 4:
            for i in range(min_length):
                actor_avg_gradients.append(
                    (actor_gradients0[i] + actor_gradients1[i] + actor_gradients2[i] + actor_gradients3[i]) / 4)
        elif self.n_agents == 2:
            for i in range(min(len(actor_gradients0), len(actor_gradients1))):
                actor_avg_gradients.append((actor_gradients0[i] + actor_gradients1[i]) / 2)
        elif self.n_agents == 1:
            for i in range(len(actor_gradients0)):
                actor_avg_gradients.append(actor_gradients0[i])

        min_length_critic = min(len(critic_gradients0), len(critic_gradients1), len(critic_gradients2),
                                len(critic_gradients3))

        if self.n_agents == 3:
            for i in range(min_length_critic):
                critic_avg_gradients.append((critic_gradients0[i] + critic_gradients1[i] + critic_gradients2[i]) / 3)
        elif self.n_agents == 4:
            for i in range(min_length_critic):
                critic_avg_gradients.append(
                    (critic_gradients0[i] + critic_gradients1[i] + critic_gradients2[i] + critic_gradients3[i]) / 4)
        elif self.n_agents == 2:
            for i in range(min(len(critic_gradients0), len(critic_gradients1))):
                critic_avg_gradients.append((critic_gradients0[i] + critic_gradients1[i]) / 2)
        elif self.n_agents == 1:
            for i in range(len(critic_gradients0)):
                critic_avg_gradients.append(critic_gradients0[i])
        i = 0

        # print(actor_avg_gradients,'pingjun')
        # for param in self.actor.parameters():
        #
        #     param.gard = actor_avg_gradients[i]
        #     print(i,param.grad)
        #     i  = i +1
        for agent_id in range(self.n_agents):
            for i, param in enumerate(self.actors[agent_id].parameters()):
                if i < len(actor_avg_gradients) and actor_avg_gradients[i] is not None:
                    param.grad = actor_avg_gradients[i]
                    # print(i,param.grad)
            self.actor_optimizers[agent_id].step()

        # print(critic_avg_gradients,'pingjun')
        for agent_id in range(self.n_agents):
            for i, param in enumerate(self.critics_1[agent_id].parameters()):
                if i < len(critic_avg_gradients) and critic_avg_gradients[i] is not None:
                    param.grad = critic_avg_gradients[i]
            self.critic_optimizers_1[agent_id].step()

            for i, param in enumerate(self.critics_2[agent_id].parameters()):
                if i < len(critic_avg_gradients) and critic_avg_gradients[i] is not None:
                    param.grad = critic_avg_gradients[i]
            self.critic_optimizers_2[agent_id].step()

        # for param in self.critic.parameters():
        #     param.gard = critic_avg_gradients[k]
        #     k = k +1
        # self.critic_optimizer.step()
        # print(actor_gradients0,"changdu",len(actor_gradients0),actor_gradients0[5])

        # print(actor_avg_gradients,"平均")
        # self.v_loss1 = np.mean(v_loss)
        # print(value1,'2',v_loss,'1')
        # update actor target network and critic target network
        for agent_id in range(self.n_agents):
            if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
                # 检查 actor_targets 和 critic_targets 列表是否有足够的长度
                if agent_id < len(self.actor_targets) and agent_id < len(self.critic_targets_1) and agent_id < len(
                        self.critic_targets_2):
                    self._soft_update_target(self.actor_targets[agent_id], self.actors[agent_id])
                    self._soft_update_target(self.critic_targets_1[agent_id], self.critics_1[agent_id])
                    self._soft_update_target(self.critic_targets_2[agent_id], self.critics_2[agent_id])
                else:
                    # 如果索引超出范围，打印警告信息
                    print(
                        f"Warning: agent_id {agent_id} is out of range for actor_targets, critic_targets_1 or critic_targets_2.")

    # predict softmax action based on state

    def _softmax_action(self, state, n_agents):
        state_var = th.tensor(state, dtype=th.float32).to(self.device)
        softmax_actions = []

        # 确保为每个智能体都创建了对应的 actor 网络
        while len(self.actors) < n_agents:
            new_actor = PolicyNet(self.state_dim, self.actor_hidden_dim, self.action_dim).to(self.device)
            self.actors.append(new_actor)

        for agent_id in range(n_agents):
            #print(f"Processing agent_id: {agent_id}")

            # 添加检查，确保 agent_id 在有效范围内
            if agent_id >= len(self.actors) or agent_id >= state_var.shape[0]:
                raise ValueError(
                    f"无效的 agent_id: {agent_id}。代理数量: {len(self.actors)}，状态形状: {state_var.shape}")

            # 获取模型的输出
            outputs = self.actors[agent_id](state_var[agent_id])  # 修改这里，去掉多余的索引

            # 如果 outputs 是元组,提取第一个元素
            if isinstance(outputs, tuple):
                action_scores = outputs[0]
            else:
                action_scores = outputs

            # 确保 action_scores 是 Tensor 类型
            if not isinstance(action_scores, th.Tensor):
                raise TypeError(f"Expected a tensor for action_scores but got {type(action_scores)}.")

            # 使用 softmax 计算概率分布
            softmax_action_var = th.softmax(action_scores, dim=-1)

            if self.use_cuda:
                softmax_actions.append(softmax_action_var.cpu().detach().numpy())
            else:
                softmax_actions.append(softmax_action_var.detach().numpy())

        return softmax_actions

    # choose an action based on state with random noise added for exploration in training

    def exploration_action(self, state, n_agents):
        # 获取每个代理的动作分布
        softmax_actions = self._softmax_action(state, n_agents)

        actions = []
        for agent_id in range(n_agents):
            pi = softmax_actions[agent_id]

            # 打印调试信息
            #print(f"Agent_id: {agent_id}, pi type: {type(pi)}, pi: {pi}")

            # 确保 pi 是一个 numpy 数组，并且有正确的形状
            if not isinstance(pi, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray but got {type(pi)} for agent_id {agent_id}.")

            # 压缩维度，确保 pi 是一维数组
            pi = np.squeeze(pi)

            if pi.ndim == 1 and len(pi) > 0:
                # 检查 pi 是否为有效的概率分布
                if np.any(pi < 0) or np.any(pi > 1) or not np.isclose(np.sum(pi), 1):
                    raise ValueError(f"Invalid probability distribution for agent_id {agent_id}: {pi}")

                # 选择动作
                action = np.random.choice(np.arange(len(pi)), p=pi)
                actions.append(action)
            else:
                raise ValueError(f"Invalid action probabilities for agent_id {agent_id}.")

        return actions

    # choose an action based on state for execution
    # def action(self, state, n_agents):
    #     softmax_actions = self._softmax_action(state, n_agents)
    #     actions = []
    #     for pi in softmax_actions:
    #         actions.append(np.random.choice(np.arange(len(pi)), p=pi))
    #     return actions
    def action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []

        for pi in softmax_actions:
            pi = np.squeeze(pi)  # 确保 pi 是一维的
            if pi.ndim == 1:  # 检查 pi 是否是 1 维的
                action = np.random.choice(np.arange(len(pi)), p=pi)
            else:
                raise ValueError("概率数组 'p' 必须是一维的。")
            actions.append(action)

        return actions

    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            while len(self.critics_1) <= agent_id:
                new_critic = QValueNet(self.state_dim, self.action_dim, self.critic_hidden_dim).to(self.device)
                self.critics_1.append(new_critic)
            while len(self.critics_2) <= agent_id:
                new_critic = QValueNet(self.state_dim, self.action_dim, self.critic_hidden_dim).to(self.device)
                self.critics_2.append(new_critic)

            state_agent = state_var[:, agent_id, :]
            action_agent = action_var[:, agent_id, :]
            value1_var = self.critics_1[agent_id](state_agent, action_agent)
            value2_var = self.critics_2[agent_id](state_agent, action_agent)
            value_var = torch.min(value1_var, value2_var)

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        headway_distances = []
        merging_cost1 = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None

        if isinstance(self.test_seeds, str):
            seeds = [int(s) for s in self.test_seeds.split(',') if s]
        elif isinstance(self.test_seeds, list):
            seeds = self.test_seeds
        else:
            raise ValueError("test_seeds must be a string or a list of integers")

        if not seeds:
            raise ValueError("seeds list is empty. Please provide at least one seed value.")

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False

            if not seeds:
                print(f"Warning: seeds list is empty for evaluation episode {i}. Skipping this episode.")
                continue  # 跳过当前评估迭代

            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i % len(seeds)], num_CAV=1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i % len(seeds)], num_CAV=2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i % len(seeds)], num_CAV=3)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i % len(seeds)])

            n_agents = len(env.controlled_vehicles)
            #n_agents = len(env.controlled_vehicles)
            # rendered_frame = env.render(mode="rgb_array")
            # video_filename = os.path.join(output_dir,
            #                               "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
            #                               '.mp4')
            # Init video recording
            # if video_filename is not None:
            #     print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
            #                                                           5))
            #     video_recorder = VideoRecorder(video_filename,
            #                                    frame_size=rendered_frame.shape, fps=5)
            #     video_recorder.add_frame(rendered_frame)
            # else:
            #     video_recorder = None
            headway_diss = 0
            merging_costs = 0
            while not done:
                step += 1
                action = self.action(state, n_agents)
                state, reward, done, info = env.step(action)
                avg_speed += info["average_speed"]
                headway_distance = info["headway_distance"]
                merging_cost = info["merging_cost"]
                # print(merging_cost)
                headway_dis = np.mean(headway_distance)
                av_merging_cost = np.mean(merging_cost)
                # print(headway_dis)
                headway_diss += headway_dis
                merging_costs += av_merging_cost
                # rendered_frame = env.render(mode="rgb_array")
                # if video_recorder is not None:
                #     video_recorder.add_frame(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            # print(headway_diss,step)
            avg_speeds.append(avg_speed / step)
            headway_distances.append(headway_diss/step)
            merging_cost1.append(merging_costs/step)
        # if video_recorder is not None:
        #     video_recorder.release()
        env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, headway_distances,merging_cost1

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.tau) * t.data + self.tau * s.data)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))

            for i in range(self.n_agents):
                if f'actor_{i}_state_dict' in checkpoint:  # 检查checkpoint中是否有对应的key
                    self.actors[i].load_state_dict(checkpoint[f'actor_{i}_state_dict'])
                if f'critic_1_{i}_state_dict' in checkpoint:
                    self.critics_1[i].load_state_dict(checkpoint[f'critic_1_{i}_state_dict'])
                if f'critic_2_{i}_state_dict' in checkpoint:
                    self.critics_2[i].load_state_dict(checkpoint[f'critic_2_{i}_state_dict'])
                if train_mode:
                    if f'actor_{i}_optimizer_state_dict' in checkpoint:
                        self.actor_optimizers[i].load_state_dict(checkpoint[f'actor_{i}_optimizer_state_dict'])
                    if f'critic_1_{i}_optimizer_state_dict' in checkpoint:
                        self.critic_optimizers_1[i].load_state_dict(checkpoint[f'critic_1_{i}_optimizer_state_dict'])
                    if f'critic_2_{i}_optimizer_state_dict' in checkpoint:
                        self.critic_optimizers_2[i].load_state_dict(checkpoint[f'critic_2_{i}_optimizer_state_dict'])
                    self.actors[i].train()
                    self.critics_1[i].train()
                    self.critics_2[i].train()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        if not os.path.exists(model_dir):  # 检查目录是否存在,不存在则创建
            os.makedirs(model_dir)
        file_path = os.path.join(model_dir, 'checkpoint-{:d}.pt'.format(global_step))
        checkpoint = {'global_step': global_step}
        for i in range(self.n_agents):
            checkpoint[f'actor_{i}_state_dict'] = self.actors[i].state_dict()
            checkpoint[f'critic_1_{i}_state_dict'] = self.critics_1[i].state_dict()
            checkpoint[f'critic_2_{i}_state_dict'] = self.critics_2[i].state_dict()

            # 添加边界检查,确保`i`在`self.actor_optimizers`的有效范围内
            if i < len(self.actor_optimizers):
                checkpoint[f'actor_{i}_optimizer_state_dict'] = self.actor_optimizers[i].state_dict()

            if i < len(self.critic_optimizers_1):
                checkpoint[f'critic_1_{i}_optimizer_state_dict'] = self.critic_optimizers_1[i].state_dict()

            if i < len(self.critic_optimizers_2):
                checkpoint[f'critic_2_{i}_optimizer_state_dict'] = self.critic_optimizers_2[i].state_dict()

        th.save(checkpoint, file_path)
