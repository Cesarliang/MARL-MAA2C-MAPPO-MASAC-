import copy

import torch
import torch as th
import os, logging
import configparser
from random import randint

from torch import nn

config_dir = 'configs/configs.ini' #指定配置文件鲁路径
config = configparser.ConfigParser()#解析配置文件
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(torch_seed)

from typing import Tuple, Union
from torch.optim import Adam, RMSprop
import numpy as np
import os, logging
from copy import deepcopy
from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork, ActorCriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var, VideoRecorder
from collections import OrderedDict

class MAA2C(Agent):
    """
    An multi-agent learned with Advantage Actor-Critic
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None, #scale 奖励缩放因子调整奖励的范围，
                 actor_hidden_size=32, critic_hidden_size=32, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, training_strategy="concurrent", epsilon=1e-5, alpha=0.99,
                 traffic_density=1, test_seeds=0, state_split=False, shared_network=False, reward_type='regionalR'):
        super(MAA2C, self).__init__(env, state_dim, action_dim,
                                    memory_capacity, max_steps,
                                    reward_gamma, reward_scale, done_penalty,
                                    actor_hidden_size, critic_hidden_size, critic_loss,
                                    actor_lr, critic_lr,
                                    optimizer_type, entropy_reg,
                                    max_grad_norm, batch_size, episodes_before_train,
                                    use_cuda)#将 以上参数传递给父类构造函数：Agent


        assert training_strategy in ["concurrent", "centralized"] #并行、集中式
        #确保 training_strategy 的值在
        # 列表 ["concurrent", "centralized"] 中，即只能是这两个选项之一。确保 training_strategy 的值在列表
        # ["concurrent", "centralized"] 中，即只能是这两个选项之一。
        assert traffic_density in [1, 2, 3]
        assert reward_type in ["greedy", "regionalR", "global_R"]
        #包含了一些断言（assert）语句，用于对输入参数进行验证，以确保其满足特定的条件。
        # 如果断言条件不满足，将会触发 AssertionError 异常。
        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.test_seeds = test_seeds
        self.traffic_density = traffic_density
        self.shared_network = shared_network
        self.reward_type = reward_type
        self.alpha1 = 0.01
        "///////"
        self.n_agents = 3  # 1
        self.actors = []
        self.critics = []
        self.actor_targets = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        # maximum number of CAVs in each mode 最大CAVs车辆数，对应三个模式
        if self.traffic_density == 1:
            max_num_vehicle = 1
        elif self.traffic_density == 2:
            max_num_vehicle = 2
        elif self.traffic_density == 3:
            max_num_vehicle = 3
        for _ in range(self.n_agents):
            if not self.shared_network:#创建演员和评论家网设置络 并相应的优化器 此处使用独立的演员和评论家网络
                """separate actor and critic network"""
                self.actor= ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, state_split)

                if self.training_strategy == "concurrent":
                    self.critic = CriticNetwork(self.state_dim, self.critic_hidden_size, 1, state_split)
                elif self.training_strategy == "centralized":
                    critic_state_dim = max_num_vehicle * self.state_dim
                    self.critic = CriticNetwork(critic_state_dim, self.critic_hidden_size, 1, state_split)
                # self.aa = 1
                if optimizer_type == "adam":
                    self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
                    self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
                elif optimizer_type == "rmsprop":
                    self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)
                    self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr, eps=epsilon, alpha=alpha)
                if self.use_cuda:
                    self.actor.cuda()
                    self.critic.cuda()
            else:
                """An actor-critic network that sharing lower-layer representations but
                have distinct output layers一个共享低层表征的行为批评网络，但有不同的输出层。有不同的输出层"""
                self.policy = ActorCriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1, state_split)
                if optimizer_type == "adam":
                    self.policy_optimizers = Adam(self.policy.parameters(), lr=self.actor_lr)
                elif optimizer_type == "rmsprop":
                    self.policy_optimizers = RMSprop(self.policy.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)
                # self.aa = 2
                if self.use_cuda:
                    self.policy.cuda()
            self.actors.append(self.actor)
            self.critics.append(self.critic)
            # self.actor_targets.append(self.actor_target)
            # self.critic_targets.append(self.critic_target)
            self.actor_optimizers.append(self.actor_optimizer)
            self.critic_optimizers.append(self.critic_optimizer)
        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience代理人与环境互动以收集经验
    def explore(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.action_mask = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        policies = []
        action_masks = []
        done = True
        average_speed = 0
        average_speed = 0
        cursh_reward = 0
        self.cursh = 0
        self.n_agents = len(self.env.controlled_vehicles) #获取当前环境中受控车辆的数量，赋值给self.n_agents。
        # take n steps
        # print(self.training_strategy,self.aa)
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action_masks.append(self.action_mask)
            action, policy = self.exploration_action(self.env_state, self.action_mask)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            # self.env.render()
            self.episode_rewards[-1] += global_reward #将全局奖励累加到当前周期的总奖励
            self.epoch_steps[-1] += 1

            if self.reward_type == "greedy":
                reward = info["agents_rewards"]
            elif self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            average_speed += info["average_speed"]
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            policies.append(policy)
            final_state = next_state

            # next state and corresponding action mask
            self.env_state = next_state
            self.action_mask = info["action_mask"]
            cursh_reward = sum(info["cursh_reward"]) + cursh_reward
            self.n_steps += 1
            if done:
                self.env_state, self.action_mask = self.env.reset()
                break
        if cursh_reward != 0:
            print("发生撞车")
            self.cursh = 1
        # discount reward
        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.episode_rewards.append(0)
            self.epoch_steps.append(0)
            self.average_speed.append(0)
        else: #当前回合未结束
            self.episode_done = False
            final_action = self.action(final_state, self.n_agents, self.action_mask)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            final_r = self.value(final_state, one_hot_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale
        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_r[agent_id])
            #对于每个智能体，将对应的奖励序列 rewards[:, agent_id] 和最终的奖励 final_r[agent_id] 进行折扣处理。
        rewards = rewards.tolist()#转换未python列表形式。

        self.memory.push(states, actions, rewards, policies, action_masks)
        #将状态、动作、奖励、策略和动作掩码存储到 self.memory 中。
        # self.memory 是一个记忆回放缓冲区，用于存储Agent的经验样本，以供训练使用。
    # train on a roll out batch 分批训练
    def compute_grad_actor(
        self,
        model: torch.nn.Module,
            states_var, actions_var, action_masks_var, advantages,
        v: Union[Tuple[torch.Tensor, ...], None] = None
    ):
        # x, y = data_batch

        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        loss_1 = self.actor_loss(model,states_var, actions_var, action_masks_var, advantages)
        # loss_1 = self.criterion(logit_1, y)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        # logit_2 = model(x)
        # loss_2 = self.criterion(logit_2, y)
        loss_2 = self.actor_loss(model,states_var,actions_var,action_masks_var,advantages)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
            return grads
    def compute_grad_critic(
        self,
        model: torch.nn.Module,
            states_var, actions_var, target_values,
        v: Union[Tuple[torch.Tensor, ...], None] = None
    ):
        # x, y = data_batch

        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        loss_1 = self.critic_lossc(model,states_var, actions_var, target_values)
        # loss_1 = self.criterion(logit_1, y)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        # logit_2 = model(x)
        # loss_2 = self.criterion(logit_2, y)
        loss_2 = self.critic_lossc(model,states_var,actions_var,target_values)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
            return grads
    def actor_loss(self,actor,states_var,actions_var,action_masks_var,advantages):

        action_log_probs = actor(states_var, action_masks_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))  ##计算动作概率的损失
        action_log_probs = th.sum(action_log_probs * actions_var, 1)

        pg_loss = -th.mean(action_log_probs * advantages)
        # 计算策略梯度损失 pg_loss，即动作对数概率乘以优势函数的均值的相反数。
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        return actor_loss

    def critic_lossc(self, critic, states_var, actions_var, target_values):

        values = critic(states_var)

        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        return critic_loss
    def train(self):

        if self.n_episodes <= self.episodes_before_train: #看是到达预期训练回合
            pass
        batch = self.memory.sample(self.batch_size)
        #从经验回放缓冲区 self.memory 中采样一个批次的经验样本，并将其存储在 batch 中。
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        action_masks_var = to_tensor_var(batch.action_masks, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)#将整体状态进行重新调整形状
        #这些处理步骤将经验样本中的数据转换为模型所需的张量形式，为接下来的训练操作做准备。
        v_loss = []
        actor_gradients1 = []
        actor_gradients2 = []
        actor_gradients3 = []
        actor_gradients0 = []
        critic_gradients1 = []
        critic_gradients2 = []
        critic_gradients3 = []
        critic_gradients0 = []

        a = 5e-10
        b = 5e-11
        for agent_id in range(self.n_agents):
            self.actor_optimizers[agent_id].zero_grad()
            self.critic_optimizers[agent_id].zero_grad()
        for agent_id in range(self.n_agents):
            if not self.shared_network:
                # Original network
                actor = self.actors[agent_id]
                action_log_probs = actor(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = torch.mean(torch.distributions.Categorical(logits=action_log_probs).entropy())

                action_log_probs = torch.sum(action_log_probs * actions_var[:, agent_id, :], 1)

                if self.training_strategy == "concurrent":
                    values = self.critics[agent_id](states_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    values = self.critics(whole_states_var)

                advantages = rewards_var[:, agent_id, :] - values.detach()

                pg_loss = -torch.mean(action_log_probs * advantages)

                actor_loss = pg_loss - entropy_loss * self.entropy_reg
                actor_loss.backward()
                # print("Original actor_loss:", actor_loss)
                "///meta"
                # Deep copy of the actor
                temp_actor = copy.deepcopy(actor)
                # print("Original actor parameters:", list(actor.parameters()))
                # print("Deep copied actor parameters:", list(temp_actor.parameters()))

                # Loss on the deep copied actor
                temp_actor_loss = self.actor_loss(temp_actor, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                                  action_masks_var[:, agent_id, :], advantages)
                temp_actor_loss.backward(retain_graph=True)
                # print("Deep copied actor_loss:", temp_actor_loss)
                #
                # # Print gradients of the deep copied actor
                # print("Gradients of deep copied actor:")
                # for param in temp_actor.parameters():
                #     if param.grad is not None:
                #         print("梯度",param.grad)
                #     else:
                #         print("Gradient is None")

                # Compute gradients using autograd
                grads1 = torch.autograd.grad(temp_actor_loss, temp_actor.parameters(), allow_unused=True)
                # print("Gradients computed with autograd:")
                # for grad in grads1:
                #     if grad is not None:
                #         print(grad)
                #     else:
                #         print("Gradient is None")
                # print(grads1)
                # grads1_tar = torch.autograd.grad(temp_actor_loss, temp_actor_target.parameters(), allow_unused=True)

                for param, grad in zip(temp_actor.parameters(), grads1):
                    param.data.sub_(a * grad)
                # for param,grad in zip(temp_actor_target.parameters(), grads1_tar):
                #     param.data.sub_(0.01 * grad)
                # actor_loss_1st = self.actor_loss(temp_actor,states_var[:, agent_id, :],actions_var[:, agent_id, :],self.actor_targets[agent_id],advantages)
                actor_loss_1st = self.actor_loss(temp_actor, states_var[:, agent_id, :], actions_var[:, agent_id, :],
                                                 action_masks_var[:, agent_id, :], advantages)
                grads_1st = torch.autograd.grad(actor_loss_1st, temp_actor.parameters())
                # grads_1st_tar = torch.autograd.grad(actor_loss_1st, temp_actor_target.parameters())

                grads_2nd = self.compute_grad_actor(
                    self.actors[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
                    action_masks_var[:, agent_id, :], advantages, v=grads_1st)
                # grads_2nd = self.compute_grad_actor(
                #     self.actors[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
                #     self.actor_targets[agent_id], advantages, v=grads_1st)
                # 获取第三个数据批次 data_batch_3，并使用临时模型、第二次梯度 grads_1st 以及参数 v 和 second_order_grads 执行二阶梯度计算，得到梯度 grads_2nd。
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                        self.actors[agent_id].parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(b * grad1 - b * a * grad2)
                # self._soft_update_target(self.actor_targets[agent_id], self.actors[agent_id])
                '///'
                if agent_id == 0:

                    # print(self.actor.parameters())
                    for param in self.actors[agent_id].parameters():
                        if param.grad is not None:

                            actor_gradients0.append(param.grad.clone())

                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            actor_gradients0.append(None)
                elif agent_id == 1:
                    for param in self.actors[agent_id].parameters():
                        if param.grad is not None:

                            actor_gradients1.append(param.grad.clone())

                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            actor_gradients1.append(None)
                elif agent_id == 2:
                    for param in self.actors[agent_id].parameters():
                        if param.grad is not None:

                            actor_gradients2.append(param.grad.clone())
                            # actor_gradients.append(agent_id)
                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            actor_gradients2.append(None)
                elif agent_id == 3:
                    for param in self.actors[agent_id].parameters():
                        if param.grad is not None:

                            actor_gradients3.append(param.grad.clone())
                            # actor_gradients.append(agent_id)
                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            actor_gradients3.append(None)
                #对Actor网络的参数进行反向传播计算梯度，并根据最大梯度范数进行梯度裁剪。
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                # self.actor_optimizers[agent_id].step()
                #使用Actor网络的优化器进行参数更新。
                # update critic network
                # self.critic_optimizers[agent_id].zero_grad()
                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)
                critic_loss.backward()
                "////meta"
                # print(self.actors[agent_id].parameters())
                new_params = []

                temp_critic = copy.deepcopy(self.critics[agent_id])
                temp_critic_loss = self.critic_lossc(temp_critic, states_var[:, agent_id, :],
                                                     actions_var[:, agent_id, :],
                                                     target_values)
                temp_critic_loss.backward(retain_graph=True)
                # for param in temp_actor.parameters():
                #     # print(param)
                #     if param.grad is not None:
                #         print(param.grad)
                grads1 = torch.autograd.grad(temp_critic_loss, temp_critic.parameters(), allow_unused=True)

                for param, grad in zip(temp_critic.parameters(), grads1):
                    param.data.sub_(self.alpha1 * grad)

                critic_loss_1st = self.critic_lossc(temp_critic, states_var[:, agent_id, :],
                                                    actions_var[:, agent_id, :],
                                                    target_values)
                grads_1st = torch.autograd.grad(critic_loss_1st, temp_critic.parameters())

                grads_2nd = self.compute_grad_critic(
                    self.critics[agent_id], states_var[:, agent_id, :], actions_var[:, agent_id, :],
                    target_values, v=grads_1st)
                # 获取第三个数据批次 data_batch_3，并使用临时模型、第二次梯度 grads_1st 以及参数 v 和 second_order_grads 执行二阶梯度计算，得到梯度 grads_2nd。
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                        self.critics[agent_id].parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(b * grad1 - b * a * grad2)
                # self._soft_update_target(self.critic_targets[agent_id], self.critics[agent_id])
                '///'
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
                if agent_id == 0:
                    for param in self.critics[agent_id].parameters():
                        if param.grad is not None:

                            critic_gradients0.append(param.grad.clone())
                        else:
                            critic_gradients0.append(None)
                elif agent_id == 1:
                    for param in self.critics[agent_id].parameters():
                        if param.grad is not None:

                            critic_gradients1.append(param.grad.clone())

                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            critic_gradients1.append(None)
                elif agent_id == 2:
                    for param in self.critics[agent_id].parameters():
                        if param.grad is not None:

                            critic_gradients2.append(param.grad.clone())
                            # critic_gradients.append(agent_id)
                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            critic_gradients2.append(None)
                elif agent_id == 3:
                    for param in self.critics[agent_id].parameters():
                        if param.grad is not None:

                            critic_gradients3.append(param.grad.clone())
                            # critic_gradients.append(agent_id)
                            # print(param.grad.data,agent_id,"梯度")
                        else:
                            critic_gradients3.append(None)


                v_loss.append(critic_loss.item())

            # self.v_loss1 = np.mean(v_loss)
                #实现了Actor-Critic算法中的参数更新步骤，其中Actor网络用于学习策略，Critic网络用于学习值函数。
                # 通过梯度下降算法，优化Actor和Critic网络的参数，以最大化策略梯度损失和最小化值函数的损失。
            else:
                # update actor-critic network
                #这部分代码实现了共享网络的Actor-Critic算法中的参数更新步骤，其中Actor-Critic网络共享低层表示，
                # 但具有不同的输出层。通过梯度下降算法，同时优化Actor和Critic网络的参数，
                # 以最大化策略梯度损失、最小化值函数的损失和熵损失的加权和。
                self.policy_optimizers.zero_grad()
                action_log_probs = self.policy(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))
                action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
                values = self.policy(states_var[:, agent_id, :], out_type='v')

                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)

                advantages = rewards_var[:, agent_id, :] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages)
                loss = pg_loss - entropy_loss * self.entropy_reg + critic_loss
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizers.step()
                v_loss.append(critic_loss.item())

        actor_avg_gradients = []
        critic_avg_gradients = []

        if self.n_agents == 3:
            for i in range(10):
                actor_avg_gradients.append((actor_gradients0[i] + actor_gradients1[i] + actor_gradients2[i]) / 3)
        elif self.n_agents == 4:

            for i in range(10):
                actor_avg_gradients.append(
                    (actor_gradients0[i] + actor_gradients1[i] + actor_gradients2[i] + actor_gradients3[i]) / 4)
        elif self.n_agents == 2:
            for i in range(10):
                actor_avg_gradients.append((actor_gradients0[i] + actor_gradients1[i]) / 2)
        elif self.n_agents == 1:
            for i in range(10):
                actor_avg_gradients.append(actor_gradients0[i])

        if self.n_agents == 3:
            for i in range(10):
                critic_avg_gradients.append((critic_gradients0[i] + critic_gradients1[i] + critic_gradients2[i]) / 3)
        elif self.n_agents == 4:

            for i in range(10):
                critic_avg_gradients.append(
                    (critic_gradients0[i] + critic_gradients1[i] + critic_gradients2[i] + critic_gradients3[i]) / 4)
        elif self.n_agents == 2:

            for i in range(10):
                critic_avg_gradients.append(
                    (critic_gradients0[i] + critic_gradients1[i]) / 2)
        elif self.n_agents == 1:

            for i in range(10):
                critic_avg_gradients.append(critic_gradients0[i])
        # self.critic_optimizers[agent_id].step()
        # print(actor_avg_gradients)
        for agent_id in  range(self.n_agents):
            for i, param in enumerate(self.actors[agent_id].parameters()):
                # print(self.actors[agent_id])
                if actor_avg_gradients[i] is not None:
                    param.grad = actor_avg_gradients[i]
                    # print(i,param.grad)
            self.actor_optimizers[agent_id].step()
        # print(critic_avg_gradients,'pingjun')
        for agent_id in range(self.n_agents):
            for i, param in enumerate(self.critics[agent_id].parameters()):
                if critic_avg_gradients[i] is not None:
                    param.grad = critic_avg_gradients[i]
        # if self.max_grad_norm is not None:
        #     nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # print(i,param.grad)
            self.critic_optimizers[agent_id].step()
        self.v_loss1 = np.mean(v_loss)
        # for agent_id in range(self.n_agents):
        #     if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
        #         self._soft_update_target(self.actor_targets[agent_id], self.actors[agent_id])
        #         self._soft_update_target(self.critic_targets[agent_id], self.critics[agent_id])
#     #训练函数：根据指定的训练轮数，在每轮训练中执行以下步骤：
#
# 从经验回放缓冲区中采样批量数据。
# 将状态、动作、奖励等数据转换为张量形式。
# 根据训练策略选择相应的网络模型进行前向计算。
# 计算动作概率、策略损失和值函数损失。
# 执行反向传播和优化器更新。
    # discount roll out rewards 折扣推出的奖励
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r
    #实现了将每个时间步的奖励值按时间顺序进行折扣累加的过程，得到每个时间步的折扣奖励值。
    # 折扣因子 self.reward_gamma 控制了未来奖励的折扣程度。
    # predict softmax actions based on state
    def _softmax_action(self, state, n_agents, action_mask):
        state_var = to_tensor_var([state], self.use_cuda)
        action_mask_var = to_tensor_var([action_mask], self.use_cuda)
        softmax_action = []
        for agent_id in range(n_agents):
            if not self.shared_network:
                #调用 actors 模型（ActorNetwork）计算当前代理在给定状态和动作掩码下的 softmax 概率，
                # 并将结果存储为变量 softmax_action_var。
                softmax_action_var = th.exp(self.actors[agent_id](state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            else:
                #调用 policy 模型（ActorCriticNetwork）计算当前代理在给定状态和动作掩码下的 softmax 概率，
                softmax_action_var = th.exp(self.policy(state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # predict actions based on state, added random noise for exploration in training
    #根据状态预测行动，在训练中加入随机噪音进行探索
    def exploration_action(self, state, action_mask):
        # print(self.n_steps)
        if self.n_steps == 100:
            print('')
        softmax_actions = self._softmax_action(state, self.n_agents, action_mask)
        policy = []
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
            policy.append(pi)
        return actions, policy
    #据每个代理的动作 softmax 概率进行随机抽样，以便在探索性阶段选择动作。
    # 在每个代理的 softmax 分布中，根据概率进行抽样，以获得选择的动作，并返回选择的动作列表和对应的 softmax 概率列表。
    # predict actions based on state for execution
    def action(self, state, n_agents, action_mask):
        softmax_actions = self._softmax_action(state, n_agents, action_mask)
        actions = np.argmax(softmax_actions, axis=1) ##这部分进行了修改
        # actions = []
        # for pi in softmax_actions:
        #     actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents * self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents * self.action_dim)
        values = [0] * self.n_agents

        for agent_id in range(self.n_agents):
            if not self.shared_network:#如果网络结构是分离的
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.critics[agent_id](state_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    value_var = self.critics(whole_state_var)
            else: #网络结构未共享的
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.policy(state_var[:, agent_id, :], out_type='v')
                elif self.training_strategy == "centralized":
                    value_var = self.policy(whole_state_var, out_type='v')

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent #评价学习过的智能体
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        #用于评估模型在环境中的性能。循环执行一定数量的评估周期，每个周期中进行动作选择、环境交互和记录结果。支持视频录制功能。
        rewards = []
        infos = []
        avg_speeds = []
        headway_distances = []
        merging_cost1 = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=3)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])
            n_agents = 3
            # rendered_frame = env.render(mode="rgb_array")
            # video_filename = os.path.join(output_dir,
            #                               "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
            #                               '.mp4')#每指定回合保存视屏文件件
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
                action = self.action(state, n_agents, action_mask)
                state, reward, done, info = env.step(action)
                action_mask = info["action_mask"]
                avg_speed += info["average_speed"]
                # rendered_frame = env.render(mode="rgb_array")
                # if video_recorder is not None:
                #     video_recorder.add_frame(rendered_frame)
                headway_distance = info["headway_distance"]
                merging_cost = info["merging_cost"]
                headway_dis = np.mean(headway_distance)
                av_merging_cost = np.mean(merging_cost)
                headway_diss += headway_dis
                merging_costs += av_merging_cost
                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            headway_distances.append(headway_diss / step)
            merging_cost1.append(merging_costs / step)
        # if video_recorder is not None:
        #     video_recorder.release()
        env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds, headway_distances,merging_cost1

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
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            if not self.shared_network:
                self.actors.load_state_dict(checkpoint['model_state_dict'])
                if train_mode:#根据train_mode确定是否加载优化器状态和设置模型为训练或评估模式。
                    self.actor_optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.actors.train()
                else:
                    self.actors.eval()
            else:
                self.policy.load_state_dict(checkpoint['model_state_dict'])
                if train_mode:
                    self.policy_optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.policy.train()
                else:
                    self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        if not self.shared_network:
            th.save({'global_step': global_step,#将模型的状态字典、优化器的状态字典和全局步数一起保存到file_path指定的文件中。
                     'model_state_dict': self.actors.state_dict(),
                     'optimizer_state_dict': self.actor_optimizers.state_dict()},
                    file_path)
        else:
            th.save({'global_step': global_step,
                     'model_state_dict': self.policy.state_dict(),
                     'optimizer_state_dict': self.policy_optimizers.state_dict()},
                    file_path)