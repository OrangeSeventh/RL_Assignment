#!/usr/bin/env python3
"""
Transport任务训练脚本 - 使用VMAS原生环境（完整实现）
支持CPPO、MAPPO、IPPO三种算法的训练
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env

# 导入配置
from transport_config import (
    ENV_CONFIG,
    TRAINING_CONFIG,
    PATH_CONFIG,
)

# 导入观测归一化
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms')
from normalization import NormalizeObservation

# ==================== 神经网络定义 ====================

class ActorCritic(nn.Module):
    """Actor-Critic网络（共享参数）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor网络（策略网络）
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出范围[-1, 1]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        shared_features = self.shared(obs)

        # Actor输出动作均值和标准差
        action_mean = self.actor_mean(shared_features)  # 输出范围[-1, 1]
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
        action_dist = Normal(action_mean, action_std)

        # Critic输出价值
        value = self.critic(shared_features).squeeze(-1)

        return action_dist, value

    def get_action(self, obs, deterministic=False):
        action_dist, value = self.forward(obs)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        # 裁剪动作到[-1, 1]范围
        action = torch.clamp(action, -1.0, 1.0)
        return action, action_dist.log_prob(action).sum(dim=-1), value

    def evaluate_actions(self, obs, actions):
        action_dist, value = self.forward(obs)
        log_prob = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1).mean()
        return log_prob, value, entropy


# ==================== PPO算法实现 ====================

class PPO:
    """PPO算法实现"""
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, lambda_=0.95,
                 clip_param=0.2, entropy_coeff=0.01, vf_loss_coeff=0.5, max_grad_norm=0.5):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_param = clip_param
        self.entropy_coeff = entropy_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计（GAE）"""
        gae = 0
        returns = []
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            # 转换dones为float类型
            done = dones[t].float() if isinstance(dones[t], torch.Tensor) else float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - done) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - done) * gae
            returns.insert(0, gae + values[t])

        return returns

    def update(self, obs_batch, action_batch, old_log_prob_batch, returns_batch,
               advantage_batch, old_value_batch, epochs=10, batch_size=64):
        """更新网络"""
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        for _ in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(obs_batch))

            for start in range(0, len(obs_batch), batch_size):
                end = start + batch_size
                idx = indices[start:end]

                # 获取batch数据（detach以避免梯度累积）
                obs = torch.stack([obs_batch[i].detach() for i in idx])
                actions = torch.stack([action_batch[i].detach() for i in idx])
                old_log_probs = torch.stack([old_log_prob_batch[i].detach() for i in idx])
                returns = torch.stack([returns_batch[i].detach() for i in idx])
                advantages = torch.stack([advantage_batch[i].detach() for i in idx])
                old_values = torch.stack([old_value_batch[i].detach() for i in idx])

                # 计算新的log_prob和value
                log_probs, values, entropy = self.actor_critic.evaluate_actions(obs, actions)

                # 计算ratio
                ratio = torch.exp(log_probs - old_log_probs)

                # 计算policy loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss_batch = -torch.min(surr1, surr2).mean()

                # 计算value loss（使用clipped value）
                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                value_loss1 = (values - returns) ** 2
                value_loss2 = (value_pred_clipped - returns) ** 2
                value_loss_batch = torch.max(value_loss1, value_loss2).mean()

                # 计算总loss
                loss = policy_loss_batch + self.vf_loss_coeff * value_loss_batch - self.entropy_coeff * entropy

                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy_loss += entropy.item()

        num_updates = (len(obs_batch) + batch_size - 1) // batch_size * epochs
        return {
            'total_loss': total_loss / num_updates,
            'policy_loss': policy_loss / num_updates,
            'value_loss': value_loss / num_updates,
            'entropy': entropy_loss / num_updates,
        }


# ==================== 训练函数 ====================

def collect_trajectories(env, policies, num_steps, num_envs, obs_normalizer=None, update_stats=True):
    """收集轨迹数据（向量化）"""
    obs = env.reset()

    # 存储每个智能体的数据
    trajectories = {
        agent.name: {
            'obs': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
        }
        for agent in env.agents
    }

    for step in range(num_steps):
        actions = []
        log_probs = []
        values = []

        # 获取每个智能体的动作
        for i, agent in enumerate(env.agents):
            policy = policies[agent.name]
            obs_tensor = obs[i] if isinstance(obs[i], torch.Tensor) else torch.tensor(obs[i], dtype=torch.float32)

            # 观测归一化
            if obs_normalizer is not None:
                obs_tensor = obs_normalizer.normalize(obs_tensor, update_stats=update_stats)

            action, log_prob, value = policy.actor_critic.get_action(obs_tensor)

            actions.append(action)
            log_probs.append(log_prob.detach())
            values.append(value.detach())

            # 存储数据（detach以避免梯度累积）
            trajectories[agent.name]['obs'].append(obs_tensor.detach())
            trajectories[agent.name]['log_probs'].append(log_prob)
            trajectories[agent.name]['values'].append(value)

        # 执行动作
        obs, rews, dones, info = env.step(actions)

        # 存储奖励和done
        for i, agent in enumerate(env.agents):
            trajectories[agent.name]['actions'].append(actions[i])
            trajectories[agent.name]['rewards'].append(rews[i])
            trajectories[agent.name]['dones'].append(dones[i])

    return trajectories


def train_algorithm(algorithm_name, num_iterations=300, num_steps_per_iter=200, num_envs=32, use_normalization=False):
    """训练指定算法"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algorithm_name} 算法")
    if use_normalization:
        print(f"使用观测归一化")
    print(f"{'='*60}\n")

    # 创建VMAS环境
    env = make_env(
        scenario=ENV_CONFIG["scenario"],
        num_envs=num_envs,
        device=ENV_CONFIG["device"],
        continuous_actions=ENV_CONFIG["continuous_actions"],
        max_steps=ENV_CONFIG["max_steps"],
        wrapper=None,
        dict_spaces=False,
        n_agents=ENV_CONFIG["n_agents"],
        n_packages=ENV_CONFIG["n_packages"],
        package_width=ENV_CONFIG["package_width"],
        package_length=ENV_CONFIG["package_length"],
        package_mass=ENV_CONFIG["package_mass"],
    )

    # 创建观测归一化器
    obs_normalizer = None
    if use_normalization:
        obs_dim = env.observation_space[0].shape[0]
        # 预收集20步数据来初始化归一化统计量
        obs_normalizer = NormalizeObservation(obs_dim, pre_collect_steps=20)
        print(f"已创建观测归一化器，观测维度: {obs_dim}")
        print(f"将预收集20步数据来初始化归一化统计量...")

    print(f"环境信息:")
    print(f"  - 智能体数量: {len(env.agents)}")
    print(f"  - 并行环境数: {num_envs}")
    print(f"  - 观测维度: {env.observation_space[0].shape}")
    print(f"  - 动作维度: {env.action_space[0].shape}")

    # 获取观测和动作维度
    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]

    # 根据算法类型创建策略
    policies = {}

    if algorithm_name == "IPPO":
        # IPPO: 每个智能体独立的策略
        print("\n使用IPPO策略: 每个智能体独立的策略")
        for agent in env.agents:
            policies[agent.name] = PPO(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=TRAINING_CONFIG["lr"],
                gamma=TRAINING_CONFIG["gamma"],
                lambda_=TRAINING_CONFIG["lambda_"],
                clip_param=TRAINING_CONFIG["clip_param"],
                entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
                vf_loss_coeff=TRAINING_CONFIG["vf_loss_coeff"],
            )

    elif algorithm_name == "MAPPO":
        # MAPPO: 集中式训练，分布式执行
        print("\n使用MAPPO策略: 集中式训练，分布式执行")
        # 所有智能体共享同一个策略
        shared_policy = PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lr=TRAINING_CONFIG["lr"],
            gamma=TRAINING_CONFIG["gamma"],
            lambda_=TRAINING_CONFIG["lambda_"],
            clip_param=TRAINING_CONFIG["clip_param"],
            entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
            vf_loss_coeff=TRAINING_CONFIG["vf_loss_coeff"],
        )
        for agent in env.agents:
            policies[agent.name] = shared_policy

    elif algorithm_name == "CPPO":
        # CPPO: 集中式训练，集中式执行
        print("\n使用CPPO策略: 集中式训练，集中式执行")
        # 与MAPPO类似，但使用全局观测（这里简化为共享策略）
        shared_policy = PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lr=TRAINING_CONFIG["lr"],
            gamma=TRAINING_CONFIG["gamma"],
            lambda_=TRAINING_CONFIG["lambda_"],
            clip_param=TRAINING_CONFIG["clip_param"],
            entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
            vf_loss_coeff=TRAINING_CONFIG["vf_loss_coeff"],
        )
        for agent in env.agents:
            policies[agent.name] = shared_policy

    # 训练循环
    all_rewards = []
    all_metrics = []
    start_time = time.time()

    # 如果使用归一化，先预收集数据
    if use_normalization and not obs_normalizer.is_pre_collection_done():
        print(f"\n>>> 预收集数据初始化归一化统计量...")
        obs = env.reset()
        for step in range(obs_normalizer.pre_collect_steps):
            # 使用随机动作收集数据（裁剪到[-1, 1]范围）
            actions = []
            for agent in env.agents:
                action = torch.randn(num_envs, env.action_space[0].shape[0], device=env.device)
                action = torch.clamp(action, -1.0, 1.0)  # 裁剪到有效范围
                actions.append(action)

            obs, rews, dones, info = env.step(actions)

            # 收集所有智能体的观测
            for i, agent in enumerate(env.agents):
                obs_tensor = obs[i] if isinstance(obs[i], torch.Tensor) else torch.tensor(obs[i], dtype=torch.float32)
                obs_normalizer.pre_collect(obs_tensor)

        obs_normalizer.finalize_pre_collection()

    for iteration in range(num_iterations):
        iteration_start = time.time()

        # 收集轨迹
        trajectories = collect_trajectories(env, policies, num_steps_per_iter, num_envs, obs_normalizer, update_stats=True)

        # 更新策略
        metrics = {}
        for agent_name in policies.keys():
            traj = trajectories[agent_name]

            # 计算returns和advantages
            rewards = traj['rewards']
            values = traj['values']
            dones = traj['dones']

            # 计算returns（使用GAE）
            next_value = torch.zeros(num_envs)
            returns = policies[agent_name].compute_gae(rewards, values, dones, next_value)

            # 计算advantages
            advantages = [r - v for r, v in zip(returns, values)]

            # 标准化advantages
            advantages = torch.stack(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 更新策略
            policy_metrics = policies[agent_name].update(
                traj['obs'],
                traj['actions'],
                traj['log_probs'],
                returns,
                advantages,
                values,
                epochs=TRAINING_CONFIG["ppo_epochs"],
                batch_size=TRAINING_CONFIG["batch_size"],
            )

            metrics[agent_name] = policy_metrics

        # 计算平均奖励
        total_reward = sum([sum(traj['rewards']) for traj in trajectories.values()])
        avg_reward = total_reward.mean() if isinstance(total_reward, torch.Tensor) else total_reward / len(trajectories)
        avg_reward = avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward
        all_rewards.append(avg_reward)

        # 打印进度
        iteration_time = time.time() - iteration_start
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Time: {iteration_time:.2f}s")
            for agent_name, agent_metrics in metrics.items():
                print(f"  {agent_name}: Loss={agent_metrics['total_loss']:.4f}, "
                      f"Policy={agent_metrics['policy_loss']:.4f}, "
                      f"Value={agent_metrics['value_loss']:.4f}, "
                      f"Entropy={agent_metrics['entropy']:.4f}")

        # 保存训练指标
        all_metrics.append({
            'iteration': iteration,
            'avg_reward': avg_reward,
            'metrics': metrics,
        })

    total_time = time.time() - start_time

    # 保存结果
    results = {
        'algorithm': algorithm_name,
        'num_iterations': num_iterations,
        'num_steps_per_iter': num_steps_per_iter,
        'num_envs': num_envs,
        'total_time': total_time,
        'rewards': all_rewards,
        'metrics': all_metrics,
    }

    # 保存到文件
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = os.path.join(
        PATH_CONFIG["results_dir"],
        algorithm_name,
        f"{algorithm_name}_transport_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if algorithm_name == "IPPO":
        for agent_name, policy in policies.items():
            torch.save(policy.actor_critic.state_dict(),
                      os.path.join(checkpoint_dir, f"{agent_name}_model.pth"))
    else:
        # MAPPO和CPPO共享策略
        torch.save(policies[env.agents[0].name].actor_critic.state_dict(),
                  os.path.join(checkpoint_dir, f"{algorithm_name}_model.pth"))

    # 绘制训练曲线
    plot_training_curve(all_rewards, algorithm_name, results_file)

    print(f"\n{algorithm_name} 训练完成!")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"结果保存在: {results_file}")
    print(f"模型保存在: {checkpoint_dir}")
    print(f"最终平均奖励: {all_rewards[-1]:.4f}")
    print(f"最高平均奖励: {max(all_rewards):.4f}")

    return results


def plot_training_curve(rewards, algorithm_name, results_file):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label=f'{algorithm_name} Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title(f'{algorithm_name} Training Progress')
    plt.legend()
    plt.grid(True)

    # 保存图片
    plot_file = results_file.replace('.json', '.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线保存在: {plot_file}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="训练Transport任务的MARL算法")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["CPPO", "MAPPO", "IPPO"],
        required=True,
        help="选择算法: CPPO, MAPPO, IPPO"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="训练迭代次数（默认300次，约2-3分钟）"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="每次迭代的步数"
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=32,
        help="并行环境数量"
    )
    parser.add_argument(
        "--normalization",
        action="store_true",
        help="是否使用观测归一化"
    )

    args = parser.parse_args()

    print(f"训练配置:")
    print(f"  - 算法: {args.algorithm}")
    print(f"  - 迭代次数: {args.iterations}")
    print(f"  - 每次迭代步数: {args.steps}")
    print(f"  - 并行环境数: {args.envs}")
    print(f"  - 观测归一化: {args.normalization}")
    print(f"  - 预计训练时间: ~{args.iterations * args.steps / 377:.1f}秒")

    results = train_algorithm(args.algorithm, args.iterations, args.steps, args.envs, args.normalization)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"算法: {args.algorithm}")
    print(f"最终平均奖励: {results['rewards'][-1]:.4f}")
    print(f"最高平均奖励: {max(results['rewards']):.4f}")
    print(f"总训练时间: {results['total_time']:.2f}秒")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()