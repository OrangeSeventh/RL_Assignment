#!/usr/bin/env python3
"""
简化版快速测试脚本 - 直接使用VMAS环境
用于快速验证改进效果
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from vmas import make_env
import torch.nn as nn
import torch.optim as optim

def make_vmas_env():
    """创建VMAS环境（不使用包装器）"""
    return make_env(
        scenario="transport",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        max_steps=500,
        wrapper=None,  # 不使用包装器
        dict_spaces=False,
        n_agents=4,
        n_packages=1,
        package_width=0.15,
        package_length=0.15,
        package_mass=50,
    )

class SimplePolicy(nn.Module):
    """简单的策略网络"""
    def __init__(self, obs_dim=11, action_dim=2, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.network(obs)

def simple_test(iterations=50):
    """简化版测试：使用随机策略作为baseline"""
    print(f"\n{'='*60}")
    print(f"简化版快速测试")
    print(f"测试迭代次数: {iterations}")
    print(f"{'='*60}\n")

    # 创建环境
    env = make_vmas_env()

    print(f"环境信息:")
    print(f"  - 智能体数量: {len(env.agents)}")
    print(f"  - 观测维度: {env.observation_space[0].shape}")
    print(f"  - 动作维度: {env.action_space[0].shape}")

    # 使用随机策略进行测试
    rewards = []

    for iteration in range(iterations):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # 使用随机动作
            actions = [env.get_random_action(agent) for agent in env.agents]
            obs, rews, dones, info = env.step(actions)

            # 计算总奖励
            episode_reward = sum(rews)
            total_reward += episode_reward
            steps += 1

            # 检查是否结束
            done = all(dones)

        rewards.append(total_reward)

        if (iteration + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"迭代 {iteration + 1}/{iterations}: 平均奖励 = {avg_reward:.4f}")

    # 统计结果
    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}\n")

    # 将tensor转换为numpy
    rewards_np = [float(r) if torch.is_tensor(r) else r for r in rewards]

    final_avg = np.mean(rewards_np[-10:])
    max_reward = np.max(rewards_np)
    min_reward = np.min(rewards_np)
    std_reward = np.std(rewards_np)

    print(f"最终平均奖励（最后10次）: {final_avg:.4f}")
    print(f"最高奖励: {max_reward:.4f}")
    print(f"最低奖励: {min_reward:.4f}")
    print(f"奖励标准差: {std_reward:.4f}")
    print(f"所有奖励: {[f'{r:.4f}' for r in rewards_np]}")

    return {
        'rewards': rewards_np,
        'final_avg': final_avg,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'std_reward': std_reward,
    }

def main():
    parser = argparse.ArgumentParser(description="简化版快速测试")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="测试迭代次数（默认50）"
    )

    args = parser.parse_args()

    results = simple_test(args.iterations)

    print(f"\n测试完成！")

if __name__ == "__main__":
    main()