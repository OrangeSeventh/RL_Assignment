#!/usr/bin/env python3
"""
快速测试脚本：快速训练少量迭代以验证改进效果
用于快速验证改进是否有效
"""

import sys
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env
from transport_config import ENV_CONFIG, TRAINING_CONFIG, PATH_CONFIG

class SimpleMetricsCallback(BaseCallback):
    """简单的指标记录回调"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self):
        if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
            self.rewards.append(self.model.logger.name_to_value['rollout/ep_rew_mean'])
        return True

def make_vmas_env():
    """创建VMAS Gymnasium环境"""
    return make_env(
        scenario=ENV_CONFIG["scenario"],
        num_envs=1,
        device=ENV_CONFIG["device"],
        continuous_actions=ENV_CONFIG["continuous_actions"],
        max_steps=ENV_CONFIG["max_steps"],
        wrapper="gymnasium",
        dict_spaces=False,
        terminated_truncated=True,
        n_agents=ENV_CONFIG["n_agents"],
        n_packages=ENV_CONFIG["n_packages"],
        package_width=ENV_CONFIG["package_width"],
        package_length=ENV_CONFIG["package_length"],
        package_mass=ENV_CONFIG["package_mass"],
    )

def quick_test(algorithm_name, num_iterations=50):
    """快速测试算法（仅训练50次迭代）"""
    print(f"\n{'='*60}")
    print(f"快速测试 {algorithm_name} 算法")
    print(f"训练迭代次数: {num_iterations}")
    print(f"{'='*60}\n")

    # 创建环境
    env = make_vmas_env()

    # 原始配置
    original_config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # 改进配置
    improved_config = {
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    results = {}

    # 测试原始配置
    print(f"\n测试原始配置...")
    print(f"  学习率: {original_config['learning_rate']}")
    print(f"  GAE参数: {original_config['gae_lambda']}")
    print(f"  熵系数: {original_config['ent_coef']}")

    metrics_original = SimpleMetricsCallback()
    model_original = PPO(
        MlpPolicy,
        env,
        verbose=0,
        **original_config
    )

    model_original.learn(
        total_timesteps=num_iterations * 2048,
        callback=metrics_original
    )

    results['original'] = {
        'rewards': metrics_original.rewards,
        'final_reward': metrics_original.rewards[-1] if metrics_original.rewards else 0,
        'max_reward': max(metrics_original.rewards) if metrics_original.rewards else 0,
    }

    print(f"  最终奖励: {results['original']['final_reward']:.4f}")
    print(f"  最高奖励: {results['original']['max_reward']:.4f}")

    # 测试改进配置
    print(f"\n测试改进配置...")
    print(f"  学习率: {improved_config['learning_rate']}")
    print(f"  GAE参数: {improved_config['gae_lambda']}")
    print(f"  熵系数: {improved_config['ent_coef']}")

    metrics_improved = SimpleMetricsCallback()
    model_improved = PPO(
        MlpPolicy,
        env,
        verbose=0,
        **improved_config
    )

    model_improved.learn(
        total_timesteps=num_iterations * 2048,
        callback=metrics_improved
    )

    results['improved'] = {
        'rewards': metrics_improved.rewards,
        'final_reward': metrics_improved.rewards[-1] if metrics_improved.rewards else 0,
        'max_reward': max(metrics_improved.rewards) if metrics_improved.rewards else 0,
    }

    print(f"  最终奖励: {results['improved']['final_reward']:.4f}")
    print(f"  最高奖励: {results['improved']['max_reward']:.4f}")

    # 对比结果
    print(f"\n{'='*60}")
    print(f"快速测试结果对比")
    print(f"{'='*60}\n")

    print(f"原始配置:")
    print(f"  最终奖励: {results['original']['final_reward']:.4f}")
    print(f"  最高奖励: {results['original']['max_reward']:.4f}")

    print(f"\n改进配置:")
    print(f"  最终奖励: {results['improved']['final_reward']:.4f}")
    print(f"  最高奖励: {results['improved']['max_reward']:.4f}")

    final_improvement = results['improved']['final_reward'] - results['original']['final_reward']
    max_improvement = results['improved']['max_reward'] - results['original']['max_reward']

    print(f"\n改进幅度:")
    print(f"  最终奖励: {final_improvement:+.4f}")
    print(f"  最高奖励: {max_improvement:+.4f}")

    # 判断改进是否有效
    if final_improvement > 0 or max_improvement > 0:
        print(f"\n✅ 改进有效！")
    else:
        print(f"\n⚠️ 改进效果不明显，可能需要调整参数")

    return results

def main():
    parser = argparse.ArgumentParser(description="快速测试改进效果")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["CPPO", "MAPPO", "IPPO"],
        default="MAPPO",
        help="选择算法进行测试"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="快速测试的迭代次数（默认50）"
    )

    args = parser.parse_args()

    results = quick_test(args.algorithm, args.iterations)

if __name__ == "__main__":
    main()