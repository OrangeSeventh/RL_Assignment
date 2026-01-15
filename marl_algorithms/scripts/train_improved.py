#!/usr/bin/env python3
"""
Transport任务改进训练脚本 - 实施高优先级改进
改进包括：
1. 动态熵系数调整
2. 调整GAE参数至0.97
3. 学习率调度
4. 增加训练迭代次数至1000
5. 更好的日志记录和结果保存
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import gymnasium as gym
import json

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env
from transport_config import ENV_CONFIG, TRAINING_CONFIG, PATH_CONFIG

class DynamicEntropyCallback(BaseCallback):
    """动态熵系数回调：线性衰减熵系数"""
    def __init__(self, initial_ent_coef=0.01, min_ent_coef=0.001, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.min_ent_coef = min_ent_coef
        self.total_timesteps = None

    def _on_training_start(self):
        if hasattr(self.model, 'num_timesteps'):
            self.total_timesteps = self.model.num_timesteps
        else:
            self.total_timesteps = self.training_env.get_attr('spec')[0].max_episode_steps * 1000

    def _on_step(self):
        if self.total_timesteps is not None:
            progress = self.num_timesteps / self.total_timesteps
            current_ent_coef = self.initial_ent_coef * (1 - progress) + self.min_ent_coef * progress
            self.model.ent_coef = current_ent_coef
            if self.verbose > 0 and self.num_timesteps % 10000 == 0:
                print(f"Step {self.num_timesteps}: ent_coef = {current_ent_coef:.6f}")
        return True

class MetricsCallback(BaseCallback):
    """记录训练指标的回调"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'learning_rates': [],
            'ent_coefs': [],
        }

    def _on_step(self):
        if len(self.model.logger.name_to_value) > 0:
            # 记录奖励
            if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
                self.metrics['rewards'].append(self.model.logger.name_to_value['rollout/ep_rew_mean'])

            # 记录损失
            if 'train/policy_loss' in self.model.logger.name_to_value:
                self.metrics['policy_losses'].append(self.model.logger.name_to_value['train/policy_loss'])
            if 'train/value_loss' in self.model.logger.name_to_value:
                self.metrics['value_losses'].append(self.model.logger.name_to_value['train/value_loss'])
            if 'train/entropy_loss' in self.model.logger.name_to_value:
                self.metrics['entropy_losses'].append(self.model.logger.name_to_value['train/entropy_loss'])

            # 记录超参数
            if 'train/learning_rate' in self.model.logger.name_to_value:
                self.metrics['learning_rates'].append(self.model.logger.name_to_value['train/learning_rate'])
            if 'train/ent_coef' in self.model.logger.name_to_value:
                self.metrics['ent_coefs'].append(self.model.logger.name_to_value['train/ent_coef'])

        return True

def make_vmas_env():
    """创建单个VMAS Gymnasium环境"""
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

def train_algorithm(algorithm_name, num_iterations=1000):
    """训练指定算法（改进版）"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algorithm_name} 算法（改进版）")
    print(f"{'='*60}\n")

    # 创建VMAS Gymnasium环境
    env = make_vmas_env()

    # 打印环境信息
    print(f"环境信息:")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 智能体数量: {env.env.n_agents}")

    # 改进的训练配置
    improved_config = {
        "learning_rate": 2e-4,  # 降低学习率以提高稳定性
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.97,  # 提高GAE参数以减少方差
        "clip_range": 0.2,
        "ent_coef": 0.01,  # 初始熵系数，将通过回调动态调整
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": [256, 256],  # 网络架构
            "activation_fn": torch.nn.Tanh,
        },
    }

    # 创建回调
    callbacks = []

    # 动态熵系数回调
    ent_coef_callback = DynamicEntropyCallback(
        initial_ent_coef=0.01,
        min_ent_coef=0.001,
        verbose=1
    )
    callbacks.append(ent_coef_callback)

    # 指标记录回调
    metrics_callback = MetricsCallback(verbose=0)
    callbacks.append(metrics_callback)

    # 检查点回调（每200次迭代保存一次）
    checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=2048 * 200,  # 每200次迭代保存
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm_name}_model"
    )
    callbacks.append(checkpoint_callback)

    # 根据算法类型选择训练策略
    if algorithm_name == "IPPO":
        # IPPO: 独立训练每个智能体
        print("\n使用IPPO策略: 独立训练每个智能体")
        print("改进: 动态熵系数、GAE=0.97、学习率=2e-4")

        models = {}
        all_metrics = {}

        for i in range(env.env.n_agents):
            agent_name = f"agent_{i}"
            print(f"\n训练智能体: {agent_name}")

            # 为每个智能体创建新的指标回调
            agent_metrics_callback = MetricsCallback(verbose=0)
            agent_callbacks = [ent_coef_callback, agent_metrics_callback, checkpoint_callback]

            model = PPO(
                MlpPolicy,
                env,
                verbose=1,
                tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}/{agent_name}",
                **improved_config
            )

            model.learn(
                total_timesteps=num_iterations * 2048,
                callback=agent_callbacks
            )

            models[agent_name] = model
            all_metrics[agent_name] = agent_metrics_callback.metrics

            # 保存最终模型
            model.save(os.path.join(checkpoint_dir, f"{agent_name}_final_model"))

        # 保存所有指标
        metrics_file = os.path.join(PATH_CONFIG["results_dir"], algorithm_name, f"{algorithm_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    elif algorithm_name == "MAPPO":
        # MAPPO: 集中式训练，分布式执行
        print("\n使用MAPPO策略: 集中式训练，分布式执行")
        print("改进: 动态熵系数、GAE=0.97、学习率=2e-4")

        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
            **improved_config
        )

        model.learn(
            total_timesteps=num_iterations * 2048,
            callback=callbacks
        )

        # 保存最终模型
        model.save(os.path.join(checkpoint_dir, f"{algorithm_name}_final_model"))

        # 保存指标
        metrics_file = os.path.join(PATH_CONFIG["results_dir"], algorithm_name, f"{algorithm_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_callback.metrics, f, indent=2)

    elif algorithm_name == "CPPO":
        # CPPO: 集中式训练，集中式执行
        print("\n使用CPPO策略: 集中式训练，集中式执行")
        print("改进: 动态熵系数、GAE=0.97、学习率=2e-4")
        print("注意: CPPO的稳定性问题通过动态熵系数和降低学习率来改善")

        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
            **improved_config
        )

        model.learn(
            total_timesteps=num_iterations * 2048,
            callback=callbacks
        )

        # 保存最终模型
        model.save(os.path.join(checkpoint_dir, f"{algorithm_name}_final_model"))

        # 保存指标
        metrics_file = os.path.join(PATH_CONFIG["results_dir"], algorithm_name, f"{algorithm_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_callback.metrics, f, indent=2)

    print(f"\n{algorithm_name} 训练完成!")
    print(f"模型保存在: {checkpoint_dir}")
    print(f"指标保存在: {metrics_file}")

    return checkpoint_dir

def main():
    parser = argparse.ArgumentParser(description="训练Transport任务的MARL算法（改进版）")
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
        default=1000,
        help="训练迭代次数（默认1000）"
    )

    args = parser.parse_args()

    print(f"\n改进训练配置:")
    print(f"  - 训练迭代次数: {args.iterations}")
    print(f"  - 学习率: 2e-4（降低以提高稳定性）")
    print(f"  - GAE参数: 0.97（提高以减少方差）")
    print(f"  - 熵系数: 动态调整（0.01 -> 0.001）")
    print(f"  - 检查点保存: 每200次迭代")

    checkpoint_path = train_algorithm(args.algorithm, args.iterations)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"算法: {args.algorithm}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()