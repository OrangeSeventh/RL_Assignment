#!/usr/bin/env python3
"""
Transport任务训练脚本 - 使用VMAS原生Gymnasium包装器 + Stable-Baselines3
支持CPPO、MAPPO、IPPO三种算法的训练
"""

import sys
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gymnasium as gym

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

def make_vmas_env():
    """创建单个VMAS Gymnasium环境"""
    return make_env(
        scenario=ENV_CONFIG["scenario"],
        num_envs=1,
        device=ENV_CONFIG["device"],
        continuous_actions=ENV_CONFIG["continuous_actions"],
        max_steps=ENV_CONFIG["max_steps"],
        wrapper="gymnasium",  # 使用VMAS的Gymnasium包装器
        dict_spaces=False,
        terminated_truncated=True,  # 必须设置为True
        n_agents=ENV_CONFIG["n_agents"],
        n_packages=ENV_CONFIG["n_packages"],
        package_width=ENV_CONFIG["package_width"],
        package_length=ENV_CONFIG["package_length"],
        package_mass=ENV_CONFIG["package_mass"],
    )

def train_algorithm(algorithm_name, num_iterations=1000):
    """训练指定算法"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algorithm_name} 算法")
    print(f"{'='*60}\n")

    # 创建VMAS Gymnasium环境
    env = make_vmas_env()

    # 打印环境信息
    print(f"环境信息:")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 智能体数量: {env.env.n_agents}")

    # 根据算法类型选择训练策略
    if algorithm_name == "IPPO":
        # IPPO: 独立训练每个智能体
        print("\n使用IPPO策略: 独立训练每个智能体")
        models = {}
        for i in range(env.env.n_agents):
            agent_name = f"agent_{i}"
            print(f"训练智能体: {agent_name}")
            model = PPO(
                MlpPolicy,
                env,
                verbose=1,
                learning_rate=TRAINING_CONFIG["lr"],
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=TRAINING_CONFIG["gamma"],
                gae_lambda=TRAINING_CONFIG["lambda_"],
                clip_range=TRAINING_CONFIG["clip_param"],
                ent_coef=TRAINING_CONFIG["entropy_coeff"],
                vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
                max_grad_norm=0.5,
                tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}/{agent_name}",
            )
            model.learn(total_timesteps=num_iterations * 2048)
            models[agent_name] = model

            # 保存模型
            checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save(os.path.join(checkpoint_dir, f"{agent_name}_model"))

    elif algorithm_name == "MAPPO":
        # MAPPO: 集中式训练，分布式执行
        print("\n使用MAPPO策略: 集中式训练，分布式执行")
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=TRAINING_CONFIG["lr"],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["lambda_"],
            clip_range=TRAINING_CONFIG["clip_param"],
            ent_coef=TRAINING_CONFIG["entropy_coeff"],
            vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
            max_grad_norm=0.5,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
        )
        model.learn(total_timesteps=num_iterations * 2048)

        # 保存模型
        checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(os.path.join(checkpoint_dir, "shared_model"))

    elif algorithm_name == "CPPO":
        # CPPO: 集中式训练，集中式执行
        print("\n使用CPPO策略: 集中式训练，集中式执行")
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=TRAINING_CONFIG["lr"],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["lambda_"],
            clip_range=TRAINING_CONFIG["clip_param"],
            ent_coef=TRAINING_CONFIG["entropy_coeff"],
            vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
            max_grad_norm=0.5,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
        )
        model.learn(total_timesteps=num_iterations * 2048)

        # 保存模型
        checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(os.path.join(checkpoint_dir, "centralized_model"))

    print(f"\n{algorithm_name} 训练完成!")
    print(f"模型保存在: {PATH_CONFIG['checkpoints_dir']}/{algorithm_name}")

    return PATH_CONFIG["checkpoints_dir"]

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
        default=1000,
        help="训练迭代次数"
    )

    args = parser.parse_args()

    checkpoint_path = train_algorithm(args.algorithm, args.iterations)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"算法: {args.algorithm}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()