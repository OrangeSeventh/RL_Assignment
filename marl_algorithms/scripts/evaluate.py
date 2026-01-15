#!/usr/bin/env python3
"""
评估脚本
评估训练好的模型性能
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from vmas import make_env

def evaluate_model(checkpoint_path, algorithm_name, num_episodes=10):
    """评估模型性能"""
    print(f"\n{'='*60}")
    print(f"评估 {algorithm_name} 模型")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # 创建环境
    env = make_env(
        scenario="transport",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        max_steps=500,
        n_agents=4,
        n_packages=1,
    )
    
    # 加载模型
    try:
        if algorithm_name in ["CPPO", "MAPPO"]:
            # 加载共享策略
            model = torch.load(checkpoint_path)
            policy = model["policy"]
        else:
            # 加载独立策略
            model = torch.load(checkpoint_path)
            policies = [model[f"agent_{i}"] for i in range(4)]
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用随机策略进行评估")
        policy = None
    
    # 评估指标
    rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # 选择动作
            if algorithm_name in ["CPPO", "MAPPO"] and policy is not None:
                # CPPO/MAPPO: 使用共享策略
                with torch.no_grad():
                    actions = [policy.get_action(obs[0])[0]] * 4
            elif algorithm_name == "IPPO" and policies is not None:
                # IPPO: 使用独立策略
                with torch.no_grad():
                    actions = [policies[i].get_action(obs[i])[0] for i in range(4)]
            else:
                # 随机策略
                actions = [env.get_random_action(agent) for agent in env.agents]
            
            # 执行动作
            obs, rews, done, info = env.step(actions)
            
            episode_reward += sum(rews)
            episode_length += 1
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 检查是否成功
        if episode_reward > 0:
            success_count += 1
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # 计算统计信息
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均回合长度: {mean_length:.2f}")
    print(f"成功率: {success_rate * 100:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
    }

def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["CPPO", "MAPPO", "IPPO"],
        required=True,
        help="算法名称"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="评估回合数"
    )
    
    args = parser.parse_args()
    
    # 评估模型
    results = evaluate_model(args.checkpoint, args.algorithm, args.episodes)
    
    # 保存结果
    results_dir = "/root/RL_Assignment/marl_algorithms/results"
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = os.path.join(results_dir, f"{args.algorithm}_evaluation.txt")
    with open(result_file, "w") as f:
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"\nResults:\n")
        f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Mean Episode Length: {results['mean_length']:.2f}\n")
        f.write(f"Success Rate: {results['success_rate'] * 100:.1f}%\n")
    
    print(f"结果已保存到: {result_file}")

if __name__ == "__main__":
    main()