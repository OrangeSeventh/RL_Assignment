#!/usr/bin/env python3
"""
对比评估脚本：比较原始算法和改进算法的性能
"""

import sys
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

def load_metrics(algorithm_name, metrics_type='original'):
    """加载训练指标"""
    if metrics_type == 'original':
        metrics_file = f"/root/RL_Assignment/marl_algorithms/results/{algorithm_name}/{algorithm_name}_transport_*.json"
    else:
        metrics_file = f"/root/RL_Assignment/marl_algorithms/results/{algorithm_name}/{algorithm_name}_metrics.json"

    # 这里简化处理，实际需要根据文件结构加载
    print(f"加载指标: {metrics_file}")
    return None

def evaluate_model(model, env, num_episodes=10):
    """评估模型性能"""
    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        all_lengths.append(steps)

    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_length': np.mean(all_lengths),
        'std_length': np.std(all_lengths),
        'all_rewards': all_rewards,
    }

def compare_algorithms(algorithms=['CPPO', 'MAPPO', 'IPPO'], num_episodes=10):
    """对比不同算法的性能"""
    print(f"\n{'='*60}")
    print(f"算法性能对比评估")
    print(f"{'='*60}\n")

    results = {}

    for algorithm in algorithms:
        print(f"\n评估 {algorithm} 算法...")

        # 创建环境
        env = make_env(
            scenario="transport",
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            max_steps=500,
            wrapper="gymnasium",
            dict_spaces=False,
            terminated_truncated=True,
            n_agents=4,
            n_packages=1,
            package_width=0.15,
            package_length=0.15,
            package_mass=50,
        )

        # 尝试加载改进的模型
        improved_model_path = f"/root/RL_Assignment/marl_algorithms/checkpoints/{algorithm}/{algorithm}_final_model"
        original_model_path = f"/root/RL_Assignment/marl_algorithms/checkpoints/{algorithm}/{algorithm}_model"

        results[algorithm] = {}

        # 评估改进模型
        if os.path.exists(improved_model_path):
            print(f"  加载改进模型: {improved_model_path}")
            try:
                improved_model = PPO.load(improved_model_path)
                improved_metrics = evaluate_model(improved_model, env, num_episodes)
                results[algorithm]['improved'] = improved_metrics
                print(f"  改进模型平均奖励: {improved_metrics['mean_reward']:.4f} ± {improved_metrics['std_reward']:.4f}")
            except Exception as e:
                print(f"  加载改进模型失败: {e}")

        # 评估原始模型
        if os.path.exists(original_model_path):
            print(f"  加载原始模型: {original_model_path}")
            try:
                original_model = PPO.load(original_model_path)
                original_metrics = evaluate_model(original_model, env, num_episodes)
                results[algorithm]['original'] = original_metrics
                print(f"  原始模型平均奖励: {original_metrics['mean_reward']:.4f} ± {original_metrics['std_reward']:.4f}")
            except Exception as e:
                print(f"  加载原始模型失败: {e}")

    # 打印对比结果
    print(f"\n{'='*60}")
    print(f"性能对比总结")
    print(f"{'='*60}\n")

    for algorithm in algorithms:
        print(f"\n{algorithm}:")
        if 'original' in results[algorithm]:
            print(f"  原始算法: {results[algorithm]['original']['mean_reward']:.4f} ± {results[algorithm]['original']['std_reward']:.4f}")
        if 'improved' in results[algorithm]:
            print(f"  改进算法: {results[algorithm]['improved']['mean_reward']:.4f} ± {results[algorithm]['improved']['std_reward']:.4f}")

        if 'original' in results[algorithm] and 'improved' in results[algorithm]:
            improvement = results[algorithm]['improved']['mean_reward'] - results[algorithm]['original']['mean_reward']
            improvement_pct = (improvement / abs(results[algorithm]['original']['mean_reward'])) * 100 if results[algorithm]['original']['mean_reward'] != 0 else 0
            print(f"  改进幅度: {improvement:.4f} ({improvement_pct:.1f}%)")

    # 保存结果
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = f"/root/RL_Assignment/marl_algorithms/results/comparison_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {results_file}")

    return results

def plot_comparison(results):
    """绘制性能对比图"""
    algorithms = list(results.keys())
    original_rewards = []
    improved_rewards = []

    for algorithm in algorithms:
        if 'original' in results[algorithm]:
            original_rewards.append(results[algorithm]['original']['mean_reward'])
        else:
            original_rewards.append(0)

        if 'improved' in results[algorithm]:
            improved_rewards.append(results[algorithm]['improved']['mean_reward'])
        else:
            improved_rewards.append(0)

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, original_rewards, width, label='原始算法')
    rects2 = ax.bar(x + width/2, improved_rewards, width, label='改进算法')

    ax.set_ylabel('平均奖励')
    ax.set_title('原始算法 vs 改进算法性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_file = f"/root/RL_Assignment/marl_algorithms/results/comparison_plot_{timestamp}.png"
    plt.savefig(plot_file)
    print(f"\n对比图已保存到: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="对比评估原始算法和改进算法")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs='+',
        default=['CPPO', 'MAPPO', 'IPPO'],
        help="要评估的算法列表"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="评估回合数"
    )

    args = parser.parse_args()

    results = compare_algorithms(args.algorithms, args.episodes)
    plot_comparison(results)

if __name__ == "__main__":
    main()