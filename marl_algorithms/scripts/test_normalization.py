#!/usr/bin/env python3
"""
快速测试脚本 - 验证观测归一化的改进效果
对比原始算法和使用观测归一化的算法
"""

import sys
import os
import argparse
import time

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/scripts')
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from train_vmas import train_algorithm


def run_comparison(algorithm_name, iterations=50):
    """
    运行对比实验
    Args:
        algorithm_name: 算法名称 (CPPO, MAPPO, IPPO)
        iterations: 训练迭代次数（快速测试用50次）
    """
    print(f"\n{'='*70}")
    print(f"开始对比实验: {algorithm_name}")
    print(f"{'='*70}\n")

    # 1. 训练原始算法（不使用归一化）
    print(">>> 步骤1: 训练原始算法（不使用观测归一化）")
    start_time = time.time()
    results_baseline = train_algorithm(
        algorithm_name=algorithm_name,
        num_iterations=iterations,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=False
    )
    baseline_time = time.time() - start_time

    baseline_final_reward = results_baseline['rewards'][-1]
    baseline_max_reward = max(results_baseline['rewards'])

    print(f"\n原始算法结果:")
    print(f"  - 训练时间: {baseline_time:.2f}秒")
    print(f"  - 最终奖励: {baseline_final_reward:.4f}")
    print(f"  - 最高奖励: {baseline_max_reward:.4f}")

    # 2. 训练改进算法（使用归一化）
    print(f"\n>>> 步骤2: 训练改进算法（使用观测归一化）")
    start_time = time.time()
    results_improved = train_algorithm(
        algorithm_name=algorithm_name,
        num_iterations=iterations,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=True
    )
    improved_time = time.time() - start_time

    improved_final_reward = results_improved['rewards'][-1]
    improved_max_reward = max(results_improved['rewards'])

    print(f"\n改进算法结果:")
    print(f"  - 训练时间: {improved_time:.2f}秒")
    print(f"  - 最终奖励: {improved_final_reward:.4f}")
    print(f"  - 最高奖励: {improved_max_reward:.4f}")

    # 3. 对比分析
    print(f"\n{'='*70}")
    print(f"对比分析结果")
    print(f"{'='*70}")

    final_improvement = improved_final_reward - baseline_final_reward
    max_improvement = improved_max_reward - baseline_max_reward

    print(f"\n最终奖励改进:")
    print(f"  - 原始: {baseline_final_reward:.4f}")
    print(f"  - 改进: {improved_final_reward:.4f}")
    print(f"  - 提升: {final_improvement:+.4f} ({final_improvement/baseline_final_reward*100:+.1f}%)" if baseline_final_reward != 0 else f"  - 提升: {final_improvement:+.4f}")

    print(f"\n最高奖励改进:")
    print(f"  - 原始: {baseline_max_reward:.4f}")
    print(f"  - 改进: {improved_max_reward:.4f}")
    print(f"  - 提升: {max_improvement:+.4f} ({max_improvement/baseline_max_reward*100:+.1f}%)" if baseline_max_reward != 0 else f"  - 提升: {max_improvement:+.4f}")

    # 4. 结论
    print(f"\n{'='*70}")
    print(f"结论")
    print(f"{'='*70}")

    if final_improvement > 0.05:
        print("✅ 观测归一化显著提升了算法性能！")
        print(f"   最终奖励提升了 {final_improvement:.4f}")
    elif final_improvement > 0:
        print("✓ 观测归一化略微提升了算法性能")
        print(f"   最终奖励提升了 {final_improvement:.4f}")
    elif final_improvement > -0.05:
        print("○ 观测归一化对性能影响不大")
        print(f"   最终奖励变化了 {final_improvement:.4f}")
    else:
        print("⚠ 观测归一化降低了性能，可能需要调整参数")
        print(f"   最终奖励下降了 {abs(final_improvement):.4f}")

    print(f"\n训练时间对比:")
    print(f"  - 原始: {baseline_time:.2f}秒")
    print(f"  - 改进: {improved_time:.2f}秒")
    print(f"  - 时间增加: {improved_time - baseline_time:+.2f}秒 ({(improved_time/baseline_time-1)*100:+.1f}%)")

    return {
        'algorithm': algorithm_name,
        'baseline': {
            'final_reward': baseline_final_reward,
            'max_reward': baseline_max_reward,
            'time': baseline_time
        },
        'improved': {
            'final_reward': improved_final_reward,
            'max_reward': improved_max_reward,
            'time': improved_time
        },
        'improvement': {
            'final': final_improvement,
            'max': max_improvement
        }
    }


def main():
    parser = argparse.ArgumentParser(description="快速测试观测归一化改进效果")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["CPPO", "MAPPO", "IPPO"],
        default="MAPPO",
        help="选择算法进行测试（默认MAPPO）"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="训练迭代次数（默认50次，约1-2分钟）"
    )

    args = parser.parse_args()

    print(f"\n快速测试配置:")
    print(f"  - 算法: {args.algorithm}")
    print(f"  - 迭代次数: {args.iterations}")
    print(f"  - 预计总时间: ~{args.iterations * 200 / 377 * 2:.1f}秒（两次训练）")

    results = run_comparison(args.algorithm, args.iterations)

    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()