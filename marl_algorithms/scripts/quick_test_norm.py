#!/usr/bin/env python3
"""
快速验证观测归一化改进效果 - 简化版
只测试50次迭代，快速验证
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/scripts')
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from train_vmas import train_algorithm


def main():
    print(f"\n{'='*70}")
    print(f"快速验证观测归一化改进效果")
    print(f"{'='*70}\n")

    # 测试1: 原始算法（不使用归一化）
    print(">>> 测试1: 原始MAPPO算法（不使用观测归一化）")
    print("训练50次迭代...\n")
    start_time = time.time()
    results_baseline = train_algorithm(
        algorithm_name="MAPPO",
        num_iterations=50,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=False
    )
    baseline_time = time.time() - start_time

    baseline_final = results_baseline['rewards'][-1]
    baseline_max = max(results_baseline['rewards'])

    print(f"\n原始算法结果:")
    print(f"  - 训练时间: {baseline_time:.2f}秒")
    print(f"  - 最终奖励: {baseline_final:.4f}")
    print(f"  - 最高奖励: {baseline_max:.4f}")

    # 测试2: 改进算法（使用归一化）
    print(f"\n{'='*70}\n")
    print(">>> 测试2: 改进MAPPO算法（使用观测归一化）")
    print("训练50次迭代...\n")
    start_time = time.time()
    results_improved = train_algorithm(
        algorithm_name="MAPPO",
        num_iterations=50,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=True
    )
    improved_time = time.time() - start_time

    improved_final = results_improved['rewards'][-1]
    improved_max = max(results_improved['rewards'])

    print(f"\n改进算法结果:")
    print(f"  - 训练时间: {improved_time:.2f}秒")
    print(f"  - 最终奖励: {improved_final:.4f}")
    print(f"  - 最高奖励: {improved_max:.4f}")

    # 对比分析
    print(f"\n{'='*70}")
    print(f"对比分析")
    print(f"{'='*70}\n")

    final_diff = improved_final - baseline_final
    max_diff = improved_max - baseline_max

    print(f"最终奖励:")
    print(f"  原始: {baseline_final:.4f}")
    print(f"  改进: {improved_final:.4f}")
    print(f"  差异: {final_diff:+.4f}")

    print(f"\n最高奖励:")
    print(f"  原始: {baseline_max:.4f}")
    print(f"  改进: {improved_max:.4f}")
    print(f"  差异: {max_diff:+.4f}")

    print(f"\n{'='*70}")
    if final_diff > 0.05:
        print("✅ 观测归一化显著提升了性能！")
    elif final_diff > 0:
        print("✓ 观测归一化略微提升了性能")
    elif final_diff > -0.05:
        print("○ 观测归一化对性能影响不大")
    else:
        print("⚠ 观测归一化降低了性能")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()