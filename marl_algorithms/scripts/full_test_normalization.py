#!/usr/bin/env python3
"""
完整测试 - 观测归一化改进效果
300次迭代，用于生成完整的实验报告
"""

import sys
import os
import time
import json
import torch

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/scripts')
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from train_vmas import train_algorithm


def main():
    print("\n" + "="*70)
    print("完整测试：观测归一化改进效果")
    print("算法: MAPPO | 迭代次数: 300")
    print("="*70 + "\n")

    results = {}

    # 测试1: 原始算法
    print(">>> 第1部分：训练原始MAPPO算法（不使用观测归一化）")
    print("预计时间：~13分钟\n")
    start_time = time.time()
    results['baseline'] = train_algorithm(
        algorithm_name="MAPPO",
        num_iterations=300,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=False
    )
    baseline_time = time.time() - start_time

    baseline_final = results['baseline']['rewards'][-1]
    baseline_max = max(results['baseline']['rewards'])
    baseline_mean = sum(results['baseline']['rewards']) / len(results['baseline']['rewards'])

    print(f"\n原始算法完成!")
    print(f"  训练时间: {baseline_time:.2f}秒 ({baseline_time/60:.2f}分钟)")
    print(f"  最终奖励: {baseline_final:.4f}")
    print(f"  最高奖励: {baseline_max:.4f}")
    print(f"  平均奖励: {baseline_mean:.4f}")

    # 测试2: 改进算法
    print(f"\n{'='*70}\n")
    print(">>> 第2部分：训练改进MAPPO算法（使用观测归一化）")
    print("预计时间：~13分钟\n")
    start_time = time.time()
    results['improved'] = train_algorithm(
        algorithm_name="MAPPO",
        num_iterations=300,
        num_steps_per_iter=200,
        num_envs=32,
        use_normalization=True
    )
    improved_time = time.time() - start_time

    improved_final = results['improved']['rewards'][-1]
    improved_max = max(results['improved']['rewards'])
    improved_mean = sum(results['improved']['rewards']) / len(results['improved']['rewards'])

    print(f"\n改进算法完成!")
    print(f"  训练时间: {improved_time:.2f}秒 ({improved_time/60:.2f}分钟)")
    print(f"  最终奖励: {improved_final:.4f}")
    print(f"  最高奖励: {improved_max:.4f}")
    print(f"  平均奖励: {improved_mean:.4f}")

    # 对比分析
    print(f"\n{'='*70}")
    print("对比分析结果")
    print("="*70)

    final_improvement = improved_final - baseline_final
    max_improvement = improved_max - baseline_max
    mean_improvement = improved_mean - baseline_mean
    time_overhead = improved_time - baseline_time

    print(f"\n1. 最终奖励对比:")
    print(f"   原始: {baseline_final:.4f}")
    print(f"   改进: {improved_final:.4f}")
    print(f"   提升: {final_improvement:+.4f} ({final_improvement/baseline_final*100:+.1f}%)" if baseline_final != 0 else f"   提升: {final_improvement:+.4f}")

    print(f"\n2. 最高奖励对比:")
    print(f"   原始: {baseline_max:.4f}")
    print(f"   改进: {improved_max:.4f}")
    print(f"   提升: {max_improvement:+.4f} ({max_improvement/baseline_max*100:+.1f}%)" if baseline_max != 0 else f"   提升: {max_improvement:+.4f}")

    print(f"\n3. 平均奖励对比:")
    print(f"   原始: {baseline_mean:.4f}")
    print(f"   改进: {improved_mean:.4f}")
    print(f"   提升: {mean_improvement:+.4f} ({mean_improvement/baseline_mean*100:+.1f}%)" if baseline_mean != 0 else f"   提升: {mean_improvement:+.4f}")

    print(f"\n4. 训练时间对比:")
    print(f"   原始: {baseline_time:.2f}秒")
    print(f"   改进: {improved_time:.2f}秒")
    print(f"   开销: {time_overhead:+.2f}秒 ({time_overhead/baseline_time*100:+.1}%)")

    # 结论
    print(f"\n{'='*70}")
    print("结论")
    print("="*70)

    if final_improvement > 0.1:
        print("✅ 观测归一化显著提升了算法性能！")
        print(f"   最终奖励提升了 {final_improvement:.4f}，改进幅度达 {final_improvement/baseline_final*100:.1f}%")
    elif final_improvement > 0:
        print("✓ 观测归一化提升了算法性能")
        print(f"   最终奖励提升了 {final_improvement:.4f}")
    elif final_improvement > -0.1:
        print("○ 观测归一化对性能影响不大")
        print(f"   最终奖励变化了 {final_improvement:.4f}")
    else:
        print("⚠ 观测归一化降低了性能")
        print(f"   最终奖励下降了 {abs(final_improvement):.4f}")

    if time_overhead < baseline_time * 0.05:
        print("\n✅ 训练开销极小，几乎无额外计算成本")
    elif time_overhead < baseline_time * 0.1:
        print(f"\n✓ 训练开销较小，仅增加 {time_overhead/baseline_time*100:.1f}%")
    else:
        print(f"\n⚠ 训练开销较大，增加 {time_overhead/baseline_time*100:.1f}%")

    print(f"\n{'='*70}")

    # 保存对比结果
    comparison_results = {
        'algorithm': 'MAPPO',
        'iterations': 300,
        'baseline': {
            'final_reward': float(baseline_final),
            'max_reward': float(baseline_max),
            'mean_reward': float(baseline_mean),
            'training_time': float(baseline_time),
            'rewards': [float(r) for r in results['baseline']['rewards']]
        },
        'improved': {
            'final_reward': float(improved_final),
            'max_reward': float(improved_max),
            'mean_reward': float(improved_mean),
            'training_time': float(improved_time),
            'rewards': [float(r) for r in results['improved']['rewards']]
        },
        'improvement': {
            'final': float(final_improvement),
            'max': float(max_improvement),
            'mean': float(mean_improvement),
            'time_overhead': float(time_overhead)
        },
        'conclusion': {
            'performance_improved': final_improvement > 0,
            'significant_improvement': final_improvement > 0.1,
            'low_overhead': time_overhead < baseline_time * 0.05
        }
    }

    # 保存到文件
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_file = f"/root/RL_Assignment/marl_algorithms/results/normalization_comparison_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n对比结果已保存到: {results_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()