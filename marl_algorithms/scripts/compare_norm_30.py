#!/usr/bin/env python3
"""
30次迭代快速对比 - 原始算法 vs 归一化算法
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/scripts')
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from train_vmas import train_algorithm


def main():
    print("\n" + "="*70)
    print("30次迭代快速对比测试")
    print("="*70 + "\n")

    # 测试1: 原始算法
    print(">>> 测试1: 原始MAPPO（不使用归一化）")
    start = time.time()
    results1 = train_algorithm("MAPPO", 30, 200, 32, False)
    time1 = time.time() - start
    final1 = results1['rewards'][-1]
    max1 = max(results1['rewards'])
    print(f"完成! 时间: {time1:.1f}s, 最终: {final1:.4f}, 最高: {max1:.4f}\n")

    # 测试2: 归一化算法
    print(">>> 测试2: 改进MAPPO（使用归一化）")
    start = time.time()
    results2 = train_algorithm("MAPPO", 30, 200, 32, True)
    time2 = time.time() - start
    final2 = results2['rewards'][-1]
    max2 = max(results2['rewards'])
    print(f"完成! 时间: {time2:.1f}s, 最终: {final2:.4f}, 最高: {max2:.4f}\n")

    # 对比
    print("="*70)
    print("对比结果:")
    print("="*70)
    print(f"最终奖励: {final1:.4f} -> {final2:.4f} (差异: {final2-final1:+.4f})")
    print(f"最高奖励: {max1:.4f} -> {max2:.4f} (差异: {max2-max1:+.4f})")
    print(f"训练时间: {time1:.1f}s -> {time2:.1f}s (差异: {time2-time1:+.1f}s)")

    if final2 > final1 + 0.05:
        print("\n✅ 归一化显著提升了性能！")
    elif final2 > final1:
        print("\n✓ 归一化略微提升了性能")
    elif final2 > final1 - 0.05:
        print("\n○ 归一化对性能影响不大")
    else:
        print("\n⚠ 归一化降低了性能")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()