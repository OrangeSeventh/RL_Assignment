#!/usr/bin/env python3
"""
对比脚本
对比CPPO、MAPPO、IPPO三种算法的性能
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

def compare_algorithms(results_dir="/root/RL_Assignment/marl_algorithms/results"):
    """对比三种算法的性能"""
    print(f"\n{'='*60}")
    print(f"对比算法性能")
    print(f"{'='*60}\n")
    
    algorithms = ["CPPO", "MAPPO", "IPPO"]
    results = {}
    
    # 读取评估结果
    for algo in algorithms:
        result_file = os.path.join(results_dir, f"{algo}_evaluation.txt")
        
        if os.path.exists(result_file):
            print(f"读取 {algo} 结果...")
            with open(result_file, "r") as f:
                content = f.read()
                
            # 解析结果
            results[algo] = {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "success_rate": 0.0,
            }
            
            for line in content.split("\n"):
                if "Mean Reward:" in line:
                    parts = line.split(":")[1].strip().split(" ± ")
                    results[algo]["mean_reward"] = float(parts[0])
                    results[algo]["std_reward"] = float(parts[1])
                elif "Mean Episode Length:" in line:
                    results[algo]["mean_length"] = float(line.split(":")[1].strip())
                elif "Success Rate:" in line:
                    results[algo]["success_rate"] = float(line.split(":")[1].strip().replace("%", "")) / 100
            
            print(f"  ✓ {algo}: Reward={results[algo]['mean_reward']:.2f}, Success={results[algo]['success_rate']*100:.1f}%")
        else:
            print(f"  ✗ {algo}: 结果文件不存在")
            results[algo] = None
    
    if not any(results.values()):
        print("没有找到任何评估结果")
        return
    
    # 打印对比结果
    print(f"\n{'='*60}")
    print(f"算法性能对比")
    print(f"{'='*60}")
    print(f"{'算法':<10} {'平均奖励':<15} {'成功率':<15} {'回合长度':<15}")
    print(f"{'-'*60}")
    
    for algo in algorithms:
        if results[algo]:
            print(f"{algo:<10} {results[algo]['mean_reward']:<15.2f} {results[algo]['success_rate']*100:<15.1f}% {results[algo]['mean_length']:<15.2f}")
    
    print(f"{'='*60}\n")
    
    # 绘制对比图表
    plot_results(results, results_dir)
    
    return results

def plot_results(results, results_dir):
    """绘制性能对比图表"""
    algorithms = [algo for algo in results.keys() if results[algo] is not None]
    
    if not algorithms:
        print("没有可用的数据用于绘图")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 提取数据
    mean_rewards = [results[algo]["mean_reward"] for algo in algorithms]
    std_rewards = [results[algo]["std_reward"] for algo in algorithms]
    success_rates = [results[algo]["success_rate"] * 100 for algo in algorithms]
    mean_lengths = [results[algo]["mean_length"] for algo in algorithms]
    
    # 1. 平均奖励对比
    axes[0].bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0].set_title("平均奖励对比")
    axes[0].set_ylabel("平均奖励")
    axes[0].set_xlabel("算法")
    axes[0].grid(True, alpha=0.3)
    
    # 2. 成功率对比
    axes[1].bar(algorithms, success_rates, alpha=0.7, color=['blue', 'green', 'red'])
    axes[1].set_title("成功率对比")
    axes[1].set_ylabel("成功率 (%)")
    axes[1].set_xlabel("算法")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 回合长度对比
    axes[2].bar(algorithms, mean_lengths, alpha=0.7, color=['blue', 'green', 'red'])
    axes[2].set_title("平均回合长度对比")
    axes[2].set_ylabel("回合长度")
    axes[2].set_xlabel("算法")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(results_dir, "algorithm_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存到: {plot_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="对比算法性能")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/root/RL_Assignment/marl_algorithms/results",
        help="结果目录"
    )
    
    args = parser.parse_args()
    
    # 对比算法
    results = compare_algorithms(args.results_dir)
    
    # 保存对比结果
    if results:
        results_dir = args.results_dir
        comparison_file = os.path.join(results_dir, "comparison_summary.txt")
        
        with open(comparison_file, "w") as f:
            f.write("算法性能对比总结\n")
            f.write("=" * 60 + "\n\n")
            
            for algo, result in results.items():
                if result:
                    f.write(f"{algo}:\n")
                    f.write(f"  平均奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
                    f.write(f"  成功率: {result['success_rate'] * 100:.1f}%\n")
                    f.write(f"  平均回合长度: {result['mean_length']:.2f}\n")
                    f.write("\n")
        
        print(f"对比总结已保存到: {comparison_file}")

if __name__ == "__main__":
    main()