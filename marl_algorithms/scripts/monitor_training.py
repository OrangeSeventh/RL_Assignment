#!/usr/bin/env python3
"""
监控训练进度
"""

import os
import time

def monitor():
    log_file = "/root/RL_Assignment/marl_algorithms/results/full_test.log"

    print("监控训练进度...")
    print("="*70)

    while True:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"\n最后20行输出 ({time.strftime('%H:%M:%S')}):")
                    print("="*70)
                    for line in lines[-20:]:
                        print(line.rstrip())
                    print("="*70)

        # 检查是否完成
        if os.path.exists("/root/RL_Assignment/marl_algorithms/results/normalization_comparison_*.json"):
            print("\n✅ 训练完成！")
            break

        time.sleep(30)

if __name__ == "__main__":
    monitor()