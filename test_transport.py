#!/usr/bin/env python3
"""
Transport环境测试脚本
测试VMAS Transport场景是否正常工作
"""

import sys
import torch

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from vmas import make_env

def test_transport_env():
    """测试Transport环境"""
    print("=" * 60)
    print("测试VMAS Transport环境")
    print("=" * 60)

    # 创建Transport环境
    print("\n1. 创建Transport环境...")
    env = make_env(
        scenario="transport",
        num_envs=4,  # 4个并行环境
        device="cpu",
        continuous_actions=True,
        n_agents=4,  # 4个智能体
        n_packages=1,  # 1个包裹
    )
    print(f"✓ 环境创建成功")
    print(f"  - 场景: transport")
    print(f"  - 并行环境数: {env.num_envs}")
    print(f"  - 智能体数量: {env.n_agents}")
    print(f"  - 设备: {env.device}")

    # 重置环境
    print("\n2. 重置环境...")
    obs = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  - 观测形状: {[o.shape for o in obs]}")

    # 运行几个步骤
    print("\n3. 运行模拟步骤...")
    for step in range(10):
        # 获取随机动作
        actions = [env.get_random_action(agent) for agent in env.agents]

        # 执行步骤
        obs, rews, dones, info = env.step(actions)

        # 打印信息
        print(f"  步骤 {step+1}:")
        print(f"    - 奖励: {[round(r.mean().item(), 3) for r in rews]}")
        print(f"    - 完成: {[d.any().item() for d in dones]}")

    print("\n4. 测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_transport_env()
        print("\n✓ 所有测试通过!")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)