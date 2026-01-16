#!/usr/bin/env python3
"""
超简单测试 - 只运行30次迭代验证归一化是否工作
"""

import sys
import os
import torch

# 添加路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env
from transport_config import ENV_CONFIG
from normalization import NormalizeObservation


def test_normalization():
    """测试观测归一化功能"""
    print("\n" + "="*70)
    print("测试观测归一化功能")
    print("="*70 + "\n")

    # 创建环境
    print("1. 创建VMAS环境...")
    env = make_env(
        scenario=ENV_CONFIG["scenario"],
        num_envs=16,
        device=ENV_CONFIG["device"],
        continuous_actions=ENV_CONFIG["continuous_actions"],
        max_steps=ENV_CONFIG["max_steps"],
        wrapper=None,
        dict_spaces=False,
        n_agents=ENV_CONFIG["n_agents"],
        n_packages=ENV_CONFIG["n_packages"],
        package_width=ENV_CONFIG["package_width"],
        package_length=ENV_CONFIG["package_length"],
        package_mass=ENV_CONFIG["package_mass"],
    )

    obs_dim = env.observation_space[0].shape[0]
    print(f"   观测维度: {obs_dim}")
    print(f"   智能体数量: {len(env.agents)}")

    # 创建归一化器
    print("\n2. 创建观测归一化器...")
    obs_normalizer = NormalizeObservation(obs_dim, pre_collect_steps=10)
    print(f"   预收集步数: {obs_normalizer.pre_collect_steps}")

    # 预收集数据
    print("\n3. 预收集10步数据...")
    obs = env.reset()
    for step in range(10):
        # 使用随机动作
        actions = []
        for agent in env.agents:
            action = torch.randn(16, 2, device=env.device)
            action = torch.clamp(action, -1.0, 1.0)
            actions.append(action)

        obs, rews, dones, info = env.step(actions)

        # 收集观测
        for i, agent in enumerate(env.agents):
            obs_tensor = obs[i] if isinstance(obs[i], torch.Tensor) else torch.tensor(obs[i], dtype=torch.float32)
            obs_normalizer.pre_collect(obs_tensor)

    obs_normalizer.finalize_pre_collection()

    # 测试归一化
    print("\n4. 测试归一化效果...")
    obs = env.reset()

    # 获取原始观测
    raw_obs = obs[0] if isinstance(obs[0], torch.Tensor) else torch.tensor(obs[0], dtype=torch.float32)
    print(f"   原始观测统计:")
    print(f"     均值: {raw_obs.mean():.4f}")
    print(f"     标准差: {raw_obs.std():.4f}")
    print(f"     范围: [{raw_obs.min():.4f}, {raw_obs.max():.4f}]")

    # 获取归一化后的观测
    norm_obs = obs_normalizer.normalize(raw_obs, update_stats=False)
    print(f"   归一化观测统计:")
    print(f"     均值: {norm_obs.mean():.4f}")
    print(f"     标准差: {norm_obs.std():.4f}")
    print(f"     范围: [{norm_obs.min():.4f}, {norm_obs.max():.4f}]")

    # 验证归一化效果
    print("\n5. 验证归一化效果...")
    mean_close_to_zero = torch.abs(norm_obs.mean()) < 0.1
    std_close_to_one = torch.abs(norm_obs.std() - 1.0) < 0.3

    if mean_close_to_zero and std_close_to_one:
        print("   ✅ 归一化工作正常！")
        print(f"      均值接近0: {norm_obs.mean():.4f}")
        print(f"      标准差接近1: {norm_obs.std():.4f}")
    else:
        print("   ⚠ 归一化效果可能不够理想")
        print(f"      均值: {norm_obs.mean():.4f} (期望接近0)")
        print(f"      标准差: {norm_obs.std():.4f} (期望接近1)")

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_normalization()