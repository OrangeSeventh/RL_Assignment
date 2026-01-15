#!/usr/bin/env python3
"""
简化版训练脚本 - 快速开始训练MAPPO算法
"""
import argparse
import os
import sys

# 添加项目路径
sys.path.insert(0, "/root/RL_Assignment")

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    
    import vmas
    from vmas import make_env
except ImportError as e:
    print(f"错误: 缺少必要的依赖包: {e}")
    print("请确保已激活虚拟环境并安装了所有依赖")
    sys.exit(1)


def create_vmas_env(config):
    """创建VMAS Transport环境"""
    from vmas.simulator.utils import Wrapper
    
    # 创建VMAS环境
    env = make_env(
        scenario="transport",
        num_envs=config.get("num_envs", 32),
        device="cpu",
        continuous_actions=True,
        wrapper=Wrapper.RLLIB,
        max_steps=500,
        seed=config.get("seed", 0),
        dict_spaces=False,
        # Transport特定参数
        n_agents=4,
        n_packages=1,
        package_mass=50,
        package_width=0.15,
        package_length=0.15,
    )
    return env


def train_mappo(iterations=100):
    """训练MAPPO算法"""
    print("=" * 60)
    print("开始训练MAPPO算法")
    print("=" * 60)
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 注册环境
    register_env("vmas_transport", lambda config: create_vmas_env(config))
    
    # 配置MAPPO
    config = (
        PPOConfig()
        .environment(
            env="vmas_transport",
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=400,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={
                # 所有智能体共享策略
                "shared_policy": (None, None, None, {"framework": "torch"}),
            },
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy"),
        )
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=200,
        )
        .resources(
            num_gpus=0,  # 使用CPU
        )
    )
    
    # 创建算法
    algo = config.build()
    
    print(f"\n训练配置:")
    print(f"  - 迭代次数: {iterations}")
    print(f"  - 批次大小: 4000")
    print(f"  - 学习率: 3e-4")
    print(f"  - SGD迭代次数: 10")
    print(f"  - 折扣因子: 0.99")
    print(f"  - GAE参数: 0.95")
    print(f"  - 裁剪参数: 0.2")
    print()
    
    # 训练循环
    for i in range(iterations):
        result = algo.train()
        
        # 每10次迭代打印一次
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{iterations}")
            print(f"  - 平均奖励: {result['episode_reward_mean']:.2f}")
            print(f"  - 平均回合长度: {result['episode_len_mean']:.2f}")
            print(f"  - 学习率: {result['info']['learner']['default_policy']['cur_lr']:.6f}")
            print(f"  - 策略损失: {result['info']['learner']['default_policy']['policy_loss']:.4f}")
            print(f"  - 价值损失: {result['info']['learner']['default_policy']['vf_loss']:.4f}")
            print()
    
    # 保存模型
    checkpoint_path = algo.save("/root/RL_Assignment/marl_algorithms/checkpoints/MAPPO")
    print(f"模型已保存到: {checkpoint_path}")
    
    # 关闭算法
    algo.stop()
    
    return checkpoint_path


def train_ippo(iterations=100):
    """训练IPPO算法"""
    print("=" * 60)
    print("开始训练IPPO算法")
    print("=" * 60)
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 注册环境
    register_env("vmas_transport", lambda config: create_vmas_env(config))
    
    # 配置IPPO（每个智能体独立的策略）
    config = (
        PPOConfig()
        .environment(
            env="vmas_transport",
            disable_env_checking=True,
        )
        .framework("torch")
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=400,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={
                # 每个智能体独立的策略
                "agent_0": (None, None, None, {"framework": "torch"}),
                "agent_1": (None, None, None, {"framework": "torch"}),
                "agent_2": (None, None, None, {"framework": "torch"}),
                "agent_3": (None, None, None, {"framework": "torch"}),
            },
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id),
        )
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=200,
        )
        .resources(
            num_gpus=0,
        )
    )
    
    # 创建算法
    algo = config.build()
    
    print(f"\n训练配置:")
    print(f"  - 迭代次数: {iterations}")
    print(f"  - 独立策略: 4个（每个智能体一个）")
    print()
    
    # 训练循环
    for i in range(iterations):
        result = algo.train()
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{iterations}")
            print(f"  - 平均奖励: {result['episode_reward_mean']:.2f}")
            print(f"  - 平均回合长度: {result['episode_len_mean']:.2f}")
            print()
    
    # 保存模型
    checkpoint_path = algo.save("/root/RL_Assignment/marl_algorithms/checkpoints/IPPO")
    print(f"模型已保存到: {checkpoint_path}")
    
    # 关闭算法
    algo.stop()
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="训练MARL算法")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["MAPPO", "IPPO"],
        default="MAPPO",
        help="选择算法: MAPPO 或 IPPO"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="训练迭代次数"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"VMAS Transport任务 - {args.algorithm}算法训练")
    print(f"{'=' * 60}\n")
    
    try:
        if args.algorithm == "MAPPO":
            checkpoint_path = train_mappo(args.iterations)
        else:  # IPPO
            checkpoint_path = train_ippo(args.iterations)
        
        print(f"\n{'=' * 60}")
        print(f"训练完成！")
        print(f"{'=' * 60}")
        print(f"算法: {args.algorithm}")
        print(f"迭代次数: {args.iterations}")
        print(f"检查点: {checkpoint_path}")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()