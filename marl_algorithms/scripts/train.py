#!/usr/bin/env python3
"""
Transport任务训练脚本
支持CPPO、MAPPO、IPPO三种算法的训练
"""

import sys
import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

import transport_config as config_module
ENV_CONFIG = config_module.ENV_CONFIG
TRAINING_CONFIG = config_module.TRAINING_CONFIG
ALGORITHM_CONFIGS = config_module.ALGORITHM_CONFIGS
PATH_CONFIG = config_module.PATH_CONFIG
LOG_CONFIG = config_module.LOG_CONFIG

def create_vmas_env(env_config):
    """创建VMAS Transport环境"""
    from vmas import make_env
    import gymnasium as gym
    
    class VMASEnv(gym.Env):
        def __init__(self, config):
            self.env = make_env(
                scenario=env_config["scenario"],
                num_envs=env_config["num_envs"],
                device=env_config["device"],
                continuous_actions=env_config["continuous_actions"],
                max_steps=env_config["max_steps"],
                dict_spaces=env_config["dict_spaces"],
                n_agents=env_config["n_agents"],
                n_packages=env_config["n_packages"],
                package_width=env_config["package_width"],
                package_length=env_config["package_length"],
                package_mass=env_config["package_mass"],
            )
            
            # 获取观测和动作空间
            self.observation_space = self.env.observation_space[0]
            self.action_space = self.env.action_space[0]
            self.num_agents = self.env.n_agents
            
        def reset(self, seed=None, options=None):
            obs = self.env.reset(seed=seed)
            return obs[0], {}
        
        def step(self, action):
            actions = [action] * self.num_agents
            obs, rews, dones, truncateds, infos = self.env.step(actions)
            return obs[0], rews[0], dones[0], {}, infos
    
    return VMASEnv(env_config)

def train_algorithm(algorithm_name, num_iterations=1000):
    """训练指定算法"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algorithm_name} 算法")
    print(f"{'='*60}\n")

    # 调试：在函数开始时检查PPOConfig
    print(f"DEBUG (function start): PPOConfig type = {type(PPOConfig)}")

    # 初始化Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # 调试：Ray初始化后检查PPOConfig
    print(f"DEBUG (after ray.init): PPOConfig type = {type(PPOConfig)}")

    # 获取算法配置
    algo_config = ALGORITHM_CONFIGS[algorithm_name]

    # 调试：获取算法配置后检查PPOConfig
    print(f"DEBUG (after algo_config): PPOConfig type = {type(PPOConfig)}")

    # 创建PPO配置
    print(f"DEBUG (before PPOConfig()): About to call PPOConfig()")
    config = (
        PPOConfig()
        .environment(
            create_vmas_env,
            env_config=ENV_CONFIG,
        )
        .framework("torch")
        .training(
            train_batch_size=TRAINING_CONFIG["train_batch_size"],
            lr=TRAINING_CONFIG["lr"],
            gamma=TRAINING_CONFIG["gamma"],
            lambda_=TRAINING_CONFIG["lambda_"],
            clip_param=TRAINING_CONFIG["clip_param"],
            vf_loss_coeff=TRAINING_CONFIG["vf_loss_coeff"],
            entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
            sgd_minibatch_size=TRAINING_CONFIG["sgd_minibatch_size"],
            num_sgd_iter=TRAINING_CONFIG["num_sgd_iter"],
        )
        .multiagent(**algo_config.get("multiagent", {}))
        .evaluation(
            evaluation_interval=TRAINING_CONFIG["evaluation_interval"],
            evaluation_num_episodes=TRAINING_CONFIG["evaluation_num_episodes"],
            evaluation_config=TRAINING_CONFIG["evaluation_config"],
        )
    )
    
    # 创建训练器
    algo = config.build()
    
    # 训练循环
    results_dir = os.path.join(PATH_CONFIG["results_dir"], algorithm_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"训练配置:")
    print(f"  - 算法: {algorithm_name}")
    print(f"  - 迭代次数: {num_iterations}")
    print(f"  - 批次大小: {TRAINING_CONFIG['train_batch_size']}")
    print(f"  - 学习率: {TRAINING_CONFIG['lr']}")
    print(f"  - 结果保存路径: {results_dir}")
    print()
    
    for i in range(num_iterations):
        result = algo.train()
        
        # 打印训练进度
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations}")
            print(f"  - Episode Reward Mean: {result['episode_reward_mean']:.2f}")
            print(f"  - Episode Length Mean: {result['episode_len_mean']:.2f}")
            print(f"  - Evaluation Reward Mean: {result.get('evaluation', {}).get('episode_reward_mean', 'N/A')}")
            print()
        
        # 保存checkpoint
        if (i + 1) % 100 == 0:
            checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"Checkpoint saved: {checkpoint_path}")
            print()
    
    # 保存最终模型
    checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint = algo.save(checkpoint_dir)
    print(f"\nFinal checkpoint saved: {final_checkpoint}")
    
    # 关闭Ray
    ray.shutdown()
    
    return final_checkpoint

def main():
    parser = argparse.ArgumentParser(description="训练Transport任务的MARL算法")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["CPPO", "MAPPO", "IPPO"],
        required=True,
        help="选择算法: CPPO, MAPPO, IPPO"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="训练迭代次数"
    )
    
    args = parser.parse_args()
    
    # 训练算法
    checkpoint_path = train_algorithm(args.algorithm, args.iterations)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"算法: {args.algorithm}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()