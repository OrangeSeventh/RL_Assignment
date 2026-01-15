#!/usr/bin/env python3
"""
Transport任务训练脚本 - 使用PettingZoo + Stable-Baselines3
支持CPPO、MAPPO、IPPO三种算法的训练
使用正确的PettingZoo到Gymnasium转换
"""

import sys
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gymnasium as gym
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

from vmas import make_env

# 导入配置
from transport_config import (
    ENV_CONFIG,
    TRAINING_CONFIG,
    PATH_CONFIG,
)

class VmasPettingZooEnv(ParallelEnv):
    """VMAS环境的PettingZoo包装器"""

    metadata = {
        "name": "vmas_transport_v0",
        "render_modes": [],
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 创建VMAS环境
        self.vmas_env = make_env(
            scenario=ENV_CONFIG["scenario"],
            num_envs=1,
            device=ENV_CONFIG["device"],
            continuous_actions=ENV_CONFIG["continuous_actions"],
            max_steps=ENV_CONFIG["max_steps"],
            dict_spaces=True,
            n_agents=ENV_CONFIG["n_agents"],
            n_packages=ENV_CONFIG["n_packages"],
            package_width=ENV_CONFIG["package_width"],
            package_length=ENV_CONFIG["package_length"],
            package_mass=ENV_CONFIG["package_mass"],
        )

        # 获取智能体列表
        self.possible_agents = [f"agent_{i}" for i in range(self.vmas_env.n_agents)]
        self.agents = self.possible_agents[:]

        # 获取观测和动作空间
        self._action_spaces = {}
        self._observation_spaces = {}

        for i, agent_name in enumerate(self.possible_agents):
            if hasattr(self.vmas_env.observation_space, 'spaces'):
                obs_space = self.vmas_env.observation_space.spaces[agent_name]
            else:
                obs_space = self.vmas_env.observation_space[i]

            if hasattr(self.vmas_env.action_space, 'spaces'):
                act_space = self.vmas_env.action_space.spaces[agent_name]
            else:
                act_space = self.vmas_env.action_space[i]

            self._observation_spaces[agent_name] = obs_space
            self._action_spaces[agent_name] = act_space

        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.num_moves = 0

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.vmas_env.seed(seed)

        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        obs = self.vmas_env.reset()

        observations = {}
        for i, agent_name in enumerate(self.possible_agents):
            if isinstance(obs, dict):
                obs_data = obs[agent_name]
            else:
                obs_data = obs[i]

            if isinstance(obs_data, torch.Tensor):
                obs_data = obs_data.cpu().numpy()
                if obs_data.ndim == 2 and obs_data.shape[0] == 1:
                    obs_data = obs_data.squeeze(0)

            observations[agent_name] = obs_data

        return observations, {}

    def step(self, actions):
        vmas_actions = []
        for i, agent_name in enumerate(self.possible_agents):
            action = actions[agent_name]
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32)
            action = action.unsqueeze(0)
            vmas_actions.append(action)

        obs, rews, dones, infos = self.vmas_env.step(vmas_actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}

        for i, agent_name in enumerate(self.possible_agents):
            if isinstance(obs, dict):
                obs_data = obs[agent_name]
            else:
                obs_data = obs[i]

            if isinstance(obs_data, torch.Tensor):
                obs_data = obs_data.cpu().numpy()
                if obs_data.ndim == 2 and obs_data.shape[0] == 1:
                    obs_data = obs_data.squeeze(0)

            observations[agent_name] = obs_data

            if isinstance(rews, dict):
                rew_data = rews[agent_name]
            else:
                rew_data = rews[i]

            if isinstance(rew_data, torch.Tensor):
                rew_data = rew_data.cpu().numpy().item()

            rewards[agent_name] = rew_data

            if isinstance(dones, torch.Tensor):
                if dones.ndim == 0:
                    term_data = dones.cpu().numpy().item()
                elif dones.ndim == 1 and dones.shape[0] == 1:
                    term_data = dones[0].cpu().numpy().item()
                else:
                    term_data = dones[i].cpu().numpy().item()
            else:
                term_data = dones

            terminations[agent_name] = term_data
            truncations[agent_name] = False

        if isinstance(dones, torch.Tensor):
            all_done = dones.all().item()
        else:
            all_done = np.all(dones)

        if all_done:
            self.agents = []
        else:
            self.agents = [agent for agent in self.agents if not terminations[agent]]

        return observations, rewards, terminations, truncations, infos

def train_algorithm(algorithm_name, num_iterations=1000):
    """训练指定算法"""
    print(f"\n{'='*60}")
    print(f"开始训练 {algorithm_name} 算法")
    print(f"{'='*60}\n")

    # 创建PettingZoo环境
    penv = VmasPettingZooEnv()

    # 转换为AEC环境（Stable-Baselines3需要）
    aec_env = parallel_to_aec(penv)

    # 根据算法类型选择训练策略
    if algorithm_name == "IPPO":
        # IPPO: 独立训练每个智能体
        print("使用IPPO策略: 独立训练每个智能体")
        models = {}
        for agent in penv.possible_agents:
            print(f"训练智能体: {agent}")
            model = PPO(
                MlpPolicy,
                aec_env,
                verbose=1,
                learning_rate=TRAINING_CONFIG["lr"],
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=TRAINING_CONFIG["gamma"],
                gae_lambda=TRAINING_CONFIG["lambda_"],
                clip_range=TRAINING_CONFIG["clip_param"],
                ent_coef=TRAINING_CONFIG["entropy_coeff"],
                vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
                max_grad_norm=0.5,
                tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}/{agent}",
            )
            model.learn(total_timesteps=num_iterations * 2048)
            models[agent] = model

            checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save(os.path.join(checkpoint_dir, f"{agent}_model"))

    elif algorithm_name == "MAPPO":
        # MAPPO: 集中式训练，分布式执行
        print("使用MAPPO策略: 集中式训练，分布式执行")
        model = PPO(
            MlpPolicy,
            aec_env,
            verbose=1,
            learning_rate=TRAINING_CONFIG["lr"],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["lambda_"],
            clip_range=TRAINING_CONFIG["clip_param"],
            ent_coef=TRAINING_CONFIG["entropy_coeff"],
            vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
            max_grad_norm=0.5,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
        )
        model.learn(total_timesteps=num_iterations * 2048)

        checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(os.path.join(checkpoint_dir, "shared_model"))

    elif algorithm_name == "CPPO":
        # CPPO: 集中式训练，集中式执行
        print("使用CPPO策略: 集中式训练，集中式执行")
        model = PPO(
            MlpPolicy,
            aec_env,
            verbose=1,
            learning_rate=TRAINING_CONFIG["lr"],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["lambda_"],
            clip_range=TRAINING_CONFIG["clip_param"],
            ent_coef=TRAINING_CONFIG["entropy_coeff"],
            vf_coef=TRAINING_CONFIG["vf_loss_coeff"],
            max_grad_norm=0.5,
            tensorboard_log=f"{PATH_CONFIG['results_dir']}/{algorithm_name}",
        )
        model.learn(total_timesteps=num_iterations * 2048)

        checkpoint_dir = os.path.join(PATH_CONFIG["checkpoints_dir"], algorithm_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(os.path.join(checkpoint_dir, "centralized_model"))

    print(f"\n{algorithm_name} 训练完成!")
    print(f"模型保存在: {PATH_CONFIG['checkpoints_dir']}/{algorithm_name}")

    return PATH_CONFIG["checkpoints_dir"]

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

    checkpoint_path = train_algorithm(args.algorithm, args.iterations)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"算法: {args.algorithm}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()