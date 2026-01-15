#!/usr/bin/env python3
"""简化版训练脚本用于测试"""

import sys
import os

# 添加VMAS路径
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/configs')

import transport_config as config_module
from ray.rllib.algorithms.ppo import PPOConfig
import ray

ENV_CONFIG = config_module.ENV_CONFIG
TRAINING_CONFIG = config_module.TRAINING_CONFIG
ALGORITHM_CONFIGS = config_module.ALGORITHM_CONFIGS
PATH_CONFIG = config_module.PATH_CONFIG

print("正在初始化Ray...")
ray.init(ignore_reinit_error=True, log_to_driver=False)

print("正在创建PPO配置...")
config = PPOConfig()

print("配置创建成功！")
print("正在关闭Ray...")
ray.shutdown()

print("测试完成！")