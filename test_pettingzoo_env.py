#!/usr/bin/env python3
"""测试PettingZoo环境包装器"""

import sys
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')
sys.path.insert(0, '/root/RL_Assignment/marl_algorithms/scripts')

from train_pettingzoo import VmasPettingZooEnv

print("创建PettingZoo环境...")
env = VmasPettingZooEnv()

print(f"智能体列表: {env.possible_agents}")
print(f"观测空间: {env.observation_space('agent_0')}")
print(f"动作空间: {env.action_space('agent_0')}")

print("\n测试环境重置...")
obs, info = env.reset()
print(f"重置后的观测: {list(obs.keys())}")
print(f"agent_0的观测形状: {obs['agent_0'].shape}")

print("\n测试环境步进...")
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
print(f"动作: {actions}")

obs, rewards, terminations, truncations, infos = env.step(actions)
print(f"步进后的观测: {list(obs.keys())}")
print(f"奖励: {rewards}")
print(f"终止: {terminations}")

print("\n测试完成!")