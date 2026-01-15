#!/usr/bin/env python3
"""测试VMAS环境以获取observation_space和action_space"""

import sys
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from vmas import make_env
from vmas.simulator.environment import Wrapper

env = make_env(
    scenario="transport",
    num_envs=1,
    device="cpu",
    continuous_actions=True,
    wrapper=Wrapper.RLLIB,
    max_steps=500,
    n_agents=4,
    n_packages=1,
    package_width=0.15,
    package_length=0.15,
    package_mass=50,
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Number of agents: {env.n_agents}")
print(f"Agent IDs: {env.agents}")