"""
Transport任务配置文件
定义VMAS Transport环境的参数和训练配置
"""

import sys
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

# 环境配置
ENV_CONFIG = {
    "scenario": "transport",
    "num_envs": 32,  # 并行环境数量
    "device": "cpu",  # 计算设备
    "continuous_actions": True,  # 连续动作
    "max_steps": 500,  # 最大步数
    "dict_spaces": False,  # 不使用字典空间
    # Transport特定参数
    "n_agents": 4,  # 智能体数量
    "n_packages": 1,  # 包裹数量
    "package_width": 0.15,  # 包裹宽度
    "package_length": 0.15,  # 包裹长度
    "package_mass": 50,  # 包裹质量
}

# 训练配置
TRAINING_CONFIG = {
    # 基础训练参数
    "num_iterations": 1000,  # 训练迭代次数
    "train_batch_size": 4000,  # 训练批次大小
    "batch_mode": "complete_episodes",  # 批次模式
    "num_cpus_per_worker": 1,  # 每个worker的CPU数
    "num_gpus_per_worker": 0,  # 每个worker的GPU数

    # PPO参数
    "lr": 3e-4,  # 学习率
    "gamma": 0.99,  # 折扣因子
    "lambda_": 0.95,  # GAE参数
    "clip_param": 0.2,  # PPO裁剪参数
    "vf_loss_coeff": 0.5,  # 价值函数损失系数
    "entropy_coeff": 0.01,  # 熵系数
    "sgd_minibatch_size": 128,  # SGD小批次大小
    "num_sgd_iter": 10,  # SGD迭代次数
    "ppo_epochs": 10,  # PPO更新轮数
    "batch_size": 64,  # 批次大小

    # 网络配置
    "model": {
        "fcnet_hiddens": [256, 256],  # 全连接层隐藏单元
        "fcnet_activation": "tanh",  # 激活函数
        "vf_share_layers": True,  # 共享价值网络层
    },

    # 评估配置
    "evaluation_interval": 20,  # 评估间隔
    "evaluation_num_episodes": 10,  # 评估回合数
    "evaluation_config": {
        "explore": False,  # 评估时不探索
    },
}

# 算法特定配置
ALGORITHM_CONFIGS = {
    # CPPO: Centralized PPO - 集中式训练，集中式执行
    "CPPO": {
        "multiagent": {
            "policies": {
                "shared_policy": (None, None, None, {"gamma": 0.99}),
            },
            "policy_mapping_fn": (lambda agent_id: "shared_policy"),
        },
    },
    
    # MAPPO: Multi-Agent PPO - 集中式训练，分布式执行
    "MAPPO": {
        "multiagent": {
            "policies": {
                "shared_policy": (None, None, None, {"gamma": 0.99}),
            },
            "policy_mapping_fn": (lambda agent_id: "shared_policy"),
        },
        "framework": "torch",
    },
    
    # IPPO: Independent PPO - 分布式训练，分布式执行
    "IPPO": {
        "multiagent": {
            "policies": {
                "agent_0": (None, None, None, {"gamma": 0.99}),
                "agent_1": (None, None, None, {"gamma": 0.99}),
                "agent_2": (None, None, None, {"gamma": 0.99}),
                "agent_3": (None, None, None, {"gamma": 0.99}),
            },
            "policy_mapping_fn": (lambda agent_id: agent_id),
        },
    },
}

# 保存路径配置
PATH_CONFIG = {
    "base_dir": "/root/RL_Assignment/marl_algorithms",
    "results_dir": "/root/RL_Assignment/marl_algorithms/results",
    "checkpoints_dir": "/root/RL_Assignment/marl_algorithms/checkpoints",
}

# 日志配置
LOG_CONFIG = {
    "log_level": "INFO",  # 日志级别
    "log_to_file": True,  # 是否记录到文件
    "log_dir": "/root/RL_Assignment/marl_algorithms/logs",
}