# Transport任务MARL算法复现

## 项目概述

本项目复现论文《VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning》中提到的三种多智能体强化学习（MARL）算法在Transport任务上的表现。

## 算法介绍

### 1. CPPO (Centralized PPO)
- **特点**: 集中式训练，集中式执行
- **适用**: 完全协作任务
- **优势**: 全局信息，性能最优
- **劣势**: 需要中心化控制器，可扩展性差

### 2. MAPPO (Multi-Agent PPO)
- **特点**: 集中式训练，分布式执行
- **适用**: 协作任务
- **优势**: 平衡性能和泛化
- **劣势**: 训练时需要全局信息

### 3. IPPO (Independent PPO)
- **特点**: 分布式训练，分布式执行
- **适用**: 通用多智能体任务
- **优势**: 可扩展性强
- **劣势**: 无全局信息，性能可能较差

## 目录结构

```
marl_algorithms/
├── configs/
│   └── transport_config.py    # 环境和训练配置
├── scripts/
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   └── compare.py              # 对比脚本
├── results/                    # 训练结果
├── checkpoints/                # 模型检查点
└── README.md                   # 本文件
```

## 环境配置

### 已安装依赖
- Python 3.11.2
- PyTorch 2.9.1+cpu
- Ray RLlib 2.6.3
- VMAS 1.5.2

### Transport环境参数
- **智能体数量**: 4
- **包裹数量**: 1
- **包裹质量**: 50
- **最大步数**: 500
- **并行环境数**: 32

## 使用方法

### 1. 训练算法

#### 训练CPPO
```bash
source /root/RL_Assignment/venv/bin/activate
python /root/RL_Assignment/marl_algorithms/scripts/train.py --algorithm CPPO --iterations 1000
```

#### 训练MAPPO
```bash
source /root/RL_Assignment/venv/bin/activate
python /root/RL_Assignment/marl_algorithms/scripts/train.py --algorithm MAPPO --iterations 1000
```

#### 训练IPPO
```bash
source /root/RL_Assignment/venv/bin/activate
python /root/RL_Assignment/marl_algorithms/scripts/train.py --algorithm IPPO --iterations 1000
```

### 2. 评估模型

```bash
source /root/RL_Assignment/venv/bin/activate
python /root/RL_Assignment/marl_algorithms/scripts/evaluate.py \
    --checkpoint /root/RL_Assignment/marl_algorithms/checkpoints/CPPO/checkpoint_xxx \
    --algorithm CPPO \
    --episodes 10
```

### 3. 对比算法

```bash
source /root/RL_Assignment/venv/bin/activate
python /root/RL_Assignment/marl_algorithms/scripts/compare.py
```

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 3e-4 | 优化器学习率 |
| 折扣因子 | 0.99 | 长期奖励折扣 |
| GAE参数 | 0.95 | 广义优势估计参数 |
| 裁剪参数 | 0.2 | PPO裁剪参数 |
| 批次大小 | 4000 | 训练批次大小 |
| SGD迭代次数 | 10 | 每个批次的SGD迭代次数 |
| 隐藏层 | [256, 256] | 网络隐藏层单元数 |

## 预期结果

根据论文，Transport任务的预期性能：

| 算法 | 平均奖励 | 成功率 | 收敛速度 |
|------|----------|--------|----------|
| CPPO | 最高 | >90% | 中等 |
| MAPPO | 高 | >85% | 快 |
| IPPO | 中等 | >75% | 慢 |

## 结果分析

### 性能对比
- **CPPO**: 性能最优，但需要中心化控制器
- **MAPPO**: 性能接近CPPO，更实用
- **IPPO**: 性能稍差，但可扩展性强

### 协作行为
- **CPPO/MAPPO**: 智能体之间有明确的协作
- **IPPO**: 智能体独立决策，协作较少

### 训练稳定性
- **CPPO**: 训练稳定，收敛快
- **MAPPO**: 训练较稳定
- **IPPO**: 训练不稳定，需要更多调参

## 常见问题

### Q1: 训练时间太长怎么办？
- 减少迭代次数（--iterations）
- 减少并行环境数（修改ENV_CONFIG）
- 使用GPU加速（修改device为"cuda"）

### Q2: 如何调整网络结构？
- 修改`transport_config.py`中的`model`配置
- 调整`fcnet_hiddens`参数

### Q3: 如何保存和加载模型？
- 模型自动保存在`checkpoints/`目录
- 使用`evaluate.py`加载模型进行评估

## 参考资料

- VMAS论文: https://arxiv.org/abs/2207.03530
- PPO论文: https://arxiv.org/abs/1707.06347
- MAPPO论文: https://arxiv.org/abs/2103.01955
- Ray RLlib文档: https://docs.ray.io/en/releases-2.6.3/rllib/