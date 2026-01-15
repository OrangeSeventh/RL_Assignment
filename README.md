# VMAS项目 - 多智能体强化学习研究与实现

## 项目概述

本项目包含两个主要任务：
1. **任务一**：为VMAS（Vectorized Multi-Agent Simulator）源代码添加关键步骤的中文注释
2. **任务二**：复现论文中提到的三种MARL算法（CPPO、MAPPO、IPPO）在Transport任务上的表现

---

## 任务一：VMAS代码注释说明

### 概述
为VMAS源代码添加了关键步骤的中文注释，帮助理解多智能体强化学习模拟器的核心实现。

### 已添加注释的文件

#### 1. `VectorizedMultiAgentSimulator/vmas/make_env.py`
**环境创建工厂函数**

##### `make_env()`
- **作用**: 创建VMAS环境的主要API入口
- **关键功能**:
  - 支持通过场景名称或场景类创建环境
  - 配置并行环境数量（num_envs）
  - 设置计算设备（CPU/GPU）
  - 支持连续动作和离散动作
  - 支持多种环境包装器（RLlib、Gym、Gymnasium等）
  - 支持可微分模拟（grad_enabled=True，梯度可流过模拟器）

---

#### 2. `VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py`
**VMAS环境主类**

##### `Environment`
- **作用**: 管理仿真循环、重置、步进等核心功能
- **关键功能**:
  - 向量化环境管理（批量并行模拟）
  - 环境重置（全部或单个环境）
  - 动作和观测空间配置
  - 支持多种渲染模式（human、rgb_array）
  - 管理随机种子和可重复性

##### `reset()`
- **作用**: 重置所有并行环境（向量化操作）
- **返回**: 所有环境和智能体的观测

##### `reset_at()`
- **作用**: 重置指定索引的单个环境
- **返回**: 该环境中所有智能体的观测

---

#### 3. `VectorizedMultiAgentSimulator/vmas/simulator/core.py`
**核心数据结构类**

##### `TorchVectorizedObject`
- **作用**: 向量化对象基类，所有实体和动作的父类
- **关键功能**:
  - 支持批量并行处理（batch_dim）
  - 支持GPU加速（device管理）
  - 提供张量设备迁移功能

##### `EntityState`
- **作用**: 物理实体状态类
- **管理状态**:
  - 位置坐标 (pos) - 形状: (batch_dim, 2)
  - 速度 (vel) - 形状: (batch_dim, 2)
  - 旋转角度 (rot) - 形状: (batch_dim, 1)，范围: -π 到 π
  - 角速度 (ang_vel) - 形状: (batch_dim, 1)

##### `AgentState`
- **作用**: 智能体状态类（继承自EntityState）
- **扩展状态**:
  - 通信信息 (c) - 形状: (batch_dim, dim_c)
  - 施加的力 (force) - 形状: (batch_dim, 2)
  - 施加的扭矩 (torque) - 形状: (batch_dim, 1)

##### `Action`
- **作用**: 智能体动作类
- **管理内容**:
  - 物理动作 (u) - 控制力和扭矩，形状: (batch_dim, action_size)
  - 通信动作 (c) - 智能体间通信，形状: (batch_dim, dim_c)
  - 动作范围（u_range）、动作乘数（u_multiplier）、动作噪声（u_noise）

##### `Entity`
- **作用**: 物理世界实体基类（智能体、地标、包裹等的父类）
- **关键属性**:
  - 形状（球体、盒子、线条等）
  - 质量、密度
  - 可移动性（movable）、可旋转性（rotatable）
  - 碰撞检测（collide）
  - 阻力（drag）和摩擦力（linear_friction、angular_friction）
  - 重力（gravity）

---

#### 4. `VectorizedMultiAgentSimulator/vmas/simulator/physics.py`
**物理引擎核心函数**

##### `_get_closest_box_box()`
- **作用**: 计算两个旋转盒子之间的最近点对
- **应用**: 碰撞检测和碰撞响应
- **核心原理**:
  - 将盒子边缘表示为线段
  - 计算所有线段对之间的最近点
  - 返回距离最小的点对
- **参数**:
  - box_pos, box2_pos: 盒子中心位置 (batch_dim, 2)
  - box_rot, box2_rot: 盒子旋转角度 (batch_dim, 1)
  - box_width, box2_width: 盒子宽度
  - box_length, box2_length: 盒子长度
- **返回**: closest_point_1, closest_point_2 - 两个盒子上最近的点

---

#### 5. `VectorizedMultiAgentSimulator/vmas/simulator/dynamics/common.py`
**动力学模型基类**

##### `Dynamics`
- **作用**: 定义智能体如何将动作转换为物理力/扭矩
- **抽象方法**:
  - `needed_action_size`: 返回所需的动作大小（子类必须实现）
  - `process_action`: 处理动作，将其转换为力/扭矩（子类必须实现）
- **关键方法**:
  - `check_and_process_action()`: 检查动作大小并处理

---

#### 6. `VectorizedMultiAgentSimulator/vmas/simulator/dynamics/holonomic.py`
**全向动力学模型**

##### `Holonomic`
- **作用**: 全向运动模型（智能体可在任意方向移动）
- **特点**:
  - 需要2个动作（x和y方向的力）
  - 直接将动作映射为力
- **needed_action_size**: 2
- **process_action**: `self.agent.state.force = self.agent.action.u[:, :2]`

---

#### 7. `VectorizedMultiAgentSimulator/vmas/simulator/controllers/velocity_controller.py`
**PID速度控制器**

##### `VelocityController`
- **作用**: 将目标速度转换为所需的力（PID控制）
- **两种形式**:
  - 标准形式: ctrl_params=[gain, intg_ts, derv_ts]
    - gain: 比例增益（kP）
    - intg_ts: 积分时间常数（误差容忍时间）
    - derv_ts: 微分时间常数（误差预测时间）
  - 并行形式: ctrl_params=[kP, kI, kD]
    - kP: 比例增益
    - kI: 积分增益
    - kD: 微分增益
- **功能**:
  - 实现比例-积分-微分（PID）控制
  - 支持积分器饱和限制（防止积分项过大）
  - 用于精确速度跟踪

---

#### 8. `VectorizedMultiAgentSimulator/vmas/simulator/sensors.py`
**传感器系统**

##### `Sensor`
- **作用**: 传感器基类，定义智能体感知环境的接口
- **抽象方法**:
  - `measure()`: 测量环境并返回观测值（子类必须实现）
  - `render()`: 渲染传感器可视化（子类必须实现）

##### `Lidar`
- **作用**: LIDAR传感器，通过射线检测障碍物距离
- **功能**:
  - 发射多条射线（n_rays）检测障碍物
  - 返回每条射线到最近障碍物的距离
  - 支持实体过滤（entity_filter）
  - 支持可视化渲染
- **参数**:
  - angle_start, angle_end: 射线角度范围
  - n_rays: 射线数量
  - max_range: 最大检测范围
  - entity_filter: 实体过滤器函数

---

#### 9. `VectorizedMultiAgentSimulator/vmas/scenarios/transport.py`
**Transport任务场景实现**

##### `Scenario.make_world()`
- **作用**: 创建世界并初始化智能体、包裹和目标
- **初始化内容**:
  - 创建World对象（设置边界）
  - 添加智能体（球体，可推动包裹）
  - 添加目标地标（绿色球体，不可碰撞）
  - 添加包裹（红色盒子，可移动，可碰撞）
- **关键参数**:
  - n_agents: 智能体数量（默认4）
  - n_packages: 包裹数量（默认1）
  - package_mass: 包裹质量（默认50）

##### `Scenario.reward()`
- **作用**: 计算智能体的奖励
- **奖励机制**:
  - 基于包裹到目标的距离
  - 使用奖励塑形：奖励 = 上一时刻距离 - 当前时刻距离
  - 鼓励包裹向目标移动
  - 包裹到达目标时颜色变为绿色
- **公式**: `reward = global_shaping - current_shaping`

##### `Scenario.observation()`
- **作用**: 构建智能体的观测空间
- **观测内容**:
  - 智能体自身的位置和速度
  - 每个包裹相对于目标的位置
  - 每个包裹相对于智能体的位置
  - 每个包裹的速度
  - 每个包裹是否在目标上

---

### 核心设计理念

#### 1. 向量化处理
- 所有状态和动作都使用PyTorch张量
- 支持批量并行模拟（num_envs，可达30,000+）
- 可在GPU上高效运行，性能比传统方法快100倍以上

#### 2. 模块化设计
- 基类继承结构清晰（TorchVectorizedObject、Entity、Agent等）
- 动力学模型可插拔（Holonomic、DiffDrive、Drone、KinematicBicycle等）
- 传感器系统可扩展（Lidar等）
- 场景通过继承BaseScenario实现

#### 3. 物理引擎
- 自定义2D物理引擎（physics.py）
- 支持刚体碰撞、旋转、关节等
- 可微分（梯度可流过模拟器，grad_enabled=True）

#### 4. 控制系统
- 支持直接力控制和速度控制
- PID控制器实现精确速度跟踪
- 支持多种动力学模型

#### 5. 传感器系统
- LIDAR传感器实现障碍物检测
- 可配置射线数量、角度、范围
- 支持实体过滤和可视化

#### 6. 协作任务设计
- Transport任务：多智能体协作推动包裹
- 奖励塑形引导学习
- 观测空间包含相对位置信息
- 支持通信机制

---

### 使用建议

#### 学习路径

1. **理解核心类**: 从TorchVectorizedObject和Entity开始，理解向量化对象的设计
2. **研究物理引擎**: physics.py中的碰撞检测算法是关键
3. **分析场景实现**: transport.py展示了如何创建一个协作任务
4. **了解环境创建**: make_env.py是创建环境的主要入口
5. **研究环境管理**: Environment类管理仿真循环和状态
6. **探索动力学模型**: dynamics/目录包含多种运动模型
7. **学习控制系统**: VelocityController实现PID速度控制
8. **使用传感器**: Lidar传感器实现障碍物检测
9. **扩展场景**: 参考transport.py实现自己的任务场景

#### 实践建议

- 先运行transport场景，观察智能体行为
- 修改奖励函数，理解奖励塑形的作用
- 尝试添加新的传感器或动力学模型
- 创建自己的协作任务场景

---

### 下一步学习方向

1. **查看其他场景实现**:
   - wheel.py: 轮子控制任务
   - balance.py: 平衡任务
   - football.py: 足球游戏
   - flocking.py: 群体聚集

2. **研究更多动力学模型**:
   - DiffDrive: 差分驱动模型
   - Drone: 无人机模型
   - KinematicBicycle: 运动学自行车模型

3. **了解环境包装器**:
   - RLlib: Ray RLlib集成
   - Gym: OpenAI Gym兼容
   - Gymnasium: Gymnasium兼容

4. **探索其他系统**:
   - 渲染系统（rendering.py）
   - 关节系统（joints.py）
   - 启发式策略（heuristic_policy.py）
   - 示例代码（examples/目录）

5. **阅读教程**:
   - notebooks/VMAS_Use_vmas_environment.ipynb
   - notebooks/VMAS_RLlib.ipynb
   - notebooks/Simulation_and_training_in_VMAS_and_BenchMARL.ipynb

---

### 参考资料

- **VMAS论文**: https://arxiv.org/abs/2207.03530
- **官方仓库**: https://github.com/proroklab/VectorizedMultiAgentSimulator
- **项目文档**: https://proroklab.github.io/VectorizedMultiAgentSimulator/

---

### 注释说明

本项目的注释遵循以下原则：
- 简洁明了，突出关键概念
- 使用中文注释，便于理解
- 注释关键数据结构和算法
- 说明参数、返回值和核心逻辑
- 适合初学者和研究者阅读

---

## 任务二：MARL算法复现

### 任务目标
选取论文中提到的三个任务场景(Transport, Wheel, Balance)的其中一个任务场景，实现论文中提到的CPPO、MAPPO、IPPO三种MARL算法，尝试复现论文中的结果。

**当前选择场景**: Transport（运输任务）

### 环境适配方案

经过对多种MARL框架的测试和评估，当前项目采用以下环境配置：

#### 方案选择过程

1. **Ray RLlib** (已测试，不兼容)
   - 问题：VMAS的RLLib包装器是为旧版本Ray设计的（要求ray[rllib]<=2.2）
   - Ray 2.3.0与Python 3.11不兼容
   - Ray 2.5.0+的API变化导致VMAS的VectorEnvWrapper不是MultiAgentEnv的子类
   - 即使使用Python 3.10，Ray RLlib与新版本VMAS仍存在API不兼容问题
   - 结论：兼容性问题严重，不适合当前任务

2. **VMAS原生Gymnasium包装器 + Stable-Baselines3** (当前方案，推荐)
   - 优势：
     - VMAS原生支持Gymnasium包装器（`wrapper="gymnasium"`）
     - Stable-Baselines3提供稳定、高效的PPO实现
     - 完全兼容Python 3.10
     - 无需自定义包装器，直接使用VMAS提供的接口
     - 支持向量化训练，提升训练效率
   - 实现方式：使用`make_env()`时指定`wrapper="gymnasium"`
   - 结论：兼容性最好，性能优秀，推荐使用

3. **Conda环境** (当前配置)
   - Python 3.10 (更稳定，兼容性更好)
   - PyTorch 2.9.1+cpu
   - Stable-Baselines3 2.7.1
   - Gymnasium 1.2.3
   - VMAS 1.5.2 (本地版本，已添加注释)

### 项目结构

```
marl_algorithms/
├── configs/
│   └── transport_config.py    # 环境和训练配置
├── scripts/
│   ├── train_vmas_gymnasium.py # VMAS Gymnasium训练脚本（当前使用，推荐）
│   ├── train_pettingzoo.py     # PettingZoo训练脚本（已废弃）
│   ├── train.py                # Ray RLlib训练脚本（已废弃，不兼容）
│   ├── train_vmas.py           # VMAS官方示例训练脚本（参考）
│   ├── train_new.py            # 临时测试脚本（已废弃）
│   ├── evaluate.py             # 评估脚本（待实现）
│   └── compare.py              # 对比脚本（待实现）
├── results/                    # 训练结果和TensorBoard日志
├── checkpoints/                # 模型检查点
└── README.md                   # 使用说明
```

### 算法介绍

#### 1. CPPO (Centralized PPO)
- **特点**: 集中式训练，集中式执行
- **适用**: 完全协作任务
- **优势**: 全局信息，性能最优
- **劣势**: 需要中心化控制器，可扩展性差

#### 2. MAPPO (Multi-Agent PPO)
- **特点**: 集中式训练，分布式执行
- **适用**: 协作任务
- **优势**: 平衡性能和泛化
- **劣势**: 训练时需要全局信息

#### 3. IPPO (Independent PPO)
- **特点**: 分布式训练，分布式执行
- **适用**: 通用多智能体任务
- **优势**: 可扩展性强
- **劣势**: 无全局信息，性能可能较差

### 环境配置

#### Conda环境（当前配置）
```bash
# 创建环境
conda create -n rl_assignment python=3.10 -y

# 激活环境
conda activate rl_assignment

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium stable-baselines3 tensorboard matplotlib pandas pyglet

# 安装VMAS（本地版本，已添加注释）
pip install -e /root/RL_Assignment/VectorizedMultiAgentSimulator
```

#### Transport环境参数
- **智能体数量**: 4
- **包裹数量**: 1
- **包裹质量**: 50
- **最大步数**: 500
- **并行环境数**: 1（Stable-Baselines3训练时）

### 使用方法

#### 1. 激活环境
```bash
conda activate rl_assignment
```

#### 2. 测试Transport环境
```bash
python /root/RL_Assignment/test_transport.py
```

#### 3. 训练算法

**训练MAPPO**（推荐）:
```bash
python /root/RL_Assignment/marl_algorithms/scripts/train_vmas_gymnasium.py --algorithm MAPPO --iterations 1000
```

**训练CPPO**:
```bash
python /root/RL_Assignment/marl_algorithms/scripts/train_vmas_gymnasium.py --algorithm CPPO --iterations 1000
```

**训练IPPO**:
```bash
python /root/RL_Assignment/marl_algorithms/scripts/train_vmas_gymnasium.py --algorithm IPPO --iterations 1000
```

#### 4. 查看训练结果
```bash
# 使用TensorBoard查看训练曲线
tensorboard --logdir /root/RL_Assignment/marl_algorithms/results/
```

#### 5. 评估模型（待实现）
```bash
python /root/RL_Assignment/marl_algorithms/scripts/evaluate.py \
    --checkpoint <checkpoint_path> \
    --algorithm <algorithm_name> \
    --episodes 10
```

#### 6. 对比算法（待实现）
```bash
python /root/RL_Assignment/marl_algorithms/scripts/compare.py
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 3e-4 | 优化器学习率 |
| 折扣因子 | 0.99 | 长期奖励折扣 |
| GAE参数 | 0.95 | 广义优势估计参数 |
| 裁剪参数 | 0.2 | PPO裁剪参数 |
| 批次大小 | 2048 | 训练批次大小 |
| SGD迭代次数 | 10 | 每个批次的SGD迭代次数 |
| 网络隐藏层 | [256, 256] | 网络隐藏层单元数 |

### 预期结果

根据论文，Transport任务的预期性能：

| 算法 | 平均奖励 | 成功率 | 收敛速度 |
|------|----------|--------|----------|
| CPPO | 85-95 | >90% | 中等 |
| MAPPO | 80-90 | >85% | 快 |
| IPPO | 60-75 | >75% | 慢 |

### 当前进度

- ✅ 完成VMAS代码注释（任务一）
- ✅ 评估多种MARL框架（Ray RLlib, PettingZoo, Stable-Baselines3）
- ✅ 创建Python 3.10 Conda环境
- ✅ 安装所有必要依赖（PyTorch, Stable-Baselines3, Gymnasium, VMAS）
- ✅ 完成Transport环境测试
- ✅ 创建VMAS原生Gymnasium包装器训练脚本
- 🔄 正在进行：使用VMAS Gymnasium + Stable-Baselines3训练MARL算法
- ⏳ 待完成：训练CPPO、MAPPO、IPPO三种算法并对比性能
- ⏳ 待完成：性能对比和结果分析

### 实验报告

详细的实验报告请查看：[EXPERIMENT_REPORT.md](./EXPERIMENT_REPORT.md)

报告包含：
- 算法详细介绍
- 实验设置和参数
- 预期结果和性能对比
- 算法优缺点分析
- 实际应用价值
- 结论和未来工作

### 环境配置说明

详细的环境配置说明请查看：[ENV_SETUP.md](./ENV_SETUP.md)

包含：
- 已安装的依赖包
- Transport环境参数
- Conda环境配置步骤
- 常见问题解答

---

## 项目文件说明

### 主要文件

- `README.md` - 本文件，项目总览
- `EXPERIMENT_REPORT.md` - 任务二实验报告
- `ENV_SETUP.md` - 环境配置说明
- `test_transport.py` - Transport环境测试脚本
- `test_pettingzoo_env.py` - PettingZoo环境包装器测试脚本
- `run_env.sh` - 环境启动脚本
- `venv/` - Python虚拟环境（已废弃，建议使用conda）

### 目录结构

- `VectorizedMultiAgentSimulator/` - VMAS源代码（已添加注释）
- `marl_algorithms/` - MARL算法实现
  - `configs/` - 配置文件
  - `scripts/` - 训练和评估脚本
  - `results/` - 训练结果
  - `checkpoints/` - 模型检查点
- `venv/` - Python虚拟环境（已废弃）

---

## 快速开始

### 1. 配置环境

**使用Conda环境（推荐）**:
```bash
# 创建环境
conda create -n rl_assignment python=3.10 -y

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl_assignment

# 安装依赖
pip install torch numpy pandas matplotlib tensorboard
pip install pettingzoo stable-baselines3
```

### 2. 测试Transport环境
```bash
python /root/RL_Assignment/test_transport.py
```

### 3. 测试PettingZoo环境包装器
```bash
python /root/RL_Assignment/test_pettingzoo_env.py
```

### 4. 训练算法
```bash
# 训练MAPPO（推荐）
conda activate rl_assignment
python /root/RL_Assignment/marl_algorithms/scripts/train_pettingzoo.py --algorithm MAPPO --iterations 1000
```

### 5. 评估模型（待实现）
```bash
python /root/RL_Assignment/marl_algorithms/scripts/evaluate.py \
    --checkpoint <checkpoint_path> \
    --algorithm MAPPO \
    --episodes 10
```

### 6. 对比算法（待实现）
```bash
python /root/RL_Assignment/marl_algorithms/scripts/compare.py
```

---

## 参考资料

### 论文
- **VMAS论文**: https://arxiv.org/abs/2207.03530
- **PPO论文**: https://arxiv.org/abs/1707.06347
- **MAPPO论文**: https://arxiv.org/abs/2103.01955

### 代码库
- **VMAS官方仓库**: https://github.com/proroklab/VectorizedMultiAgentSimulator
- **Ray RLlib文档**: https://docs.ray.io/en/releases-2.6.3/rllib/

---

## 项目状态

- ✅ 任务一：VMAS代码注释完成
- 🔄 任务二：MARL算法复现进行中
  - ✅ 评估多种MARL框架
  - ✅ 创建PettingZoo环境包装器
  - ✅ 完成环境测试
  - 🔄 正在实现CPPO、MAPPO、IPPO三种算法
- ⏳ 任务三：算法改进（待实施）

---

## 许可证

本项目遵循VMAS的GPLv3许可证。