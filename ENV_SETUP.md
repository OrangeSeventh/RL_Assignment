# Transport任务环境配置说明

## 环境信息

- **Python版本**: 3.11.2
- **虚拟环境路径**: `/root/RL_Assignment/venv`
- **VMAS路径**: `/root/RL_Assignment/VectorizedMultiAgentSimulator`

## 已安装的依赖包

### 核心依赖
- `numpy==2.4.1` - 数值计算库
- `torch==2.9.1+cpu` - PyTorch深度学习框架（CPU版本）
- `gym==0.26.2` - OpenAI Gym环境接口
- `gym-notices==0.0.8` - Gym通知
- `pyglet==1.5.27` - 渲染库
- `six==1.17.0` - Python 2/3兼容库
- `vmas==1.5.2` - VMAS多智能体模拟器（开发模式）

### 依赖说明
- **PyTorch**: 使用CPU版本（适合学习和开发）
- **pyglet**: 用于可视化渲染（版本限制<=1.5.27）
- **gym**: 提供标准的环境接口

## 使用方法

### 激活虚拟环境
```bash
source /root/RL_Assignment/venv/bin/activate
```

### 运行测试脚本
```bash
/root/RL_Assignment/venv/bin/python /root/RL_Assignment/test_transport.py
```

### 在Python中使用
```python
import sys
sys.path.insert(0, '/root/RL_Assignment/VectorizedMultiAgentSimulator')

from vmas import make_env

# 创建Transport环境
env = make_env(
    scenario="transport",
    num_envs=4,
    device="cpu",
    continuous_actions=True,
    n_agents=4,
    n_packages=1,
)

# 重置环境
obs = env.reset()

# 运行步骤
actions = [env.get_random_action(agent) for agent in env.agents]
obs, rews, dones, info = env.step(actions)
```

## Transport环境参数

### 必需参数
- `scenario`: 场景名称（"transport"）
- `num_envs`: 并行环境数量（推荐：32-128）
- `device`: 计算设备（"cpu"或"cuda"）
- `continuous_actions`: 是否使用连续动作（True/False）

### 场景特定参数
- `n_agents`: 智能体数量（默认：4）
- `n_packages`: 包裹数量（默认：1）
- `package_width`: 包裹宽度（默认：0.15）
- `package_length`: 包裹长度（默认：0.15）
- `package_mass`: 包裹质量（默认：50）

### 环境输出
- **观测**: 每个智能体的观测（形状：[num_envs, obs_dim]）
- **奖励**: 每个智能体的奖励（形状：[num_envs]）
- **完成**: 是否完成任务（形状：[num_envs]）
- **信息**: 额外信息（字典）

## 下一步：实现MARL算法

### 需要安装的额外依赖

#### 选项1：使用Ray RLlib（推荐）
```bash
source /root/RL_Assignment/venv/bin/activate
pip install "ray[rllib]<=2.2"
```

#### 选项2：使用BenchMARL
```bash
source /root/RL_Assignment/venv/bin/activate
pip install benchmarl
```

#### 选项3：手动实现
需要安装：
- `stable-baselines3` - PPO实现
- `torch` - 已安装
- `numpy` - 已安装

### 推荐实现方案

#### 方案A：使用Ray RLlib（最简单）
- 优点：开箱即用，支持CPPO、MAPPO、IPPO
- 缺点：依赖较多
- 适用：快速复现论文结果

#### 方案B：使用BenchMARL（推荐）
- 优点：专门为多智能体RL设计，易于对比
- 缺点：学习曲线稍陡
- 适用：深入研究多智能体RL

#### 方案C：手动实现（学习价值高）
- 优点：深入理解算法细节
- 缺点：实现复杂，容易出错
- 适用：学习和研究

## 论文中需要复现的算法

### 1. CPPO (Centralized PPO)
- **特点**: 集中式训练，集中式执行
- **适用**: 完全协作任务
- **优势**: 全局信息，性能最优

### 2. MAPPO (Multi-Agent PPO)
- **特点**: 集中式训练，分布式执行
- **适用**: 协作任务
- **优势**: 平衡性能和泛化

### 3. IPPO (Independent PPO)
- **特点**: 分布式训练，分布式执行
- **适用**: 通用多智能体任务
- **优势**: 可扩展性强

## 预期结果

根据论文，Transport任务的性能指标：
- **任务完成率**: 应达到90%+
- **平均奖励**: 应接近最优值
- **收敛速度**: MAPPO > CPPO > IPPO

## 常见问题

### Q1: 如何切换到GPU？
```python
env = make_env(
    scenario="transport",
    num_envs=32,
    device="cuda",  # 改为cuda
    ...
)
```

### Q2: 如何保存和加载模型？
```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model.load_state_dict(torch.load("model.pth"))
```

### Q3: 如何可视化？
```python
# 渲染一帧
frame = env.render(mode="rgb_array")

# 保存视频
from vmas.simulator.utils import save_video
save_video("transport", frame_list, fps=30)
```

## 参考资料

- VMAS论文: https://arxiv.org/abs/2207.03530
- VMAS文档: https://proroklab.github.io/VectorizedMultiAgentSimulator/
- PPO论文: https://arxiv.org/abs/1707.06347
- MAPPO论文: https://arxiv.org/abs/2103.01955