# 算法改进使用指南

## 概述

本指南介绍如何使用改进后的训练脚本来提升CPPO、MAPPO、IPPO三种算法在Transport任务上的性能。

## 改进内容

### 1. 动态熵系数调整
- **原始**：固定熵系数0.01
- **改进**：从0.01线性衰减到0.001
- **效果**：早期鼓励探索，后期鼓励利用，提高稳定性

### 2. 调整GAE参数
- **原始**：GAE参数λ=0.95
- **改进**：GAE参数λ=0.97
- **效果**：减少优势函数的方差，提高训练稳定性

### 3. 降低学习率
- **原始**：学习率3e-4
- **改进**：学习率2e-4
- **效果**：提高训练稳定性，减少策略崩溃

### 4. 增加训练迭代次数
- **原始**：300次迭代
- **改进**：1000次迭代（默认）
- **效果**：给算法充分的时间收敛和稳定

### 5. 自动检查点保存
- **新增**：每200次迭代自动保存检查点
- **效果**：防止训练中断导致的数据丢失，便于选择最佳模型

## 使用方法

### 1. 快速测试（推荐先运行）

在完整训练之前，建议先运行快速测试验证改进效果：

```bash
# 激活环境
source /root/RL_Assignment/venv/bin/activate

# 快速测试MAPPO（仅训练50次迭代）
python /root/RL_Assignment/marl_algorithms/scripts/quick_test.py \
    --algorithm MAPPO \
    --iterations 50

# 快速测试CPPO
python /root/RL_Assignment/marl_algorithms/scripts/quick_test.py \
    --algorithm CPPO \
    --iterations 50

# 快速测试IPPO
python /root/RL_Assignment/marl_algorithms/scripts/quick_test.py \
    --algorithm IPPO \
    --iterations 50
```

**预期结果**：
- 如果改进有效，改进配置的最终奖励和最高奖励应该优于原始配置
- 如果改进效果不明显，可以尝试调整参数

### 2. 完整训练

快速测试验证改进有效后，进行完整训练：

```bash
# 训练改进的MAPPO（1000次迭代）
python /root/RL_Assignment/marl_algorithms/scripts/train_improved.py \
    --algorithm MAPPO \
    --iterations 1000

# 训练改进的CPPO
python /root/RL_Assignment/marl_algorithms/scripts/train_improved.py \
    --algorithm CPPO \
    --iterations 1000

# 训练改进的IPPO
python /root/RL_Assignment/marl_algorithms/scripts/train_improved.py \
    --algorithm IPPO \
    --iterations 1000
```

**训练时间估计**：
- MAPPO：约40-50分钟
- CPPO：约40-50分钟
- IPPO：约40-50分钟

### 3. 对比评估

训练完成后，对比原始算法和改进算法的性能：

```bash
# 对比所有算法
python /root/RL_Assignment/marl_algorithms/scripts/compare_improvements.py \
    --algorithms CPPO MAPPO IPPO \
    --episodes 10

# 仅对比MAPPO
python /root/RL_Assignment/marl_algorithms/scripts/compare_improvements.py \
    --algorithms MAPPO \
    --episodes 10
```

**输出**：
- 控制台显示各算法的平均奖励和标准差
- 生成JSON格式的详细结果文件
- 生成性能对比柱状图

## 文件结构

训练完成后，文件结构如下：

```
marl_algorithms/
├── checkpoints/
│   ├── MAPPO/
│   │   ├── MAPPO_model_0_steps.zip
│   │   ├── MAPPO_model_409600_steps.zip
│   │   ├── MAPPO_model_819200_steps.zip
│   │   └── MAPPO_final_model.zip
│   ├── CPPO/
│   │   └── ...
│   └── IPPO/
│       └── ...
├── results/
│   ├── MAPPO/
│   │   ├── MAPPO_metrics.json
│   │   └── MAPPO_transport_*.json
│   ├── CPPO/
│   │   └── ...
│   ├── IPPO/
│   │   └── ...
│   ├── comparison_results_*.json
│   └── comparison_plot_*.png
└── scripts/
    ├── train_improved.py
    ├── quick_test.py
    └── compare_improvements.py
```

## 预期改进效果

根据实验报告的分析，预期改进效果如下：

### CPPO
- **峰值奖励**：0.3457 → 0.4000（+15.7%）
- **最终奖励**：-0.1356 → 0.2000（+247.6%）
- **稳定性**：大幅改善，波动减少80%

### MAPPO
- **峰值奖励**：0.2976 → 0.3500（+17.6%）
- **最终奖励**：0.0381 → 0.1500（+293.7%）
- **收敛速度**：提升30%

### IPPO
- **峰值奖励**：0.1458 → 0.2000（+37.2%）
- **最终奖励**：-0.0477 → 0.0500（+204.8%）
- **协作能力**：显著提升

## 监控训练进度

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir /root/RL_Assignment/marl_algorithms/results

# 在浏览器中打开
# http://localhost:6006
```

**查看指标**：
- `rollout/ep_rew_mean`：平均奖励
- `train/learning_rate`：学习率
- `train/ent_coef`：熵系数（应该看到线性衰减）
- `train/policy_loss`：策略损失
- `train/value_loss`：价值损失

### 查看日志

训练脚本会输出详细的日志信息：
- 当前训练进度
- 熵系数变化
- 奖励变化
- 模型保存信息

## 常见问题

### Q1: 快速测试显示改进效果不明显怎么办？

**A**: 可以尝试调整以下参数：
1. 进一步降低学习率（如1e-4）
2. 调整熵系数衰减速度
3. 增加快速测试的迭代次数（如100次）

### Q2: 训练过程中策略崩溃怎么办？

**A**: 可能的解决方案：
1. 降低学习率（如1e-4）
2. 增加GAE参数（如0.98）
3. 增加梯度裁剪（如1.0）
4. 增加批次大小（如128）

### Q3: 训练时间太长怎么办？

**A**: 可以采取以下措施：
1. 减少训练迭代次数（如500次）
2. 增加并行环境数（需要修改代码）
3. 使用GPU加速（需要修改代码）

### Q4: 如何选择最佳检查点？

**A**: 建议选择：
1. 验证集上性能最好的检查点
2. 或者选择训练后期的检查点（如800-1000次迭代）
3. 可以使用对比评估脚本来评估每个检查点

## 进阶改进

如果基础改进效果不理想，可以尝试以下进阶改进：

### 1. 学习率调度
实现余弦退火或指数衰减的学习率调度。

### 2. 价值函数集成
使用多个价值函数进行集成，提高估计准确性。

### 3. 注意力机制
在MAPPO的Critic中添加注意力机制。

### 4. 通信机制
在IPPO中添加显式通信机制。

### 5. 角色自适应
实现角色自适应的策略网络。

这些进阶改进需要修改训练脚本的代码，具体实现请参考实验报告第7章。

## 参考文献

1. Bettini, M., et al. "VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning." arXiv:2207.03530
2. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347
3. Yu, C., et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." arXiv:2103.01955

## 联系支持

如果在使用过程中遇到问题，请：
1. 检查日志文件
2. 查看TensorBoard可视化
3. 参考实验报告的详细分析
4. 尝试调整超参数

---

**最后更新**：2026年1月15日
**版本**：1.0
**作者**：陈俊帆