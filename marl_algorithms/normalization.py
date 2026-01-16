#!/usr/bin/env python3
"""
观测归一化Wrapper - 简单实现
使用运行均值和方差对观测进行归一化
"""

import torch
import numpy as np


class RunningMeanStd:
    """运行均值和方差计算器"""

    def __init__(self, shape, epsilon=1e-8):
        self.shape = shape
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x):
        """更新运行统计量"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class NormalizeObservation:
    """观测归一化Wrapper"""

    def __init__(self, obs_dim, clip_range=10.0, pre_collect_steps=0):
        """
        Args:
            obs_dim: 观测维度
            clip_range: 归一化后的值裁剪范围
            pre_collect_steps: 预收集数据的步数（用于初始化统计量）
        """
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.running_stats = RunningMeanStd(obs_dim)
        self.initialized = False
        self.pre_collect_steps = pre_collect_steps
        self.collected_steps = 0

    def pre_collect(self, obs):
        """
        预收集数据用于初始化统计量
        Args:
            obs: 观测张量，shape为 [batch_size, obs_dim]
        """
        self.running_stats.update(obs)
        self.collected_steps += 1

    def is_pre_collection_done(self):
        """检查预收集是否完成"""
        return self.collected_steps >= self.pre_collect_steps

    def finalize_pre_collection(self):
        """完成预收集，标记为已初始化"""
        if self.collected_steps > 0:
            self.initialized = True
            print(f"归一化统计量初始化完成，已收集 {self.collected_steps} 步数据")
            print(f"  均值范围: [{self.running_stats.mean.min():.4f}, {self.running_stats.mean.max():.4f}]")
            print(f"  方差范围: [{self.running_stats.var.min():.4f}, {self.running_stats.var.max():.4f}]")

    def normalize(self, obs, update_stats=True):
        """
        归一化观测
        Args:
            obs: 观测张量，shape为 [batch_size, obs_dim]
            update_stats: 是否更新统计量（预收集阶段设为False）
        Returns:
            归一化后的观测
        """
        # 如果还在预收集阶段，只更新不归一化
        if not self.initialized:
            if update_stats:
                self.running_stats.update(obs)
            return obs

        # 更新运行统计量
        if update_stats:
            self.running_stats.update(obs)

        # 计算归一化后的观测
        normalized_obs = (obs - self.running_stats.mean) / torch.sqrt(self.running_stats.var + 1e-8)

        # 裁剪到合理范围
        normalized_obs = torch.clamp(normalized_obs, -self.clip_range, self.clip_range)

        return normalized_obs

    def get_stats(self):
        """获取当前统计量"""
        return {
            'mean': self.running_stats.mean,
            'var': self.running_stats.var,
            'count': self.running_stats.count
        }