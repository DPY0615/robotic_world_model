from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_stand_still( # 惩罚机器人在静止状态下姿态偏移的奖励函数，计算机器人的关节位置与默认关节位置之间的绝对误差，并乘以一个权重来得到最终的奖励值，如果机器人保持静止状态
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    command = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1) * (command < threshold)


def foot_clearance( # 奖励机器人在行走过程中脚部离地的奖励函数，计算机器人的脚部与地面之间的高度误差，并乘以一个权重来得到最终的奖励值
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    return torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)
