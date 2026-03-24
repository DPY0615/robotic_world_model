# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch

from mbrl.mbrl.envs import ManagerBasedVisualizeEnv
from .anymal_d_manager_based_mbrl_env import ANYmalDManagerBasedMBRLEnv
from mbrl.mbrl.envs.mdp.events import reset_joints_to_specified, reset_root_velocity_to_specified
from isaaclab.utils.math import quat_apply


class ANYmalDManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, ANYmalDManagerBasedMBRLEnv):
    
    
    def _reset_imagination_sim(self, parsed_imagination_states): # 在重置 imagination sim 时，根据解析后的 imagination states 来设置机器人的状态，包括位置、速度等信息
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_pos += self.default_joint_pos # 将解析得到的关节位置加上默认关节位置，得到最终的关节位置
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_vel += self.default_joint_vel # 将解析得到的关节速度加上默认关节速度，得到最终的关节速度
        
        root_quat_w = self.scene["robot"].data.root_quat_w[self.env_ids_imagination]
        
        base_lin_vel_w = quat_apply(root_quat_w, base_lin_vel)
        base_ang_vel_w = quat_apply(root_quat_w, base_ang_vel)
                
        velocities = torch.cat([base_lin_vel_w, base_ang_vel_w], dim=1)
        
        reset_joints_to_specified(self, self.env_ids_imagination, joint_pos, joint_vel) # 调用 reset_joints_to_specified 函数来重置机器人的关节位置和速度
        reset_root_velocity_to_specified(self, self.env_ids_imagination, velocities) # 调用 reset_root_velocity_to_specified 函数来重置机器人的根部线速度和角速度
