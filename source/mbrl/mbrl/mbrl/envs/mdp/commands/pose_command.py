# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""
# 该文件是一个“用于 pose tracking 的命令生成器子模块”

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg


class UniformPoseCommand_Visualize(UniformPoseCommand): 
    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        
    def _resample_command(self, env_ids: Sequence[int]): # 对指定env_ids进行重采样
        # intersect the reset envs with only the real envs
        uniques, counts = torch.cat([env_ids, self.env.env_ids_real]).unique(return_counts=True)
        env_ids = uniques[counts > 1] # 计算传进来的id里只留下那些属于real env的id
        super()._resample_command(env_ids) # 对real env的id进行重采样
        self.pose_command_b[env_ids + 1] = self.pose_command_b[env_ids].clone() # 将重采样后的pose_command_b的值复制到pose_command_b[env_ids + 1]中，即将real env的命令复制到imagination env中

    def _set_debug_vis_impl(self, debug_vis: bool): # 控制debug可视化的显示与隐藏
        super()._set_debug_vis_impl(debug_vis)
        
        if debug_vis:
            if not hasattr(self, "real_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Model/real"
                self.real_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Model/imagination"
                self.imagination_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.real_visualizer.set_visibility(True)
            self.imagination_visualizer.set_visibility(True)
        else:
            if hasattr(self, "real_visualizer"):
                self.real_visualizer.set_visibility(False)
                self.imagination_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        # update the markers
        # -- real
        marker_position = self.robot.data.root_pos_w.clone()
        marker_position[:, 2] += 1.0
        self.real_visualizer.visualize(marker_position[self.env.env_ids_real])
        # -- imagination
        self.imagination_visualizer.visualize(marker_position[self.env.env_ids_imagination])
