# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv

class Lite3ManagerBasedMBRLEnv(ManagerBasedMBRLEnv): # Lite3 的 manager-based MBRL 环境
    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0] # 获取机器人的默认关节位置
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0] # 获取机器人的默认关节速度
        self.base_velocity = None # 速度命令，在计算奖励时会用到
    
    def _init_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device) # 最后一次离地时间
        self.current_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device) # 当前离地时间
        self.last_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device) # 最后一次接触时间
        self.current_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device) # 当前接触时间
    
    def _reset_additional_imagination_attributes(self, env_ids): # 在 imagination env 中重置与接触相关的属性
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
    
    def get_imagination_observation(self, state_history, action_history, observation_noise=True): # 从状态历史和动作历史中获取 imagination observation
        obs_base_lin_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 0:3] 
        obs_base_ang_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 3:6]
        obs_projected_gravity = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 6:9]
        obs_joint_pos = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 9:21]
        obs_joint_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 21:33]
        self.obs_last_action = self.imagination_action_normalizer.inverse(action_history[:, -1])
        
        critic_base_lin_vel = obs_base_ang_vel
        critic_base_ang_vel = obs_base_ang_vel
        critic_projected_gravity = obs_projected_gravity
        critic_joint_pos = obs_joint_pos
        critic_joint_vel = obs_joint_vel
        
        policy_base_ang_vel = obs_base_ang_vel.clone()
        policy_projected_gravity = obs_projected_gravity.clone()
        policy_joint_pos = obs_joint_pos.clone()
        policy_joint_vel = obs_joint_vel.clone()
        
        if observation_noise:
            policy_base_ang_vel += 2 * (torch.rand_like(policy_base_ang_vel) - 0.5) * 0.2
            policy_projected_gravity += 2 * (torch.rand_like(policy_projected_gravity) - 0.5) * 0.05
            policy_joint_pos += 2 * (torch.rand_like(policy_joint_pos) - 0.5) * 0.01
            policy_joint_vel += 2 * (torch.rand_like(policy_joint_vel) - 0.5) * 1.5

        policy_obs = torch.cat(
            [policy_base_ang_vel, policy_projected_gravity, self.base_velocity, policy_joint_pos, policy_joint_vel, self.obs_last_action],
            dim=1,
        )

        critic_obs = torch.cat(
            [critic_base_lin_vel, critic_base_ang_vel, critic_projected_gravity, self.base_velocity, critic_joint_pos, critic_joint_vel, self.obs_last_action],
            dim=1,
        )
            # obs = torch.cat([obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action], dim=1) # 将观测的各个部分拼接在一起，形成最终的观测向量
        # obs = TensorDict({"policy": obs}, batch_size=[self.num_imagination_envs], device=self.device)
        obs = TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states
    
    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {
        }
        return parsed_extensions
    
    def _parse_contacts(self, contacts): # 解析接触信息，分为大腿接触和脚部接触
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None # 大腿接触信息[0:4]，经过 sigmoid 和 round 处理后得到二值化的接触状态
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None # 脚部接触信息[4:8]，经过 sigmoid 和 round 处理后得到二值化的接触状态
        
        parsed_contacts = {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }
        return parsed_contacts
    
    def _parse_terminations(self, terminations): # 解析终止信息，经过 sigmoid 和 round 处理后得到二值化的终止状态
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None
        return parsed_terminations
    
    def _compute_imagination_reward_terms(
        self,
        parsed_imagination_states,
        rollout_action,
        parsed_extensions,
        parsed_contacts,
    ):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]

        # Lite3 policy obs: [ang(0:3), gravity(3:6), cmd(6:9), q(9:21), dq(21:33), a(33:45)]
        prev_joint_vel = self.last_obs["policy"][:, 21:33]
        joint_acc = (joint_vel - prev_joint_vel) / self.step_dt

        thigh_contact = None if parsed_contacts is None else parsed_contacts.get("thigh_contact", None)
        foot_contact = None if parsed_contacts is None else parsed_contacts.get("foot_contact", None)

        cmd_norm = torch.norm(self.base_velocity, dim=1)
        cmd_xy_norm = torch.norm(self.base_velocity[:, :2], dim=1)

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        joint_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        joint_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        joint_power = torch.sum(torch.abs(joint_vel * joint_torque), dim=1)

        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # Lite3 : [HipX, HipY, Knee]
        joint_pos_error = joint_pos - self.default_joint_pos.unsqueeze(0)
        hipx_ids = [0, 3, 6, 9]
        joint_deviation_l1 = torch.sum(torch.abs(joint_pos_error[:, hipx_ids]), dim=1)

        stand_still = torch.sum(torch.abs(joint_pos_error), dim=1) * (cmd_norm < 0.1)

        # Defaults for contact-derived terms
        feet_air_time = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_air_time_variance = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_contact_without_cmd = torch.zeros(self.num_imagination_envs, device=self.device)
        undesired_contacts = torch.zeros(self.num_imagination_envs, device=self.device)

        if foot_contact is not None:
            # First contact event from internal contact-time buffer
            first_contact = (self.current_contact_time > 0.0) & (
                self.current_contact_time < (self.step_dt + 1.0e-8)
            )

            feet_air_time = torch.sum((self.last_air_time - 0.5) * first_contact, dim=1) * (cmd_xy_norm > 0.1)
            feet_contact_without_cmd = torch.sum(first_contact.float(), dim=1) * (cmd_norm < 0.5)

            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0.0) & is_contact
            is_first_detached = (self.current_contact_time > 0.0) & (~is_contact)

            self.last_air_time = torch.where(is_first_contact, self.current_air_time + self.step_dt, self.last_air_time)
            self.current_air_time = torch.where(~is_contact, self.current_air_time + self.step_dt, 0.0)
            self.last_contact_time = torch.where(
                is_first_detached, self.current_contact_time + self.step_dt, self.last_contact_time
            )
            self.current_contact_time = torch.where(is_contact, self.current_contact_time + self.step_dt, 0.0)

            feet_air_time_variance = torch.var(torch.clamp(self.last_air_time, max=0.5), dim=1) + torch.var(
                torch.clamp(self.last_contact_time, max=0.5), dim=1
            )

        if thigh_contact is not None:
            undesired_contacts = torch.sum(thigh_contact.float(), dim=1)

        # Unavailable from current model outputs -> keep zero placeholders
        base_height_l2 = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_slide = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_height = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_height_body = torch.zeros(self.num_imagination_envs, device=self.device)
        contact_forces = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_gait = torch.zeros(self.num_imagination_envs, device=self.device)
        joint_mirror = torch.zeros(self.num_imagination_envs, device=self.device)
        joint_pos_limits = torch.zeros(self.num_imagination_envs, device=self.device)

        self.imagination_reward_per_step = {
            "action_rate_l2": action_rate_l2,
            "base_height_l2": base_height_l2,
            "feet_air_time": feet_air_time,
            "feet_air_time_variance": feet_air_time_variance,
            "feet_slide": feet_slide,
            "stand_still": stand_still,
            "feet_height_body": feet_height_body,
            "feet_height": feet_height,
            "contact_forces": contact_forces,
            "lin_vel_z_l2": lin_vel_z_l2,
            "ang_vel_xy_l2": ang_vel_xy_l2,
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": track_ang_vel_z_exp,
            "undesired_contacts": undesired_contacts,
            "joint_torques_l2": joint_torques_l2,
            "joint_acc_l2": joint_acc_l2,
            "joint_deviation_l1": joint_deviation_l1,
            "joint_power": joint_power,
            "flat_orientation_l2": flat_orientation_l2,
            "feet_gait": feet_gait,
            "joint_mirror": joint_mirror,
            "joint_pos_limits": joint_pos_limits,
            "feet_contact_without_cmd": feet_contact_without_cmd,
        }

        # Keep last_obs aligned with your Lite3 policy obs layout (45 dim)
        last_policy_obs = torch.cat(
            [
                base_ang_vel,
                projected_gravity,
                self.base_velocity,
                joint_pos,
                joint_vel,
                rollout_action,
            ],
            dim=1,
        )
        last_critic_obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                self.base_velocity,
                joint_pos,
                joint_vel,
                rollout_action,
            ],
            dim=1,
        )
        self.last_obs = TensorDict(
            {"policy": last_policy_obs, "critic": last_critic_obs},
            batch_size=[self.num_imagination_envs],
            device=self.device,
        )