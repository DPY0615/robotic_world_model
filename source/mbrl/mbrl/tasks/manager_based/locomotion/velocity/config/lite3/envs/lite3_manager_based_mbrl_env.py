# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv

class Lite3ManagerBasedMBRLEnv(ManagerBasedMBRLEnv):
    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0]
        self.base_velocity = None
    
    def prepare_imagination(self):
        self.imagination_common_step_counter = 0
        self.system_dynamics_model_ids = torch.randint(
            0, self.system_dynamics.ensemble_size, (1, self.num_imagination_envs, 1), device=self.device
        )
        self._init_imagination_reward_buffer()
        self._init_intervals()
        self._init_additional_imagination_attributes()
        self._init_imagination_command()
        self.last_obs = TensorDict(
            {
                "policy": torch.zeros(
                    self.num_imagination_envs, self.observation_manager.group_obs_dim["policy"][0], device=self.device
                ),
                "critic": torch.zeros(
                    self.num_imagination_envs, self.observation_manager.group_obs_dim["critic"][0], device=self.device
                ),
            },
            batch_size=[self.num_imagination_envs],
            device=self.device,
        )
        self._last_joint_vel_raw = torch.zeros(
            self.num_imagination_envs,
            self.default_joint_vel.shape[0],
            device=self.device,
        )
        self.imagination_extras = {}
        self._reset_imagination_idx(torch.arange(self.num_imagination_envs, device=self.device))

    def _reset_imagination_idx(self, env_ids):
        super()._reset_imagination_idx(env_ids)
        if "critic" in self.last_obs.keys():
            self.last_obs["critic"][env_ids] = 0.0
        self._last_joint_vel_raw[env_ids] = 0.0

    def _init_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_imagination_envs, 4, device=self.device)
        self._feet_gait_pairs_cache = None
        self._joint_mirror_pairs_cache = None
    
    def _reset_additional_imagination_attributes(self, env_ids):
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0

    def _resolve_foot_indices(self, foot_names):
        # The imagination contact layout is fixed to local foot order: [FL, FR, HL, HR].
        # Resolve gait feet by semantic names, not global body ids from contact sensors.
        resolved = []
        for foot_name in foot_names:
            key = str(foot_name).upper()
            if "FL" in key:
                resolved.append(0)
            elif "FR" in key:
                resolved.append(1)
            elif "HL" in key:
                resolved.append(2)
            elif "HR" in key:
                resolved.append(3)
            else:
                return []

        if len(set(resolved)) != len(resolved):
            return []
        return resolved

    def _resolve_joint_indices(self, joint_pattern):
        robot = self.scene["robot"]
        try:
            return list(robot.find_joints(joint_pattern)[0])
        except Exception:
            return []

    def _build_feet_gait_pairs(self):
        if "feet_gait" not in self.reward_term_names:
            return None
        gait_cfg = self.reward_manager.get_term_cfg("feet_gait")
        gait_pairs = gait_cfg.params.get("synced_feet_pair_names", None)
        if gait_pairs is None or len(gait_pairs) != 2:
            return None
        pair0 = self._resolve_foot_indices(gait_pairs[0])
        pair1 = self._resolve_foot_indices(gait_pairs[1])
        if len(pair0) != 2 or len(pair1) != 2:
            return None
        return [pair0, pair1]

    def _build_joint_mirror_pairs(self):
        if "joint_mirror" not in self.reward_term_names:
            return None
        mirror_cfg = self.reward_manager.get_term_cfg("joint_mirror")
        mirror_pairs = mirror_cfg.params.get("mirror_joints", [])
        resolved_pairs = []
        for joint_pair in mirror_pairs:
            if len(joint_pair) != 2:
                continue
            left_ids = self._resolve_joint_indices(joint_pair[0])
            right_ids = self._resolve_joint_indices(joint_pair[1])
            pair_len = min(len(left_ids), len(right_ids))
            if pair_len == 0:
                continue
            resolved_pairs.append((left_ids[:pair_len], right_ids[:pair_len]))
        return resolved_pairs if len(resolved_pairs) > 0 else None
    
    def get_imagination_observation(self, state_history, action_history, observation_noise=None):
        obs_base_lin_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 0:3] 
        obs_base_ang_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 3:6]
        obs_projected_gravity = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 6:9]
        obs_joint_pos = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 9:21]
        obs_joint_vel = self.imagination_state_normalizer.inverse(state_history[:, -1])[:, 21:33]
        self.obs_last_action = self.imagination_action_normalizer.inverse(action_history[:, -1])
        self._last_joint_vel_raw = obs_joint_vel.clone()
        if observation_noise is None:
            observation_noise = bool(getattr(self.cfg.observations.policy, "enable_corruption", False))
        base_ang_vel_scale = getattr(self.cfg.observations.policy.base_ang_vel, "scale", 1.0)
        joint_vel_scale = getattr(self.cfg.observations.policy.joint_vel, "scale", 1.0)
        
        critic_base_lin_vel = obs_base_lin_vel
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

        policy_base_ang_vel = policy_base_ang_vel * base_ang_vel_scale
        policy_joint_vel = policy_joint_vel * joint_vel_scale

        policy_obs = torch.cat(
            [policy_base_ang_vel, policy_projected_gravity, self.base_velocity, policy_joint_pos, policy_joint_vel, self.obs_last_action],
            dim=1,
        )

        critic_obs = torch.cat(
            [critic_base_lin_vel, critic_base_ang_vel, critic_projected_gravity, self.base_velocity, critic_joint_pos, critic_joint_vel, self.obs_last_action],
            dim=1,
        )

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
        self._latest_projected_gravity = projected_gravity
        return parsed_imagination_states
    
    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {
        }
        return parsed_extensions
    
    def _parse_contacts(self, contacts):
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None
        
        parsed_contacts = {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }
        return parsed_contacts
    
    def _parse_terminations(self, terminations):
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None
        bad_orientation_2 = None
        if hasattr(self, "_latest_projected_gravity"):
            bad_orientation_2 = (self._latest_projected_gravity[:, 2] > 0) | (
                self._latest_projected_gravity[:, :2].abs() > 0.7
            ).any(-1)
        if parsed_terminations is None:
            return bad_orientation_2
        if bad_orientation_2 is None:
            return parsed_terminations
        return parsed_terminations | bad_orientation_2
    
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

        prev_joint_vel = self._last_joint_vel_raw
        joint_acc = (joint_vel - prev_joint_vel) / self.step_dt

        thigh_contact = None if parsed_contacts is None else parsed_contacts.get("thigh_contact", None)
        foot_contact = None if parsed_contacts is None else parsed_contacts.get("foot_contact", None)

        cmd_norm = torch.norm(self.base_velocity, dim=1)

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_std = 0.71
        if "track_lin_vel_xy_exp" in self.reward_term_names:
            track_lin_vel_xy_std = self.reward_manager.get_term_cfg("track_lin_vel_xy_exp").params.get(
                "std", track_lin_vel_xy_std
            )
        track_ang_vel_z_std = 0.71
        if "track_ang_vel_z_exp" in self.reward_term_names:
            track_ang_vel_z_std = self.reward_manager.get_term_cfg("track_ang_vel_z_exp").params.get(
                "std", track_ang_vel_z_std
            )
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / track_lin_vel_xy_std**2)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / track_ang_vel_z_std**2)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        joint_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        joint_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        joint_power = torch.sum(torch.abs(joint_vel * joint_torque), dim=1)

        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # The world-model state stores joint_pos_rel, so this is already an offset from default pose.
        joint_pos_error = joint_pos
        hipx_ids = [0, 3, 6, 9]
        joint_deviation_l1 = torch.sum(torch.abs(joint_pos_error[:, hipx_ids]), dim=1)

        stand_still_threshold = 0.06
        if "stand_still" in self.reward_term_names:
            stand_still_threshold = self.reward_manager.get_term_cfg("stand_still").params.get(
                "command_threshold", stand_still_threshold
            )
        elif "stand_still_without_cmd" in self.reward_term_names:
            stand_still_threshold = self.reward_manager.get_term_cfg("stand_still_without_cmd").params.get(
                "command_threshold", stand_still_threshold
            )
        stand_still = torch.sum(torch.abs(joint_pos_error), dim=1) * (cmd_norm < stand_still_threshold)

        # Defaults for contact-derived terms
        feet_air_time = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_air_time_variance = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_contact_without_cmd = torch.zeros(self.num_imagination_envs, device=self.device)
        undesired_contacts = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_gait = torch.zeros(self.num_imagination_envs, device=self.device)

        if foot_contact is not None:
            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0.0) & is_contact
            is_first_detached = (self.current_contact_time > 0.0) & (~is_contact)

            feet_air_time_threshold = 0.5
            if "feet_air_time" in self.reward_term_names:
                feet_air_time_threshold = self.reward_manager.get_term_cfg("feet_air_time").params.get(
                    "threshold", feet_air_time_threshold
                )

            feet_air_time = torch.sum((self.current_air_time - feet_air_time_threshold) * is_first_contact, dim=1) * (
                cmd_norm > 0.1
            )
            feet_contact_without_cmd = torch.sum(is_first_contact.float(), dim=1) * (cmd_norm < 0.5)

            self.last_air_time = torch.where(is_first_contact, self.current_air_time, self.last_air_time)
            self.current_air_time = torch.where(~is_contact, self.current_air_time + self.step_dt, 0.0)
            self.last_contact_time = torch.where(
                is_first_detached, self.current_contact_time, self.last_contact_time
            )
            self.current_contact_time = torch.where(is_contact, self.current_contact_time + self.step_dt, 0.0)

            feet_air_time_variance = torch.var(torch.clamp(self.last_air_time, max=0.5), dim=1) + torch.var(
                torch.clamp(self.last_contact_time, max=0.5), dim=1
            )

            if "feet_gait" in self.reward_term_names:
                if self._feet_gait_pairs_cache is None:
                    self._feet_gait_pairs_cache = self._build_feet_gait_pairs()
                gait_pairs = self._feet_gait_pairs_cache
                if gait_pairs is not None:
                    gait_cfg = self.reward_manager.get_term_cfg("feet_gait")
                    std = gait_cfg.params.get("std", 0.5**0.5)
                    max_err = gait_cfg.params.get("max_err", 0.2)
                    velocity_threshold = gait_cfg.params.get("velocity_threshold", 0.5)
                    command_threshold = gait_cfg.params.get("command_threshold", 0.1)

                    pair_0 = gait_pairs[0]
                    pair_1 = gait_pairs[1]

                    air_time = self.current_air_time
                    contact_time = self.current_contact_time

                    def _sync_reward_func(foot_0, foot_1):
                        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
                        se_contact = torch.clip(
                            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2
                        )
                        return torch.exp(-(se_air + se_contact) / std)

                    def _async_reward_func(foot_0, foot_1):
                        se_act_0 = torch.clip(
                            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2
                        )
                        se_act_1 = torch.clip(
                            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2
                        )
                        return torch.exp(-(se_act_0 + se_act_1) / std)

                    sync_reward = _sync_reward_func(pair_0[0], pair_0[1]) * _sync_reward_func(pair_1[0], pair_1[1])
                    async_reward = (
                        _async_reward_func(pair_0[0], pair_1[0])
                        * _async_reward_func(pair_0[1], pair_1[1])
                        * _async_reward_func(pair_0[0], pair_1[1])
                        * _async_reward_func(pair_1[0], pair_0[1])
                    )
                    body_vel = torch.norm(base_lin_vel[:, :2], dim=1)
                    enforce_mask = (cmd_norm > command_threshold) | (body_vel > velocity_threshold)
                    feet_gait = torch.where(enforce_mask, sync_reward * async_reward, torch.zeros_like(sync_reward))
        if thigh_contact is not None:
            undesired_contacts = torch.sum(thigh_contact.float(), dim=1)

        joint_mirror = torch.zeros(self.num_imagination_envs, device=self.device)
        if "joint_mirror" in self.reward_term_names:
            if self._joint_mirror_pairs_cache is None:
                self._joint_mirror_pairs_cache = self._build_joint_mirror_pairs()
            mirror_pairs = self._joint_mirror_pairs_cache
            if mirror_pairs is not None and len(mirror_pairs) > 0:
                joint_mirror_acc = torch.zeros(self.num_imagination_envs, device=self.device)
                joint_pos_abs = joint_pos + self.default_joint_pos.unsqueeze(0)
                for left_ids, right_ids in mirror_pairs:
                    left_joint_pos = joint_pos_abs[:, left_ids]
                    right_joint_pos = joint_pos_abs[:, right_ids]
                    joint_mirror_acc += torch.sum(torch.square(left_joint_pos - right_joint_pos), dim=1)
                joint_mirror = joint_mirror_acc / len(mirror_pairs)
                joint_mirror *= torch.clamp(-projected_gravity[:, 2], 0.0, 0.7) / 0.7

        joint_pos_limits = torch.zeros(self.num_imagination_envs, device=self.device)
        if "joint_pos_limits" in self.reward_term_names:
            robot = self.scene["robot"]
            soft_joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
            if soft_joint_pos_limits is not None:
                if soft_joint_pos_limits.ndim == 3:
                    soft_joint_pos_limits = soft_joint_pos_limits[0]
                joint_pos_abs = joint_pos + self.default_joint_pos.unsqueeze(0)
                lower_limits = soft_joint_pos_limits[:, 0]
                upper_limits = soft_joint_pos_limits[:, 1]
                lower_violation = (lower_limits.unsqueeze(0) - joint_pos_abs).clamp(min=0.0)
                upper_violation = (joint_pos_abs - upper_limits.unsqueeze(0)).clamp(min=0.0)
                joint_pos_limits = torch.sum(lower_violation + upper_violation, dim=1)

        # Unavailable from current model outputs -> keep zero placeholders
        base_height_l2 = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_slide = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_height = torch.zeros(self.num_imagination_envs, device=self.device)
        feet_height_body = torch.zeros(self.num_imagination_envs, device=self.device)
        contact_forces = torch.zeros(self.num_imagination_envs, device=self.device)

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

        base_ang_vel_scale = getattr(self.cfg.observations.policy.base_ang_vel, "scale", 1.0)
        joint_vel_scale = getattr(self.cfg.observations.policy.joint_vel, "scale", 1.0)

        # Keep last_obs aligned with Lite3 real policy observation scaling.
        last_policy_obs = torch.cat(
            [
                base_ang_vel * base_ang_vel_scale,
                projected_gravity,
                self.base_velocity,
                joint_pos,
                joint_vel * joint_vel_scale,
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
