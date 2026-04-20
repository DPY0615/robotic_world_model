from envs.base import BaseEnv
import torch
from tensordict import TensorDict


class Lite3FlatEnv(BaseEnv):
    policy_observation_dim = 45
    critic_observation_dim = 48
    base_ang_vel_scale = 0.25
    joint_vel_scale = 0.05

    def _build_dummy_obs(self):
        return TensorDict(
            {
                "policy": torch.zeros(self.num_envs, self.policy_observation_dim, device=self.device),
                "critic": torch.zeros(self.num_envs, self.critic_observation_dim, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _init_additional_attributes(self):
        self.base_velocity = None
        self._last_joint_vel_raw = None
        # Semantic mappings aligned with Lite3 online reward setup.
        self._feet_gait_pairs = ((0, 3), (1, 2))
        self._joint_mirror_pairs = (((0, 1, 2), (9, 10, 11)), ((3, 4, 5), (6, 7, 8)))
        self._default_joint_pos = torch.tensor([0.0, -0.8, 1.6] * 4, dtype=torch.float32, device=self.device)
        # Soft joint-position limits in joint_pos_rel space (URDF limits + soft factor 0.99, minus default pose).
        lower_single = [-0.51777, -1.85508, -1.06466]
        upper_single = [0.51777, 1.09908, 1.18066]
        self._joint_pos_rel_lower = torch.tensor(lower_single * 4, dtype=torch.float32, device=self.device)
        self._joint_pos_rel_upper = torch.tensor(upper_single * 4, dtype=torch.float32, device=self.device)

    def _init_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_joint_vel_raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def _reset_additional_imagination_attributes(self, env_ids):
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self._last_joint_vel_raw[env_ids] = 0.0

    def _init_imagination_command(self):
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sample_commands(torch.arange(self.num_envs, device=self.device))

    def _reset_imagination_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self._sample_commands(env_ids)

    def _sample_commands(self, env_ids):
        count = len(env_ids)
        r = torch.empty(count, device=self.device)
        self.base_velocity[env_ids, 0] = r.uniform_(-1.5, 1.5)
        self.base_velocity[env_ids, 1] = r.uniform_(-0.8, 0.8)
        self.base_velocity[env_ids, 2] = r.uniform_(-0.8, 0.8)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= 0.02
        standing_env_ids = env_ids[self.is_standing_env[env_ids]]
        if len(standing_env_ids) > 0:
            self.base_velocity[standing_env_ids] = 0.0

    def get_imagination_observation(self, state_history, action_history):
        state_history_denormalized, action_history_denormalized = self.dataset.denormalize(
            state_history[:, -1], action_history[:, -1]
        )
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:21]
        obs_joint_vel = state_history_denormalized[:, 21:33]
        self.obs_last_action = action_history_denormalized
        self._last_joint_vel_raw = obs_joint_vel.clone()
        
        critic_base_lin_vel = obs_base_lin_vel
        critic_base_ang_vel = obs_base_ang_vel
        critic_projected_gravity = obs_projected_gravity
        critic_joint_pos = obs_joint_pos
        critic_joint_vel = obs_joint_vel

        policy_base_ang_vel = obs_base_ang_vel.clone()
        policy_projected_gravity = obs_projected_gravity.clone()
        policy_joint_pos = obs_joint_pos.clone()
        policy_joint_vel = obs_joint_vel.clone()

        if self.observation_noise:
            policy_base_ang_vel += 2 * (torch.rand_like(policy_base_ang_vel) - 0.5) * 0.2
            policy_projected_gravity += 2 * (torch.rand_like(policy_projected_gravity) - 0.5) * 0.05
            policy_joint_pos += 2 * (torch.rand_like(policy_joint_pos) - 0.5) * 0.01
            policy_joint_vel += 2 * (torch.rand_like(policy_joint_vel) - 0.5) * 1.5

        policy_base_ang_vel = policy_base_ang_vel * self.base_ang_vel_scale
        policy_joint_vel = policy_joint_vel * self.joint_vel_scale

        policy_obs = torch.cat(
            [
                policy_base_ang_vel,
                policy_projected_gravity,
                self.base_velocity,
                policy_joint_pos,
                policy_joint_vel,
                self.obs_last_action,
            ],
            dim=1,
        )
        critic_obs = torch.cat(
            [
                critic_base_lin_vel,
                critic_base_ang_vel,
                critic_projected_gravity,
                self.base_velocity,
                critic_joint_pos,
                critic_joint_vel,
                self.obs_last_action,
            ],
            dim=1,
        )
        self.last_obs = TensorDict(
            {"policy": policy_obs, "critic": critic_obs},
            batch_size=[self.num_envs],
            device=self.device,
        )
        return self.last_obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        self._latest_projected_gravity = projected_gravity
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

    def _parse_contacts(self, contacts):
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None
        return {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }

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

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        
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

        track_lin_vel_xy_std = self.reward_term_params.get("track_lin_vel_xy_exp", {}).get("std", 0.71)
        track_ang_vel_z_std = self.reward_term_params.get("track_ang_vel_z_exp", {}).get("std", 0.71)
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / track_lin_vel_xy_std**2)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / track_ang_vel_z_std**2)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        joint_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        joint_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        joint_power = torch.sum(torch.abs(joint_vel * joint_torque), dim=1)

        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        hipx_ids = [0, 3, 6, 9]
        joint_deviation_l1 = torch.sum(torch.abs(joint_pos[:, hipx_ids]), dim=1)
        stand_still_threshold = self.reward_term_params.get("stand_still", {}).get("command_threshold", 0.1)
        stand_still = torch.sum(torch.abs(joint_pos), dim=1) * (cmd_norm < stand_still_threshold)
        joint_pos_abs = joint_pos + self._default_joint_pos.unsqueeze(0)

        feet_air_time = torch.zeros(self.num_envs, device=self.device)
        feet_air_time_variance = torch.zeros(self.num_envs, device=self.device)
        feet_contact_without_cmd = torch.zeros(self.num_envs, device=self.device)
        undesired_contacts = torch.zeros(self.num_envs, device=self.device)
        feet_gait = torch.zeros(self.num_envs, device=self.device)

        if foot_contact is not None:
            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0.0) & is_contact
            is_first_detached = (self.current_contact_time > 0.0) & (~is_contact)

            feet_air_time_threshold = self.reward_term_params.get("feet_air_time", {}).get("threshold", 0.5)
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

            std = 0.5**0.5
            max_err = 0.2
            velocity_threshold = 0.5
            command_threshold = 0.1

            pair_0 = self._feet_gait_pairs[0]
            pair_1 = self._feet_gait_pairs[1]
            air_time = self.current_air_time
            contact_time = self.current_contact_time

            def _sync_reward_func(foot_0, foot_1):
                se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
                se_contact = torch.clip(
                    torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2
                )
                return torch.exp(-(se_air + se_contact) / std)

            def _async_reward_func(foot_0, foot_1):
                se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
                se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
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

        joint_mirror_acc = torch.zeros(self.num_envs, device=self.device)
        for left_ids, right_ids in self._joint_mirror_pairs:
            left_joint_pos = joint_pos_abs[:, list(left_ids)]
            right_joint_pos = joint_pos_abs[:, list(right_ids)]
            joint_mirror_acc += torch.sum(torch.square(left_joint_pos - right_joint_pos), dim=1)
        joint_mirror = joint_mirror_acc / len(self._joint_mirror_pairs)
        joint_mirror *= torch.clamp(-projected_gravity[:, 2], 0.0, 0.7) / 0.7

        lower_violation = (self._joint_pos_rel_lower.unsqueeze(0) - joint_pos).clamp(min=0.0)
        upper_violation = (joint_pos - self._joint_pos_rel_upper.unsqueeze(0)).clamp(min=0.0)
        joint_pos_limits = torch.sum(lower_violation + upper_violation, dim=1)

        zeros = torch.zeros(self.num_envs, device=self.device)
        self.imagination_reward_per_step = {
            "action_rate_l2": action_rate_l2,
            "base_height_l2": zeros.clone(),
            "feet_air_time": feet_air_time,
            "feet_air_time_variance": feet_air_time_variance,
            "feet_slide": zeros.clone(),
            "stand_still": stand_still,
            "feet_height_body": zeros.clone(),
            "feet_height": zeros.clone(),
            "contact_forces": zeros.clone(),
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

        last_policy_obs = torch.cat(
            [
                base_ang_vel * self.base_ang_vel_scale,
                projected_gravity,
                self.base_velocity,
                joint_pos,
                joint_vel * self.joint_vel_scale,
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
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        return imagination_states

    @property
    def state_dim(self):
        return 45

    @property
    def observation_dim(self):
        return self.policy_observation_dim

    @property
    def action_dim(self):
        return 12
