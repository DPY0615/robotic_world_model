# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
from mbrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3FlatEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def disable_mbrl_unsupported_rewards(self):
        # These terms depend on signals that the current world model does not output.
        unsupported_reward_names = [
            "base_height_l2",
            "feet_slide",
            "feet_height",
            "feet_height_body",
            "contact_forces",
        ]
        for reward_name in unsupported_reward_names:
            reward_term = getattr(self.rewards, reward_name, None)
            if reward_term is not None:
                reward_term.weight = 0.0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        

        # override rewards
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane" 
        self.scene.terrain.terrain_generator = None 
        # no height scan
        # self.scene.height_scanner = None
        self.observations.policy.height_scan = None 
        if hasattr(self.observations, "critic"):
            self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None 

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_THIGH"]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsLite3FlatEnvCfg":
            self.disable_zero_weight_rewards() 

@configclass
class DeeproboticsLite3FlatEnvCfg_INIT(DeeproboticsLite3FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # revert rewards to easier warm-start settings
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.feet_air_time.weight = 0.125

        # revert terrain from flat to easy rough
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0

        self.disable_zero_weight_rewards()
        
@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):
    @configclass
    class SystemStateCfg(ObsGroup):
        # Keep aligned with Lite3 world-model state layout (45 dim):
        # [base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, joint_torque]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemContactCfg(ObsGroup):
        thigh_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_THIGH"), "threshold": 1.0},
        )
        foot_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):
        # Lite3 base link name is TORSO.
        base_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="TORSO"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups used by model-based pretraining
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class DeeproboticsLite3FlatEnvCfg_PRETRAIN(DeeproboticsLite3FlatEnvCfg):
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()

    def __post_init__(self):
        super().__post_init__()
        self.disable_mbrl_unsupported_rewards()
        self.disable_zero_weight_rewards()

@configclass
class DeeproboticsLite3FlatEnvCfg_FINETUNE(DeeproboticsLite3FlatEnvCfg_PRETRAIN):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # smaller scene for finetune/debug
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5

        # disable observation corruption
        self.observations.policy.enable_corruption = False

        # use sampled velocity command class in finetune
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsLite3FlatEnvCfg_FINETUNE_GLOBAL_BALANCED(DeeproboticsLite3FlatEnvCfg_FINETUNE):
    """Balanced omni-directional command coverage with moderate tracking emphasis."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Increase command diversity for x/y/yaw and resample more frequently.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.2, 1.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.resampling_time_range = (0.25, 2.0)

        # Keep reward profile close to stable defaults while reducing directional bias.
        self.rewards.track_lin_vel_xy_exp.weight = 3.2
        self.rewards.track_ang_vel_z_exp.weight = 1.8
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.stand_still.weight = -0.2
        self.rewards.feet_contact_without_cmd.weight = 0.05

        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsLite3FlatEnvCfg_FINETUNE_CONS_YAW(DeeproboticsLite3FlatEnvCfg_FINETUNE):
    """Conservative posture with stronger yaw/lateral command learning."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Bias coverage toward lateral/yaw commands to improve non-forward locomotion.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.3, 1.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.resampling_time_range = (0.2, 1.5)

        # Strengthen yaw tracking while relaxing symmetry/command smoothness constraints.
        self.rewards.track_lin_vel_xy_exp.weight = 2.8
        self.rewards.track_ang_vel_z_exp.weight = 2.3
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_mirror.weight = -0.02
        self.rewards.feet_gait.weight = 0.35
        self.rewards.stand_still.weight = -0.15
        self.rewards.feet_contact_without_cmd.weight = 0.05

        self.disable_zero_weight_rewards()

@configclass
class DeeproboticsLite3FlatEnvCfg_VISUALIZE(DeeproboticsLite3FlatEnvCfg_PRETRAIN): # 可视化阶段的环境配置，继承自 DeeproboticsLite3FlatEnvCfg_PRETRAIN
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for visualize
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for visualize
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # override commands
        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        # override randomization
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform_visualize
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            }
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_scale_visualize

        self.disable_mbrl_unsupported_rewards()
        self.disable_zero_weight_rewards()
