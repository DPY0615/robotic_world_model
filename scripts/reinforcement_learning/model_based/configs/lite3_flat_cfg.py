from .base_cfg import BaseConfig
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Lite3FlatConfig(BaseConfig):
    experiment_name: str = "offline"

    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "lite3_flat"

    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        reward_term_weights: Dict[str, float] = field(
            default_factory=lambda: {
                # Contact-light profile for the current Lite3 world model: the FL/FR contact heads
                # are much weaker than AnymalD, so keep contact-derived shaping small by default.
                "action_rate_l2": -0.055,
                "base_height_l2": 0.0,
                "feet_air_time": 0.4,
                "feet_air_time_variance": -0.3,
                "feet_slide": 0.0,
                "stand_still": 0.0,
                "feet_height_body": 0.0,
                "feet_height": 0.0,
                "contact_forces": 0.0,
                "lin_vel_z_l2": -3.4,
                "ang_vel_xy_l2": -0.15,
                "track_lin_vel_xy_exp": 2.1,
                "track_ang_vel_z_exp": 1.0,
                "undesired_contacts": -0.5,
                "joint_torques_l2": -2.5e-5,
                "joint_acc_l2": -1.0e-8,
                "joint_deviation_l1": -0.07,
                "joint_power": -2.0e-5,
                "flat_orientation_l2": -8.8,
                "feet_gait": 0.0,
                "joint_mirror": -0.02,
                "joint_pos_limits": -5.0,
                "feet_contact_without_cmd": 0.0,
            }
        )
        reward_term_params: Dict[str, Dict[str, object]] = field(
            default_factory=lambda: {
                "track_lin_vel_xy_exp": {"std": 1.05},
                "track_ang_vel_z_exp": {"std": 1.05},
                "feet_air_time": {"threshold": 0.28},
                "stand_still": {"command_threshold": 0.02},
            }
        )
        uncertainty_penalty_weight: float = -1.0
        observation_noise: bool = False
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] | None = None

    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets/data"
        dataset_folder: str = "lite3"
        batch_data_size: int = 500000
        state_idx_dict: Dict[str, List[int]] = field(
            default_factory=lambda: {
                r"$v$\n$[m/s]$": [0, 1, 2],
                r"$\omega$\n$[rad/s]$": [3, 4, 5],
                r"$g$\n$[1]$": [6, 7, 8],
                r"$q$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            }
        )
        state_data_mean: List[float] = field(
            default_factory=lambda: [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 0.0, 0.0,
                0.0, 0.1, 0.0, -0.2, -0.2, 0.4,
                0.4, 0.3, 0.3,
                0.0, 0.0, 0.1, -0.1,
                -0.5, -0.6, -0.5, -0.5,
                1.1, 1.2, 1.3, 1.3,
                0.4, -0.6, 1.0, -0.7, 0.6, 0.6,
                -1.3, -1.3, -4.4, -4.6, -6.0, -6.0,
            ]
        )
        state_data_std: List[float] = field(
            default_factory=lambda: [
                0.8, 0.4, 0.2,
                0.9, 0.8, 0.3,
                0.1, 0.1, 0.1,
                0.1, 0.1, 0.1,
                0.1, 0.3, 0.3, 0.3, 0.3, 0.2,
                0.2, 0.2, 0.2,
                2.1, 2.2, 2.2, 2.1,
                6.0, 5.1, 4.8, 5.0,
                5.6, 5.9, 5.6, 5.6,
                2.7, 2.7, 3.0, 2.9, 4.3, 4.2,
                4.2, 4.1, 8.7, 8.2, 8.6, 9.1,
            ]
        )
        action_data_mean: List[float] = field(
            default_factory=lambda: [
                0.2, 0.5, 1.1,
                -0.3, -0.1, 1.2,
                0.3, -0.9, 0.6,
                -0.3, -1.1, 0.7,
            ]
        )
        action_data_std: List[float] = field(
            default_factory=lambda: [
                1.1, 1.5, 1.5,
                1.1, 1.3, 1.6,
                1.2, 1.2, 1.4,
                1.1, 1.3, 1.5,
            ]
        )

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 32
        forecast_horizon: int = 8
        ensemble_size: int = 5
        contact_dim: int = 8
        termination_dim: int = 1
        architecture_config: Dict[str, object] = field(
            default_factory=lambda: {
                "type": "rnn",
                "rnn_type": "gru",
                "rnn_num_layers": 2,
                "rnn_hidden_size": 256,
                "state_mean_shape": [128],
                "state_logstd_shape": [128],
                "extension_shape": [128],
                "contact_shape": [128],
                "termination_shape": [128],
            }
        )
        resume_path: str | None = "assets/models/lite3/pretrain_rnn_ens.pt"

    @dataclass
    class ModelOptimizerConfig(BaseConfig.ModelOptimizerConfig):
        learning_rate: float = 1.0e-4
        weight_decay: float = 1.0e-5

    @dataclass
    class ModelTrainingConfig(BaseConfig.ModelTrainingConfig):
        # Lite3 needs longer retraining once the dataset is regenerated from
        # policy rollouts; the old 10k random-action file underfits contacts.
        max_iterations: int = 2500
        save_interval: int = 100
        
    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        observation_dim: int = 45
        obs_groups: Dict[str, List[str]] = field(
            default_factory=lambda: {"policy": ["policy"], "critic": ["critic"]}
        )
        action_dim: int = 12
        
        actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        noise_std_type: str = "log"

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 1.0e-4
        entropy_coef: float = 0.0001

    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 50
        max_iterations: int = 500

    @dataclass
    class SimReferenceConfig:
        enabled: bool = True
        task: str = "Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0"
        num_envs: int = 64
        num_steps: int = 600

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    model_optimizer_config: ModelOptimizerConfig = field(default_factory=ModelOptimizerConfig)
    model_training_config: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
    sim_reference_config: SimReferenceConfig = field(default_factory=SimReferenceConfig)


# Presets around finetune run:
# logs/rsl_rl/deeprobotics_lite3_flat/2026-04-18_16-19-52_s2_term_balance_20260418_161940
LITE3_OFFLINE_PRESETS: Dict[str, Dict[str, object]] = {
    "wm_safe": {
        "reward_term_weights": {},
        "reward_term_params": {
            "track_lin_vel_xy_exp": {"std": 1.05},
            "track_ang_vel_z_exp": {"std": 1.05},
            "feet_air_time": {"threshold": 0.28},
            "stand_still": {"command_threshold": 0.02},
        },
        "uncertainty_penalty_weight": -1.0,
    },
    "ftbest_ref": {
        "reward_term_weights": {
            "action_rate_l2": -0.022,
            "feet_air_time": 1.0,
            "feet_air_time_variance": -1.0,
            "stand_still": 0.0,
            "track_lin_vel_xy_exp": 3.0,
            "track_ang_vel_z_exp": 1.6,
            "feet_gait": 0.45,
            "joint_mirror": -0.04,
            "feet_contact_without_cmd": 0.0,
        },
        "uncertainty_penalty_weight": -0.0,
    },
    "ftbest_ref_u03": {
        "reward_term_weights": {
            "action_rate_l2": -0.022,
            "feet_air_time": 1.0,
            "feet_air_time_variance": -1.0,
            "stand_still": 0.0,
            "track_lin_vel_xy_exp": 3.0,
            "track_ang_vel_z_exp": 1.6,
            "feet_gait": 0.45,
            "joint_mirror": -0.04,
            "feet_contact_without_cmd": 0.0,
        },
        "uncertainty_penalty_weight": -0.3,
    },
    "ftbest_track": {
        "reward_term_weights": {
            "action_rate_l2": -0.020,
            "feet_air_time": 0.8,
            "feet_air_time_variance": -1.2,
            "stand_still": -0.1,
            "track_lin_vel_xy_exp": 3.4,
            "track_ang_vel_z_exp": 1.9,
            "feet_gait": 0.35,
            "joint_mirror": -0.03,
            "feet_contact_without_cmd": 0.05,
        },
        "uncertainty_penalty_weight": -0.5,
    },
    "ftbest_stable": {
        "reward_term_weights": {
            "action_rate_l2": -0.018,
            "feet_air_time": 0.6,
            "feet_air_time_variance": -1.5,
            "stand_still": -0.2,
            "track_lin_vel_xy_exp": 3.2,
            "track_ang_vel_z_exp": 1.7,
            "feet_gait": 0.25,
            "joint_mirror": -0.02,
            "feet_contact_without_cmd": 0.05,
        },
        "uncertainty_penalty_weight": -0.8,
    },
    "ftbest_recover": {
        "reward_term_weights": {
            "action_rate_l2": -0.019,
            "feet_air_time": 0.9,
            "feet_air_time_variance": -1.1,
            "stand_still": -0.05,
            "track_lin_vel_xy_exp": 3.1,
            "track_ang_vel_z_exp": 1.8,
            "feet_gait": 0.35,
            "joint_mirror": -0.03,
            "feet_contact_without_cmd": 0.03,
        },
        "uncertainty_penalty_weight": -0.4,
    },
    "ftbest_track_aggr": {
        "reward_term_weights": {
            "action_rate_l2": -0.020,
            "feet_air_time": 0.75,
            "feet_air_time_variance": -1.0,
            "stand_still": -0.05,
            "track_lin_vel_xy_exp": 3.6,
            "track_ang_vel_z_exp": 2.1,
            "feet_gait": 0.30,
            "joint_mirror": -0.02,
            "feet_contact_without_cmd": 0.02,
        },
        "uncertainty_penalty_weight": -0.6,
    },
    "ftbest_track_aggr_u05": {
        "reward_term_weights": {
            "action_rate_l2": -0.020,
            "feet_air_time": 0.75,
            "feet_air_time_variance": -1.0,
            "stand_still": -0.05,
            "track_lin_vel_xy_exp": 3.6,
            "track_ang_vel_z_exp": 2.1,
            "feet_gait": 0.30,
            "joint_mirror": -0.02,
            "feet_contact_without_cmd": 0.02,
        },
        "uncertainty_penalty_weight": -0.5,
    },
    "ftbest_track_aggr_smooth": {
        "reward_term_weights": {
            "action_rate_l2": -0.022,
            "feet_air_time": 0.75,
            "feet_air_time_variance": -1.0,
            "stand_still": -0.05,
            "track_lin_vel_xy_exp": 3.6,
            "track_ang_vel_z_exp": 2.0,
            "feet_gait": 0.30,
            "joint_mirror": -0.02,
            "feet_contact_without_cmd": 0.02,
        },
        "uncertainty_penalty_weight": -0.6,
    },
    "ftbest_track_aggr_gait": {
        "reward_term_weights": {
            "action_rate_l2": -0.020,
            "feet_air_time": 0.85,
            "feet_air_time_variance": -1.1,
            "stand_still": -0.05,
            "track_lin_vel_xy_exp": 3.6,
            "track_ang_vel_z_exp": 2.1,
            "feet_gait": 0.34,
            "joint_mirror": -0.025,
            "feet_contact_without_cmd": 0.025,
        },
        "uncertainty_penalty_weight": -0.55,
    },
    "ftbest_anti_knee": {
        "reward_term_weights": {
            "action_rate_l2": -0.024,
            "feet_air_time": 0.95,
            "feet_air_time_variance": -1.3,
            "stand_still": -0.20,
            "track_lin_vel_xy_exp": 3.0,
            "track_ang_vel_z_exp": 1.7,
            "feet_gait": 0.42,
            "joint_mirror": -0.045,
            "joint_deviation_l1": -0.7,
            "feet_contact_without_cmd": 0.08,
        },
        "uncertainty_penalty_weight": -0.8,
    },
}


def make_lite3_flat_config(preset: str | None = None) -> Lite3FlatConfig:
    config = Lite3FlatConfig()
    if preset is None:
        return config

    if preset not in LITE3_OFFLINE_PRESETS:
        valid = ", ".join(sorted(LITE3_OFFLINE_PRESETS.keys()))
        raise ValueError(f"Unknown Lite3 offline preset: {preset}. Valid presets: {valid}")

    preset_cfg = LITE3_OFFLINE_PRESETS[preset]
    config.environment_config.reward_term_weights.update(preset_cfg.get("reward_term_weights", {}))
    for term, params in preset_cfg.get("reward_term_params", {}).items():
        config.environment_config.reward_term_params.setdefault(term, {}).update(params)
    config.environment_config.uncertainty_penalty_weight = float(preset_cfg["uncertainty_penalty_weight"])
    return config
