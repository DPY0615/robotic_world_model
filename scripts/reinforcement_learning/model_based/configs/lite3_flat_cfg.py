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
                "action_rate_l2": -0.02,
                "base_height_l2": 0.0,
                "feet_air_time": 0.5,
                "feet_air_time_variance": -1.0,
                "feet_slide": 0.0,
                "stand_still": -1.0,
                "feet_height_body": 0.0,
                "feet_height": 0.0,
                "contact_forces": 0.0,
                "lin_vel_z_l2": -2.0,
                "ang_vel_xy_l2": -0.05,
                "track_lin_vel_xy_exp": 5.0,
                "track_ang_vel_z_exp": 0.5,
                "undesired_contacts": -1.0,
                "joint_torques_l2": -2.5e-5,
                "joint_acc_l2": -1.0e-8,
                "joint_deviation_l1": -0.5,
                "joint_power": -2.0e-5,
                "flat_orientation_l2": -5.0,
                "feet_gait": 0.0,
                "joint_mirror": 0.2,
                "joint_pos_limits": 0.2,
                "feet_contact_without_cmd": 0.1,
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
        batch_data_size: int = 10000
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

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
