from dataclasses import dataclass, asdict, field
from typing import List, Dict


@dataclass # 数据类装饰器
class BaseConfig:
    experiment_name: str = "online" # 实验名，影响wandb的project和日志目录

    @dataclass
    class ExperimentConfig: # 实验配置类
        environment: str = "dummy" 
        device: str = "cuda"
        
        def to_dict(self):
            return asdict(self)

    @dataclass
    class EnvironmentConfig:
        num_envs: int = 8192
        max_episode_length: int = 256
        step_dt: float = 0.02 
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {"dummy": 0.0})
        reward_term_params: Dict[str, Dict[str, object]] = field(default_factory=dict)
        uncertainty_penalty_weight: float = -0.0
        observation_noise: bool = True
        command_resample_interval_range: List[int] | None = None
        event_interval_range: List[int] | None = None
        
        def to_dict(self):
            return asdict(self)
    
    @dataclass
    class DataConfig:       
        dataset_root: str = "logs/online"
        dataset_folder: str = "train" 
        file_data_size: int = 10000 
        batch_data_size: int = 50000

        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {"dummy": [0]}) 
        state_data_mean: List[float] = field(default_factory=lambda: [0.0]) 
        state_data_std: List[float] = field(default_factory=lambda: [0.0])
        action_data_mean: List[float] = field(default_factory=lambda: [0.0])
        action_data_std: List[float] = field(default_factory=lambda: [0.0]) 

        init_data_ratio: float = 0.8 
        num_eval_trajectories: int = 10
        len_eval_trajectory: int = 400
        num_visualizations: int = 4

        def to_dict(self):
            return asdict(self)
        
    @dataclass
    class ModelArchitectureConfig: 
        history_horizon: int = 1 # the window size of the input state transitions
        forecast_horizon: int = 1 # the autoregressive prediction steps
        extension_dim: int = 0
        contact_dim: int = 0
        termination_dim: int = 0
        ensemble_size: int = 1 # 集成模型中子模型的数量
        architecture_config: Dict[str, object] = field(default_factory=lambda: { # 模型架构配置字典，包含模型类型和各层的维度等信息
            
            "type": "mlp",
            "base_shape": [256, 256],
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        })
        # architecture_config: Dict[str, object] = field(default_factory=lambda: {
        #     "type": "rnn",
        #     "rnn_type": "gru",
        #     "rnn_num_layers": 2,
        
        #     "rnn_hidden_size": 256,
        #     "state_mean_shape": [128],
        #     "state_logstd_shape": [128],
        #     "extension_shape": [128],
        #     "contact_shape": [128],
        #     "termination_shape": [128],
        # })
        freeze_auxiliary: bool = False
        resume_path: str | None = None

        def to_dict(self):
            return asdict(self)

    @dataclass
    class ModelOptimizerConfig:
        learning_rate: float = 1.0e-4
        weight_decay: float = 1.0e-5

        def to_dict(self):
            return asdict(self)

    @dataclass
    class ModelTrainingConfig:
        save_interval: int = 200
        max_iterations: int = 1000
        batch_size: int = 1024
        random_batch_updates: bool = True
        max_eval_batches: int = 64
        eval_traj_noise_scale: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.5, 0.8])
        system_dynamics_loss_weights: Dict[str, float] = field(default_factory=lambda: {
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        })

        def to_dict(self):
            return asdict(self)
    
    @dataclass
    class PolicyArchitectureConfig: 
        observation_dim: int = 0
        obs_groups: Dict[str, List[str]] = field(default_factory=lambda: {"policy": ["policy"]})
        action_dim: int = 0
        actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
        activation: str = "elu"
        init_noise_std: float = 1.0
        noise_std_type: str = "scalar"
        resume_path: str | None = None

        def to_dict(self):
            return asdict(self)

    @dataclass
    class PolicyAlgorithmConfig: 
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.005
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 1.0e-3
        schedule: str = "adaptive"
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0

        def to_dict(self):
            return asdict(self)

    @dataclass
    class PolicyTrainingConfig: # 策略训练配置类
        num_steps_per_env: int = 24
        save_interval: int = 200
        max_iterations: int = 500
        export_dir: str | None = None
        
        def to_dict(self):
            return asdict(self)

    def to_dict(self):
        return asdict(self)
    # 下面是各个配置类的实例化，作为BaseConfig的属性
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    model_optimizer_config: ModelOptimizerConfig = field(default_factory=ModelOptimizerConfig)
    model_training_config: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
