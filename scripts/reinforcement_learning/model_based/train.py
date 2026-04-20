from configs import (
    BaseConfig,
    AnymalDFlatConfig,
    Lite3FlatConfig,
    LITE3_OFFLINE_PRESETS,
    make_lite3_flat_config,
)
from envs import BaseEnv, AnymalDFlatEnv, Lite3FlatEnv
from model_training import ModelTraining
from policy_training import PolicyTraining
from rsl_rl.modules import ActorCritic, SystemDynamicsEnsemble
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_obs_groups
import os
import glob
import json
import subprocess
import sys
import torch
from torch.utils.data import Dataset
import argparse
from datetime import datetime
import time
import numpy as np
import wandb


class ModelBasedExperiment:
      

    def __init__(self, environment, device):
        self.env_cls = self.resolve_environment_cls(environment)
        self.device = device
        self.data_file_idx = 0
        
    
    def resolve_environment_cls(self, environment):
        if environment == "anymal_d_flat":
            return AnymalDFlatEnv
        if environment == "lite3_flat":
            return Lite3FlatEnv
        else:
            raise ValueError(f"Unknown environment: {environment}")

    
    def _load_data(self, dataset_root, dataset_folder, file_data_size=10000, batch_data_size=50000):
        batch_state_data = []
        batch_action_data = []
        batch_extension_data = []
        batch_contact_data = []
        batch_termination_data = []
        batch_total_num_data = 0
        while True:
            try:
                file = f"state_action_data_{self.data_file_idx}.csv"
                data = np.loadtxt(os.path.join(dataset_root, dataset_folder, file), delimiter=",", dtype=np.float32)
            except FileNotFoundError:
                if batch_total_num_data > 0:
                    print(f"[Motion Loader] No more files after {file}. Using {batch_total_num_data} rows.")
                    break
                print(f"[Motion Loader] No data found in {os.path.join(dataset_root, dataset_folder, file)}. Waiting for new data.")
                time.sleep(1)
                continue
            if data.ndim == 1:
                data = data[None, :]
            if len(data) < file_data_size:
                if batch_total_num_data > 0:
                    print(f"[Motion Loader] Incomplete file {file}. Using {batch_total_num_data} rows already loaded.")
                    break
                print(f"[Motion Loader] Not enough data in {os.path.join(dataset_root, dataset_folder, file)}. Waiting for new data.")
                time.sleep(1)
                continue
            else:
                state_data = torch.as_tensor(data[:, :self.state_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
                action_data = torch.as_tensor(data[:, self.state_dim:self.state_dim + self.action_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
                extension_data = torch.as_tensor(data[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + self.extension_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
                contact_data = torch.as_tensor(data[:, self.state_dim + self.action_dim + self.extension_dim:self.state_dim + self.action_dim + self.extension_dim + self.contact_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
                termination_data = torch.as_tensor(data[:, self.state_dim + self.action_dim + self.extension_dim + self.contact_dim:], dtype=torch.float32, device=self.device).unsqueeze(0)
                batch_state_data.append(state_data)
                batch_action_data.append(action_data)
                batch_extension_data.append(extension_data)
                batch_contact_data.append(contact_data)
                batch_termination_data.append(termination_data)
                batch_total_num_data += len(data)
                num_trajs, num_steps, state_dim = state_data.shape
                print(f"[Motion Loader] Loaded {num_trajs} {dataset_folder} trajectories of {num_steps} steps from {file}.")
                print(f"[Motion Loader] Total number of data: {batch_total_num_data} / {batch_data_size}")
                self.data_file_idx += 1
            if batch_total_num_data >= batch_data_size:
                break
        batch_state_data = torch.cat(batch_state_data, dim=0)
        batch_action_data = torch.cat(batch_action_data, dim=0)
        batch_extension_data = torch.cat(batch_extension_data, dim=0)
        batch_contact_data = torch.cat(batch_contact_data, dim=0)
        batch_termination_data = torch.cat(batch_termination_data, dim=0)
        action_dim, extension_dim, contact_dim, termination_dim = action_data.shape[-1], extension_data.shape[-1], contact_data.shape[-1], termination_data.shape[-1]
        print(f"[Motion Loader] State dim: {state_dim} | Action dim: {action_dim} | Extension dim: {extension_dim} | Contact dim: {contact_dim} | Termination dim: {termination_dim}")
        return batch_state_data, batch_action_data, batch_extension_data, batch_contact_data, batch_termination_data
    

    def _build_eval_traj_config(self, eval_state_data, eval_action_data, eval_extension_data, eval_contact_data, eval_termination_data, num_eval_trajectories, num_visualizations, len_eval_trajectory, state_idx_dict):
        num_eval_trajs, num_eval_steps, _ = eval_state_data.shape
        ids = torch.randint(0, num_eval_trajs, (num_eval_trajectories,))
        len_eval_trajectory = min(len_eval_trajectory, num_eval_steps)
        start_steps = torch.randint(0, num_eval_steps - len_eval_trajectory + 1, (num_eval_trajectories,))
        start_steps_expanded = start_steps[:, None] + torch.arange(len_eval_trajectory)
        traj_data = [
            eval_state_data[ids[:, None], start_steps_expanded],
            eval_action_data[ids[:, None], start_steps_expanded],
            eval_extension_data[ids[:, None], start_steps_expanded],
            eval_contact_data[ids[:, None], start_steps_expanded],
            eval_termination_data[ids[:, None], start_steps_expanded],
            ]
        self.eval_traj_config = {
            "num_trajs": num_eval_trajectories,
            "num_visualizations": num_visualizations,
            "len_traj": len_eval_trajectory,
            "traj_data": traj_data,
            "state_idx_dict": state_idx_dict,
        }


    def prepare_environment(self, num_envs, max_episode_length, step_dt, reward_term_weights, reward_term_params, uncertainty_penalty_weight, observation_noise, command_resample_interval_range, event_interval_range):
        self.env: BaseEnv = self.env_cls(num_envs, max_episode_length, step_dt, reward_term_weights, reward_term_params, self.device, uncertainty_penalty_weight, observation_noise, command_resample_interval_range, event_interval_range)


    def prepare_data(self, dataset_root, dataset_folder, file_data_size, batch_data_size, state_data_mean=None, state_data_std=None, action_data_mean=None, action_data_std=None, init_data_ratio=0.0, num_eval_trajectories=100, num_visualizations=2, len_eval_trajectory=400, state_idx_dict=None):
        state_data, action_data, extension_data, contact_data, termination_data = self._load_data(dataset_root, dataset_folder=dataset_folder, file_data_size=file_data_size, batch_data_size=batch_data_size)
        self._build_eval_traj_config(state_data, action_data, extension_data, contact_data, termination_data, num_eval_trajectories, num_visualizations, len_eval_trajectory, state_idx_dict)
        class SystemDynamicsDataset(Dataset):
            def __init__(
                self,
                history_horizon,
                forecast_horizon,
                state_data,
                action_data,
                extension_data,
                contact_data,
                termination_data,
                state_data_mean=None,
                state_data_std=None,
                action_data_mean=None,
                action_data_std=None,
                ):
                self.history_horizon = history_horizon
                self.forecast_horizon = forecast_horizon
                self.window_horizon = history_horizon + forecast_horizon
                
                self.state_data_mean = torch.tensor(state_data_mean, device=state_data.device) if state_data_mean is not None else state_data.mean(dim=(0, 1))
                self.state_data_std = torch.tensor(state_data_std, device=state_data.device) if state_data_std is not None else state_data.std(dim=(0, 1)) + 1e-6
                self.action_data_mean = torch.tensor(action_data_mean, device=action_data.device) if action_data_mean is not None else action_data.mean(dim=(0, 1))
                self.action_data_std = torch.tensor(action_data_std, device=action_data.device) if action_data_std is not None else action_data.std(dim=(0, 1)) + 1e-6
                self.state_data, self.action_data = self.normalize(state_data, action_data)
                self.extension_data = extension_data
                self.contact_data = contact_data
                self.termination_data = termination_data

                termination_mask = termination_data[..., 0] > 0.5
                num_terminations = int(termination_mask.sum().item())
                if wandb.run is not None:
                    wandb.log(
                        {
                            "Data/num_terminations": num_terminations,
                            }
                    )
                self.traj_ids, self.start_steps = self._build_valid_window_index(termination_mask)
                self.traj_ids_device = self.traj_ids.to(state_data.device)
                self.start_steps_device = self.start_steps.to(state_data.device)
                print(
                    "[Motion Loader] Valid model windows: "
                    f"{len(self.traj_ids)} from {state_data.shape[0]} trajectories."
                )

            def _build_valid_window_index(self, termination_mask):
                traj_ids = []
                start_steps = []
                num_trajs, num_steps = termination_mask.shape
                num_starts = num_steps - self.window_horizon + 1
                if num_starts <= 0:
                    raise ValueError(
                        "Dataset is shorter than history_horizon + forecast_horizon "
                        f"({self.window_horizon})."
                    )

                for traj_id in range(num_trajs):
                    terminal = termination_mask[traj_id]
                    prefix = torch.zeros(num_steps + 1, dtype=torch.int64, device=terminal.device)
                    prefix[1:] = torch.cumsum(terminal.to(torch.int64), dim=0)
                    starts = torch.arange(num_starts, device=terminal.device)
                    # Allow a terminal flag on the final target row, but never inside the conditioning window.
                    terminal_count = prefix[starts + self.window_horizon - 1] - prefix[starts]
                    valid = starts[terminal_count == 0].detach().cpu()
                    if valid.numel() > 0:
                        traj_ids.append(torch.full((valid.numel(),), traj_id, dtype=torch.long))
                        start_steps.append(valid.to(torch.long))

                if len(traj_ids) == 0:
                    raise ValueError("No valid model windows remain after filtering termination crossings.")
                return torch.cat(traj_ids, dim=0), torch.cat(start_steps, dim=0)

            def __len__(self):
                return len(self.traj_ids)

            def __getitem__(self, idx):
                traj_id = int(self.traj_ids[idx])
                start = int(self.start_steps[idx])
                end = start + self.window_horizon
                return (
                    self.state_data[traj_id, start:end],
                    self.action_data[traj_id, start:end],
                    self.extension_data[traj_id, start:end],
                    self.contact_data[traj_id, start:end],
                    self.termination_data[traj_id, start:end],
                )

            def sample_model_batch(self, batch_size, valid_indices=None):
                if valid_indices is None:
                    idx = torch.randint(0, len(self), (batch_size,), device=self.state_data.device)
                else:
                    pick = torch.randint(0, len(valid_indices), (batch_size,), device=self.state_data.device)
                    idx = valid_indices[pick]
                traj_ids = self.traj_ids_device[idx]
                start_steps = self.start_steps_device[idx]
                offsets = start_steps[:, None] + torch.arange(self.window_horizon, device=self.state_data.device)[None, :]
                return (
                    self.state_data[traj_ids[:, None], offsets],
                    self.action_data[traj_ids[:, None], offsets],
                    self.extension_data[traj_ids[:, None], offsets],
                    self.contact_data[traj_ids[:, None], offsets],
                    self.termination_data[traj_ids[:, None], offsets],
                )
            
            def normalize(self, state_data=None, action_data=None):
                state_data = (state_data - self.state_data_mean) / self.state_data_std if state_data is not None else None
                action_data = (action_data - self.action_data_mean) / self.action_data_std if action_data is not None else None
                return state_data, action_data
            
            def denormalize(self, state_data, action_data):
                state_data = state_data * self.state_data_std + self.state_data_mean if state_data is not None else None
                action_data = action_data * self.action_data_std + self.action_data_mean if action_data is not None else None
                return state_data, action_data
            
            def sample_batch(self, batch_size, normalized=True):
                idx = torch.randint(0, len(self), (batch_size,), device=self.state_data.device)
                traj_ids = self.traj_ids_device[idx]
                start_steps = self.start_steps_device[idx]
                offsets = start_steps[:, None] + torch.arange(self.history_horizon, device=self.state_data.device)[None, :]
                if normalized:
                    return self.state_data[traj_ids[:, None], offsets], self.action_data[traj_ids[:, None], offsets]
                state_sample = self.state_data[traj_ids[:, None], offsets]
                action_sample = self.action_data[traj_ids[:, None], offsets]
                return self.denormalize(state_sample, action_sample)
            
        self.dataset = SystemDynamicsDataset(
            self.history_horizon,
            self.forecast_horizon,
            state_data,
            action_data,
            extension_data,
            contact_data,
            termination_data,
            state_data_mean=state_data_mean,
            state_data_std=state_data_std,
            action_data_mean=action_data_mean,
            action_data_std=action_data_std
            )
        # init env dataset
        self.env.set_dataset(self.dataset)
        if self.env.init_dataset is None:
            self.env.set_init_dataset(self.dataset, init_data_ratio)


    def prepare_model(self, history_horizon, forecast_horizon, extension_dim, contact_dim, termination_dim, ensemble_size, architecture_config, freeze_auxiliary=False, resume_path=None):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.system_dynamics = SystemDynamicsEnsemble(
            self.state_dim,
            self.action_dim,
            self.extension_dim,
            self.contact_dim,
            self.termination_dim,
            self.device,
            ensemble_size=ensemble_size,
            history_horizon=self.history_horizon,
            architecture_config=architecture_config,
            freeze_auxiliary=freeze_auxiliary,
            )
        self.model_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Model] Loading model from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.system_dynamics.load_state_dict(loaded_dict["system_dynamics_state_dict"])
            self.model_learning_iteration = loaded_dict["iter"]
        # init env system dynamics
        self.env.set_system_dynamics(self.system_dynamics)


    def prepare_model_optimizer(self, learning_rate, weight_decay):
        self.model_optimizer = torch.optim.Adam(
            self.system_dynamics.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )


    def train_model(self, log_dir, batch_size, eval_traj_noise_scale, system_dynamics_loss_weights, save_interval, max_iterations, random_batch_updates=True, max_eval_batches=64):
        print(f"[Train Model] Training model for {max_iterations} iterations.")
        model_training = ModelTraining(
            log_dir,
            self.history_horizon,
            self.forecast_horizon,
            self.dataset,
            self.system_dynamics,
            device=self.device,
            optimizer=self.model_optimizer,
            eval_traj_config=self.eval_traj_config,
            batch_size=batch_size,
            eval_traj_noise_scale=eval_traj_noise_scale,
            system_dynamics_loss_weights=system_dynamics_loss_weights,
            save_interval=save_interval,
            max_iterations=max_iterations,
            random_batch_updates=random_batch_updates,
            max_eval_batches=max_eval_batches,
        )
        model_training.current_learning_iteration = self.model_learning_iteration
        model_training.train()
        self.model_learning_iteration += model_training.max_iterations
            

    def prepare_policy(self, observation_dim, obs_groups, action_dim, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, noise_std_type="scalar", resume_path=None):
        self.observation_dim = observation_dim
        default_sets = ["critic"]
        obs_groups = resolve_obs_groups(self.env.dummy_obs, obs_groups, default_sets)
        self.actor_critic = ActorCritic(
            obs=self.env.dummy_obs,
            obs_groups=obs_groups,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            ).to(self.device)
        self.policy_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Policy] Loading policy from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            self.policy_learning_iteration = loaded_dict["iter"]

        
    def prepare_algorithm(self, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl):
        self.alg = PPO(
            self.actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device=self.device,
            )

    
    def train_policy(self, log_dir, num_steps_per_env, save_interval, max_iterations, export_dir):
        print(f"[Train Policy] Training policy for {max_iterations} iterations.")
        policy_training = PolicyTraining(
            log_dir,
            env=self.env,
            alg=self.alg,
            device=self.device,
            num_steps_per_env=num_steps_per_env,
            save_interval=save_interval,
            max_iterations=max_iterations,
            export_dir=export_dir,
            )
        policy_training.current_learning_iteration = self.policy_learning_iteration
        policy_training.learn()
        self.policy_learning_iteration += policy_training.max_iterations


def run_experiment(config: BaseConfig):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_num}"
    log_dir = os.path.join('logs', config.experiment_name, args_cli.task, run_name)
    wandb.run.name = run_name
    os.makedirs(log_dir, exist_ok=True)
    model_experiment = ModelBasedExperiment(**config.experiment_config.to_dict())
    model_experiment.prepare_environment(**config.environment_config.to_dict())
    model_experiment.prepare_model(**config.model_architecture_config.to_dict())
    model_experiment.prepare_data(**config.data_config.to_dict())

    if args_cli.mode in ("model", "both"):
        model_experiment.prepare_model_optimizer(**config.model_optimizer_config.to_dict())
        model_experiment.train_model(log_dir, **config.model_training_config.to_dict())

    if args_cli.mode in ("policy", "both"):
        model_experiment.prepare_policy(**config.policy_architecture_config.to_dict())
        model_experiment.prepare_algorithm(**config.policy_algorithm_config.to_dict())
        model_experiment.train_policy(log_dir, **config.policy_training_config.to_dict())

    # Run simulator reference evaluation after offline training if enabled.
    sim_cfg = getattr(config, "sim_reference_config", None)
    if args_cli.mode in ("policy", "both") and sim_cfg is not None and getattr(sim_cfg, "enabled", False):
        policy_ckpts = glob.glob(os.path.join(log_dir, "policy_*.pt"))
        if len(policy_ckpts) == 0:
            print("[SimRef] No policy checkpoint found. Skip sim reference eval.")
        else:
            def _policy_iter(path: str) -> int:
                base = os.path.basename(path)
                stem = os.path.splitext(base)[0]
                try:
                    return int(stem.split("_")[-1])
                except ValueError:
                    return -1

            policy_ckpt = sorted(policy_ckpts, key=_policy_iter)[-1]
            sim_metrics_path = os.path.join(log_dir, "sim_reference_metrics.json")

            sim_num_steps = sim_cfg.num_steps
            if args_cli.sim_ref_steps_override is not None:
                sim_num_steps = args_cli.sim_ref_steps_override
            sim_num_envs = sim_cfg.num_envs
            if args_cli.sim_ref_num_envs_override is not None:
                sim_num_envs = args_cli.sim_ref_num_envs_override

            cmd = [
                sys.executable,
                "scripts/reinforcement_learning/model_based/sim_reference_eval.py",
                "--headless",
                "--task",
                sim_cfg.task,
                "--offline_task",
                args_cli.task,
                "--checkpoint",
                policy_ckpt,
                "--num_envs",
                str(sim_num_envs),
                "--num_steps",
                str(sim_num_steps),
                "--output_json",
                sim_metrics_path,
                "--device",
                str(config.experiment_config.device),
            ]
            print("[SimRef] Running simulator reference eval:")
            print(" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print("[SimRef] Evaluation failed.")
                if result.stderr:
                    print(result.stderr)
            elif os.path.exists(sim_metrics_path):
                with open(sim_metrics_path, "r", encoding="utf-8") as f:
                    sim_metrics = json.load(f)
                print(
                    "[SimRef] Summary: "
                    f"mean_episode_reward={sim_metrics.get('mean_episode_reward')}, "
                    f"mean_episode_length={sim_metrics.get('mean_episode_length')}, "
                    f"mean_step_reward={sim_metrics.get('mean_step_reward')}, "
                    f"finished_episodes={sim_metrics.get('num_finished_episodes')}"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "SimRef/mean_episode_reward": sim_metrics.get("mean_episode_reward", float("nan")),
                            "SimRef/mean_episode_length": sim_metrics.get("mean_episode_length", float("nan")),
                            "SimRef/mean_step_reward": sim_metrics.get("mean_step_reward", float("nan")),
                            "SimRef/num_finished_episodes": sim_metrics.get("num_finished_episodes", 0),
                        }
                    )
    print(f"Training completed. Artifacts saved to {log_dir}.")


def run(config: BaseConfig):
    wandb.init(project=config.experiment_name)
    wandb.config.update(config.to_dict())
    run_experiment(config)

def resolve_task_config(task: str):
    if task == "anymal_d_flat":
        config = AnymalDFlatConfig()
        return config
    if task == "lite3_flat":
        config = Lite3FlatConfig()
        return config
    lite3_preset_tasks = {
        "lite3_flat_wm_safe": "wm_safe",
        "lite3_flat_ftbest_ref": "ftbest_ref",
        "lite3_flat_ftbest_ref_u03": "ftbest_ref_u03",
        "lite3_flat_ftbest_track": "ftbest_track",
        "lite3_flat_ftbest_stable": "ftbest_stable",
        "lite3_flat_ftbest_recover": "ftbest_recover",
        "lite3_flat_ftbest_track_aggr": "ftbest_track_aggr",
        "lite3_flat_ftbest_track_aggr_u05": "ftbest_track_aggr_u05",
        "lite3_flat_ftbest_track_aggr_smooth": "ftbest_track_aggr_smooth",
        "lite3_flat_ftbest_track_aggr_gait": "ftbest_track_aggr_gait",
        "lite3_flat_ftbest_anti_knee": "ftbest_anti_knee",
    }
    if task in lite3_preset_tasks:
        preset = lite3_preset_tasks[task]
        config = make_lite3_flat_config(preset)
        return config
    else:
        valid = ["anymal_d_flat", "lite3_flat"] + [
            "lite3_flat_wm_safe",
            "lite3_flat_ftbest_ref",
            "lite3_flat_ftbest_ref_u03",
            "lite3_flat_ftbest_track",
            "lite3_flat_ftbest_stable",
            "lite3_flat_ftbest_recover",
            "lite3_flat_ftbest_track_aggr",
            "lite3_flat_ftbest_track_aggr_u05",
            "lite3_flat_ftbest_track_aggr_smooth",
            "lite3_flat_ftbest_track_aggr_gait",
            "lite3_flat_ftbest_anti_knee",
        ]
        valid_presets = ", ".join(sorted(LITE3_OFFLINE_PRESETS.keys()))
        raise ValueError(
            f"Unknown task: {task}. Valid tasks: {', '.join(valid)}. "
            f"Available Lite3 presets: {valid_presets}"
        )

def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got {value!r}."
        ) from exc
    if len(parsed) == 0:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def apply_paper_aligned_overrides(config: BaseConfig):
    """Match the offline RWM-U paper's main model/policy training scale."""
    # Offline data / imagination reset pool.
    config.data_config.batch_data_size = 6_000_000
    config.data_config.init_data_ratio = 1.0

    # World model training setup from the paper.
    config.model_architecture_config.history_horizon = 32
    config.model_architecture_config.forecast_horizon = 8
    config.model_architecture_config.ensemble_size = 5
    config.model_optimizer_config.learning_rate = 1.0e-4
    config.model_optimizer_config.weight_decay = 1.0e-5
    config.model_training_config.batch_size = 1024
    config.model_training_config.max_iterations = 2500
    config.model_training_config.random_batch_updates = True

    # Offline MOPO-PPO setup.
    config.environment_config.num_envs = 4096
    config.environment_config.uncertainty_penalty_weight = -1.0
    config.policy_architecture_config.actor_hidden_dims = [128, 128, 128]
    config.policy_architecture_config.critic_hidden_dims = [128, 128, 128]
    config.policy_algorithm_config.learning_rate = 1.0e-3
    config.policy_algorithm_config.entropy_coef = 0.005
    config.policy_algorithm_config.num_learning_epochs = 5
    config.policy_algorithm_config.num_mini_batches = 4
    config.policy_algorithm_config.clip_param = 0.2
    config.policy_algorithm_config.gamma = 0.99
    config.policy_algorithm_config.lam = 0.95
    config.policy_algorithm_config.value_loss_coef = 1.0
    config.policy_algorithm_config.max_grad_norm = 1.0
    config.policy_algorithm_config.use_clipped_value_loss = True
    config.policy_algorithm_config.schedule = "adaptive"
    config.policy_training_config.num_steps_per_env = 100
    config.policy_training_config.max_iterations = 2500
    config.policy_training_config.save_interval = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online learning training.")
    parser.add_argument("--task", type=str, default="anymal_d_flat", help="Task to use for the experiment.")
    parser.add_argument("--run_num", type=int, default=None, help="Run number for the experiment on the cluster.")
    parser.add_argument(
        "--mode",
        type=str,
        default="policy",
        choices=("policy", "model", "both"),
        help="Train only the offline policy, only the world model, or train the world model then policy.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional override for experiment_config.device.")
    parser.add_argument(
        "--paper_aligned_overrides",
        action="store_true",
        help=(
            "Use paper-aligned offline settings: 6M data/init pool, 4096 imagination envs, "
            "100 steps/env, 2500 policy iters, PPO lr 1e-3, entropy 0.005, and 128x3 MLPs."
        ),
    )
    parser.add_argument(
        "--max_iterations_override",
        type=int,
        default=None,
        help="Optional override for policy_training_config.max_iterations.",
    )
    parser.add_argument(
        "--model_max_iterations_override",
        type=int,
        default=None,
        help="Optional override for model_training_config.max_iterations.",
    )
    parser.add_argument(
        "--model_save_interval_override",
        type=int,
        default=None,
        help="Optional override for model_training_config.save_interval.",
    )
    parser.add_argument(
        "--model_resume_path",
        type=str,
        default=None,
        help="Optional model checkpoint path override. Use 'none' to train the world model from scratch.",
    )
    parser.add_argument(
        "--dataset_root_override",
        type=str,
        default=None,
        help="Optional override for data_config.dataset_root.",
    )
    parser.add_argument(
        "--dataset_folder_override",
        type=str,
        default=None,
        help="Optional override for data_config.dataset_folder.",
    )
    parser.add_argument(
        "--batch_data_size_override",
        type=int,
        default=None,
        help="Optional override for data_config.batch_data_size.",
    )
    parser.add_argument(
        "--file_data_size_override",
        type=int,
        default=None,
        help="Optional override for data_config.file_data_size.",
    )
    parser.add_argument(
        "--init_data_ratio_override",
        type=float,
        default=None,
        help="Optional override for data_config.init_data_ratio.",
    )
    parser.add_argument(
        "--policy_num_envs_override",
        type=int,
        default=None,
        help="Optional override for environment_config.num_envs used by offline policy imagination.",
    )
    parser.add_argument(
        "--policy_steps_per_env_override",
        type=int,
        default=None,
        help="Optional override for policy_training_config.num_steps_per_env.",
    )
    parser.add_argument(
        "--policy_learning_rate_override",
        type=float,
        default=None,
        help="Optional override for policy_algorithm_config.learning_rate.",
    )
    parser.add_argument(
        "--policy_entropy_coef_override",
        type=float,
        default=None,
        help="Optional override for policy_algorithm_config.entropy_coef.",
    )
    parser.add_argument(
        "--policy_actor_hidden_dims_override",
        type=_parse_int_list,
        default=None,
        help="Optional comma-separated actor hidden dims, e.g. 128,128,128.",
    )
    parser.add_argument(
        "--policy_critic_hidden_dims_override",
        type=_parse_int_list,
        default=None,
        help="Optional comma-separated critic hidden dims, e.g. 128,128,128.",
    )
    parser.add_argument(
        "--uncertainty_penalty_weight_override",
        type=float,
        default=None,
        help="Optional override for environment_config.uncertainty_penalty_weight.",
    )
    parser.add_argument(
        "--sim_ref_steps_override",
        type=int,
        default=None,
        help="Optional override for simulator reference evaluation rollout steps.",
    )
    parser.add_argument(
        "--sim_ref_num_envs_override",
        type=int,
        default=None,
        help="Optional override for simulator reference evaluation num_envs.",
    )
    args_cli = parser.parse_args()
    config = resolve_task_config(args_cli.task)
    if args_cli.paper_aligned_overrides:
        apply_paper_aligned_overrides(config)
    if args_cli.device is not None:
        config.experiment_config.device = args_cli.device
    if args_cli.max_iterations_override is not None:
        config.policy_training_config.max_iterations = args_cli.max_iterations_override
    if args_cli.model_max_iterations_override is not None:
        config.model_training_config.max_iterations = args_cli.model_max_iterations_override
    if args_cli.model_save_interval_override is not None:
        config.model_training_config.save_interval = args_cli.model_save_interval_override
    if args_cli.model_resume_path is not None:
        config.model_architecture_config.resume_path = None if args_cli.model_resume_path.lower() == "none" else args_cli.model_resume_path
    if args_cli.dataset_root_override is not None:
        config.data_config.dataset_root = args_cli.dataset_root_override
    if args_cli.dataset_folder_override is not None:
        config.data_config.dataset_folder = args_cli.dataset_folder_override
    if args_cli.batch_data_size_override is not None:
        config.data_config.batch_data_size = args_cli.batch_data_size_override
    if args_cli.file_data_size_override is not None:
        config.data_config.file_data_size = args_cli.file_data_size_override
    if args_cli.init_data_ratio_override is not None:
        config.data_config.init_data_ratio = args_cli.init_data_ratio_override
    if args_cli.policy_num_envs_override is not None:
        config.environment_config.num_envs = args_cli.policy_num_envs_override
    if args_cli.policy_steps_per_env_override is not None:
        config.policy_training_config.num_steps_per_env = args_cli.policy_steps_per_env_override
    if args_cli.policy_learning_rate_override is not None:
        config.policy_algorithm_config.learning_rate = args_cli.policy_learning_rate_override
    if args_cli.policy_entropy_coef_override is not None:
        config.policy_algorithm_config.entropy_coef = args_cli.policy_entropy_coef_override
    if args_cli.policy_actor_hidden_dims_override is not None:
        config.policy_architecture_config.actor_hidden_dims = args_cli.policy_actor_hidden_dims_override
    if args_cli.policy_critic_hidden_dims_override is not None:
        config.policy_architecture_config.critic_hidden_dims = args_cli.policy_critic_hidden_dims_override
    if args_cli.uncertainty_penalty_weight_override is not None:
        config.environment_config.uncertainty_penalty_weight = args_cli.uncertainty_penalty_weight_override
    run(config)
