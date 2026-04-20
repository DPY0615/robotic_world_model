"""Train a standalone world model with online simulator interaction.

This script follows the online pretrain flow (collecting transitions from Isaac
Sim through MBPOOnPolicyRunner) but aligns the system dynamics architecture
with the offline model-based chain and exports standalone world-model
checkpoints that contain only ``system_dynamics_state_dict`` plus the matching
normalization/config metadata.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


PRETRAIN_STATE_MEAN = [
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
PRETRAIN_STATE_STD = [
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
PRETRAIN_ACTION_MEAN = [
    0.2, 0.5, 1.1,
    -0.3, -0.1, 1.2,
    0.3, -0.9, 0.6,
    -0.3, -1.1, 0.7,
]
PRETRAIN_ACTION_STD = [
    1.1, 1.5, 1.5,
    1.1, 1.3, 1.6,
    1.2, 1.2, 1.4,
    1.1, 1.3, 1.5,
]


parser = argparse.ArgumentParser(description="Train a standalone world model with online simulator interaction.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0",
    help="Name of the task.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=2500, help="World-model training iterations.")
parser.add_argument("--save_interval", type=int, default=200, help="Combined/standalone checkpoint interval.")
parser.add_argument("--ensemble_size", type=int, default=5, help="Offline-aligned world-model ensemble size.")
parser.add_argument(
    "--system_dynamics_learning_rate",
    type=float,
    default=1.0e-4,
    help="Learning rate for the world model optimizer.",
)
parser.add_argument(
    "--system_dynamics_weight_decay",
    type=float,
    default=1.0e-5,
    help="Weight decay for the world model optimizer.",
)
parser.add_argument(
    "--system_dynamics_mini_batch_size",
    type=int,
    default=1024,
    help="Mini-batch size for world-model updates.",
)
parser.add_argument(
    "--system_dynamics_num_mini_batches",
    type=int,
    default=20,
    help="Number of mini-batches per world-model update step.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.run_name is None:
    args_cli.run_name = "world_model_online"

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401
from rsl_rl.runners import MBPOOnPolicyRunner


MODEL_BASED_DIR = Path(__file__).resolve().parents[1] / "model_based"
if str(MODEL_BASED_DIR) not in sys.path:
    sys.path.append(str(MODEL_BASED_DIR))

from configs import Lite3FlatConfig  # noqa: E402


def _align_world_model_cfg(agent_cfg):
    """Force the online runner to use the offline-aligned world-model config."""
    offline_cfg = Lite3FlatConfig()

    agent_cfg.run_name = args_cli.run_name
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.resume = False
    agent_cfg.load_system_dynamics = False
    agent_cfg.system_dynamics_load_path = None

    agent_cfg.system_dynamics.history_horizon = offline_cfg.model_architecture_config.history_horizon
    agent_cfg.system_dynamics.ensemble_size = args_cli.ensemble_size
    agent_cfg.system_dynamics.architecture_config = dict(offline_cfg.model_architecture_config.architecture_config)
    agent_cfg.system_dynamics.freeze_auxiliary = offline_cfg.model_architecture_config.freeze_auxiliary

    agent_cfg.algorithm.system_dynamics_learning_rate = args_cli.system_dynamics_learning_rate
    agent_cfg.algorithm.system_dynamics_weight_decay = args_cli.system_dynamics_weight_decay
    agent_cfg.algorithm.system_dynamics_forecast_horizon = offline_cfg.model_architecture_config.forecast_horizon
    agent_cfg.algorithm.system_dynamics_mini_batch_size = args_cli.system_dynamics_mini_batch_size
    agent_cfg.algorithm.system_dynamics_num_mini_batches = args_cli.system_dynamics_num_mini_batches
    agent_cfg.algorithm.system_dynamics_eval_traj_noise_scale = [0.1]
    agent_cfg.algorithm.system_dynamics_num_eval_trajectories = offline_cfg.data_config.num_eval_trajectories
    agent_cfg.algorithm.system_dynamics_len_eval_trajectory = offline_cfg.data_config.len_eval_trajectory

    # Explicitly pin to pretrain defaults so normalizer source is unambiguous.
    agent_cfg.imagination.state_normalizer.mean = list(PRETRAIN_STATE_MEAN)
    agent_cfg.imagination.state_normalizer.std = list(PRETRAIN_STATE_STD)
    agent_cfg.imagination.action_normalizer.mean = list(PRETRAIN_ACTION_MEAN)
    agent_cfg.imagination.action_normalizer.std = list(PRETRAIN_ACTION_STD)

    offline_cfg.data_config.state_data_mean = list(PRETRAIN_STATE_MEAN)
    offline_cfg.data_config.state_data_std = list(PRETRAIN_STATE_STD)
    offline_cfg.data_config.action_data_mean = list(PRETRAIN_ACTION_MEAN)
    offline_cfg.data_config.action_data_std = list(PRETRAIN_ACTION_STD)

    agent_cfg.system_dynamics_num_visualizations = offline_cfg.data_config.num_visualizations
    agent_cfg.system_dynamics_state_idx_dict = dict(offline_cfg.data_config.state_idx_dict)
    return agent_cfg, offline_cfg


def _export_standalone_world_models(log_dir: str, task_name: str, agent_cfg, offline_cfg):
    output_dir = Path(log_dir) / "world_model_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_checkpoints = sorted(Path(log_dir).glob("model_*.pt"))
    if not combined_checkpoints:
        raise FileNotFoundError(f"No combined checkpoints found under {log_dir}")

    for checkpoint_path in combined_checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        output_path = output_dir / checkpoint_path.name
        payload = {
            "iter": checkpoint.get("iter"),
            "task": task_name,
            "system_dynamics_state_dict": checkpoint["system_dynamics_state_dict"],
            "state_data_mean": torch.tensor(agent_cfg.imagination.state_normalizer.mean, dtype=torch.float32),
            "state_data_std": torch.tensor(agent_cfg.imagination.state_normalizer.std, dtype=torch.float32),
            "action_data_mean": torch.tensor(agent_cfg.imagination.action_normalizer.mean, dtype=torch.float32),
            "action_data_std": torch.tensor(agent_cfg.imagination.action_normalizer.std, dtype=torch.float32),
            "history_horizon": agent_cfg.system_dynamics.history_horizon,
            "forecast_horizon": agent_cfg.algorithm.system_dynamics_forecast_horizon,
            "ensemble_size": agent_cfg.system_dynamics.ensemble_size,
            "architecture_config": dict(agent_cfg.system_dynamics.architecture_config),
            "state_idx_dict": dict(agent_cfg.system_dynamics_state_idx_dict),
            "offline_model_architecture_config": offline_cfg.model_architecture_config.to_dict(),
        }
        torch.save(payload, output_path)

    latest_payload = torch.load(output_dir / combined_checkpoints[-1].name, map_location="cpu")
    torch.save(latest_payload, output_dir / "world_model_final.pt")

    print(f"[INFO] Exported {len(combined_checkpoints)} standalone world-model checkpoints to: {output_dir}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg, offline_cfg = _align_world_model_cfg(agent_cfg)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    agent_cfg.seed = env_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MBPOOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print("[INFO] Standalone world-model training config:")
    print(
        f"history_horizon={agent_cfg.system_dynamics.history_horizon}, "
        f"forecast_horizon={agent_cfg.algorithm.system_dynamics_forecast_horizon}, "
        f"ensemble_size={agent_cfg.system_dynamics.ensemble_size}, "
        f"rnn={agent_cfg.system_dynamics.architecture_config}"
    )
    print(
        f"lr={agent_cfg.algorithm.system_dynamics_learning_rate}, "
        f"weight_decay={agent_cfg.algorithm.system_dynamics_weight_decay}, "
        f"mini_batch_size={agent_cfg.algorithm.system_dynamics_mini_batch_size}, "
        f"num_mini_batches={agent_cfg.algorithm.system_dynamics_num_mini_batches}, "
        f"save_interval={agent_cfg.save_interval}, "
        f"max_iterations={agent_cfg.max_iterations}"
    )
    print(
        f"[INFO] Approx. simulator transitions = "
        f"{env_cfg.scene.num_envs} envs * {agent_cfg.num_steps_per_env} steps/iter * {agent_cfg.max_iterations} iters "
        f"= {env_cfg.scene.num_envs * agent_cfg.num_steps_per_env * agent_cfg.max_iterations}"
    )
    print(
        "[INFO] Using explicit pretrain normalizer defaults "
        f"(state_dim={len(PRETRAIN_STATE_MEAN)}, action_dim={len(PRETRAIN_ACTION_MEAN)})."
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    _export_standalone_world_models(log_dir, args_cli.task, agent_cfg, offline_cfg)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
