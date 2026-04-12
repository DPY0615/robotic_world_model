"""Export Lite3 simulator rollouts to offline `state_action_data_*.csv` files.

The exported row layout matches the offline model-based loader exactly:

    [system_state(45), system_action(12), system_contact(8), effective_termination(1)]

For Lite3 this expands to:

    state:
        [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
         joint_pos_rel(12), joint_vel_rel(12), joint_torque(12)]
    action:
        [last_action(12)]
    contact:
        [thigh_contact(4), foot_contact(4)]
    termination:
        [effective_termination(1)]

`effective_termination` is intentionally not the raw online `system_termination`
group. For Lite3 offline data we export:

    effective_termination = torso_contact OR bad_orientation_2

so dataset windowing respects Lite3's real termination semantics instead of only
hard torso contact.

The script intentionally captures the `system_*` observation groups *before*
terminated environments are reset, so the exported rows stay aligned with the
termination flags expected by the offline world-model pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export Lite3 simulator rollouts to offline CSV dataset files.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0",
    help="Gym task used to generate the dataset. Use the Lite3 pretrain task by default.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated envs. Only 1 is supported here.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps_per_file", type=int, default=10000, help="Rows written to each CSV file.")
parser.add_argument("--num_files", type=int, default=1, help="How many CSV files to export.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="assets/data/lite3",
    help="Directory where state_action_data_*.csv files are written.",
)
parser.add_argument(
    "--start_index",
    type=int,
    default=None,
    help="Optional first file index. If omitted, the next free index is used.",
)
parser.add_argument(
    "--action_source",
    type=str,
    default="random",
    choices=("random", "zero"),
    help="How to generate actions. Random samples uniform actions in [-1, 1].",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import mbrl.tasks  # noqa: F401


def _next_file_index(output_dir: Path) -> int:
    max_index = -1
    for file_path in output_dir.glob("state_action_data_*.csv"):
        suffix = file_path.stem.removeprefix("state_action_data_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _sample_actions(action_dim: int, device: torch.device, source: str) -> torch.Tensor:
    if source == "zero":
        return torch.zeros((1, action_dim), device=device)
    return 2.0 * torch.rand((1, action_dim), device=device) - 1.0


def _pre_reset_step(env, action: torch.Tensor):
    """Step the env once and capture system groups before automatic resets.

    Isaac Lab computes terminations and then resets envs before writing the next
    observation buffer. For offline dataset export we want the *terminal-step*
    state/contact/termination tuple, so we mirror the step structure and grab
    the `system_*` groups before `_reset_idx(...)` runs.
    """

    env.action_manager.process_action(action.to(env.device))

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()

    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    env.episode_length_buf += 1
    env.common_step_counter += 1

    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    groups = env.observation_manager.compute()

    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        env._reset_idx(reset_env_ids)
        if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
            env.sim.render()

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)

    env.obs_buf = env.observation_manager.compute()
    return groups


def _compute_effective_termination(groups: dict[str, torch.Tensor]) -> torch.Tensor:
    projected_gravity = groups["system_state"][:, 6:9].to(torch.float32)
    torso_contact = groups["system_termination"].to(torch.float32) > 0.5
    bad_orientation_2 = (projected_gravity[:, 2] > 0.0) | (projected_gravity[:, :2].abs() > 0.7).any(dim=-1)
    return (torso_contact.squeeze(-1) | bad_orientation_2).to(torch.float32).unsqueeze(-1)


def _groups_to_row(groups) -> list[float]:
    state = groups["system_state"].to(torch.float32)
    action = groups["system_action"].to(torch.float32)
    contact = groups["system_contact"].to(torch.float32)
    termination = _compute_effective_termination(groups)
    row = torch.cat([state, action, contact, termination], dim=-1)
    return row[0].detach().cpu().tolist()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs != 1:
        raise ValueError("This exporter currently supports only `--num_envs 1` so each CSV remains a single trajectory stream.")

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_index = args_cli.start_index if args_cli.start_index is not None else _next_file_index(output_dir)

    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped
    raw_env.reset()

    action_dim = raw_env.action_space.shape[-1]
    expected_dim = (
        raw_env.observation_manager.group_obs_dim["system_state"][0]
        + raw_env.observation_manager.group_obs_dim["system_action"][0]
        + raw_env.observation_manager.group_obs_dim["system_contact"][0]
        + raw_env.observation_manager.group_obs_dim["system_termination"][0]
    )

    print(f"[INFO] Exporting Lite3 dataset with {expected_dim} columns per row.")
    print(f"[INFO] Output directory: {output_dir.resolve()}")
    print(f"[INFO] Starting file index: {file_index}")

    for file_offset in range(args_cli.num_files):
        file_path = output_dir / f"state_action_data_{file_index + file_offset}.csv"
        print(f"[INFO] Writing {file_path} ...")
        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            for step in range(args_cli.steps_per_file):
                with torch.inference_mode():
                    action = _sample_actions(action_dim, raw_env.device, args_cli.action_source)
                    groups = _pre_reset_step(raw_env, action)
                    row = _groups_to_row(groups)
                    writer.writerow(row)
                if (step + 1) % 1000 == 0 or (step + 1) == args_cli.steps_per_file:
                    print(f"[INFO]   {step + 1}/{args_cli.steps_per_file} rows written")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
