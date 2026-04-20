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
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


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
NORMALIZER_METADATA_FILE = "normalizer_pretrain_defaults.json"


parser = argparse.ArgumentParser(description="Export Lite3 simulator rollouts to offline CSV dataset files.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0",
    help="Gym task used to generate the dataset. Use the Lite3 pretrain task by default.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=128,
    help="Number of simulated envs. Each env is exported as an independent trajectory CSV.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps_per_file", type=int, default=10000, help="Rows written to each CSV file.")
parser.add_argument(
    "--num_files",
    type=int,
    default=1,
    help="How many rollout shards to export. Total CSV files = num_files * num_envs.",
)
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
    choices=("random", "zero", "policy"),
    help="How to generate actions. Policy uses --policy_checkpoint; random samples uniform actions in [-1, 1].",
)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default=None,
    help="RSL-RL policy checkpoint used when --action_source policy.",
)
parser.add_argument(
    "--policy_stochastic",
    action="store_true",
    help="Sample from the policy distribution instead of using act_inference.",
)
parser.add_argument(
    "--action_mix_random_prob",
    type=float,
    default=0.05,
    help="Per-env probability of replacing a policy action with a random action for coverage.",
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
from rsl_rl.modules import ActorCritic

import isaaclab_tasks  # noqa: F401
import mbrl.tasks  # noqa: F401


def _next_file_index(output_dir: Path) -> int:
    max_index = -1
    for file_path in output_dir.glob("state_action_data_*.csv"):
        suffix = file_path.stem.removeprefix("state_action_data_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _write_normalizer_metadata(output_dir: Path) -> None:
    metadata_path = output_dir / NORMALIZER_METADATA_FILE
    payload = {
        "source": "explicit_pretrain_defaults",
        "collection": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps_per_file": args_cli.steps_per_file,
            "num_files": args_cli.num_files,
            "action_source": args_cli.action_source,
            "policy_checkpoint": args_cli.policy_checkpoint,
            "policy_stochastic": args_cli.policy_stochastic,
            "action_mix_random_prob": args_cli.action_mix_random_prob,
        },
        "state_data_mean": PRETRAIN_STATE_MEAN,
        "state_data_std": PRETRAIN_STATE_STD,
        "action_data_mean": PRETRAIN_ACTION_MEAN,
        "action_data_std": PRETRAIN_ACTION_STD,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Wrote normalizer metadata: {metadata_path}")


def _normalize_reset_output(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _sanitize_obs_groups(obs, obs_groups):
    if not isinstance(obs, dict):
        return {"policy": ["policy"], "critic": ["policy"]}
    available = list(obs.keys())
    if len(available) == 0:
        return {"policy": ["policy"], "critic": ["policy"]}
    if not hasattr(obs_groups, "get"):
        policy_groups = ["policy"] if "policy" in obs else [available[0]]
        critic_groups = ["critic"] if "critic" in obs else policy_groups
        return {"policy": policy_groups, "critic": critic_groups}
    policy_groups = [key for key in obs_groups.get("policy", []) if key in obs]
    if len(policy_groups) == 0:
        policy_groups = [available[0]]
    critic_groups = [key for key in obs_groups.get("critic", []) if key in obs]
    if len(critic_groups) == 0:
        critic_groups = policy_groups
    return {"policy": policy_groups, "critic": critic_groups}


def _default_policy_obs_groups(obs):
    if not isinstance(obs, dict):
        return {"policy": ["policy"], "critic": ["policy"]}
    available = list(obs.keys())
    if len(available) == 0:
        return {"policy": ["policy"], "critic": ["policy"]}
    policy_group = ["policy"] if "policy" in obs else [available[0]]
    critic_group = ["critic"] if "critic" in obs else policy_group
    return {"policy": policy_group, "critic": critic_group}


def _sample_actions(num_envs: int, action_dim: int, device: torch.device, source: str) -> torch.Tensor:
    if source == "zero":
        return torch.zeros((num_envs, action_dim), device=device)
    return 2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0


def _build_policy(obs, action_dim: int, agent_cfg, device: torch.device) -> ActorCritic:
    if args_cli.policy_checkpoint is None:
        raise ValueError("`--action_source policy` requires `--policy_checkpoint`.")

    policy_cfg = agent_cfg.policy
    obs_groups_cfg = getattr(policy_cfg, "obs_groups", None)
    if obs_groups_cfg is None:
        obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
    if obs_groups_cfg is None or not hasattr(obs_groups_cfg, "get"):
        obs_groups_cfg = _default_policy_obs_groups(obs)
    obs_groups = _sanitize_obs_groups(obs, obs_groups_cfg)
    actor_critic = ActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=action_dim,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std,
        noise_std_type=getattr(policy_cfg, "noise_std_type", "scalar"),
    ).to(device)

    checkpoint = torch.load(args_cli.policy_checkpoint, map_location=device)
    checkpoint_state = checkpoint.get("model_state_dict", checkpoint.get("policy_state_dict", checkpoint))
    try:
        actor_critic.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        current_state = actor_critic.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        missing_actor_keys = [
            key
            for key in current_state
            if (key.startswith("actor.") or key in ("std", "log_std")) and key not in compatible_state
        ]
        if missing_actor_keys:
            raise RuntimeError(
                "Policy checkpoint actor weights are incompatible with the exporter observation/action setup. "
                f"Missing compatible actor keys: {missing_actor_keys}"
            ) from exc
        actor_critic.load_state_dict(compatible_state, strict=False)
        skipped_keys = sorted(set(checkpoint_state) - set(compatible_state))
        print(
            "[WARN] Loaded policy checkpoint with non-strict compatible weights. "
            f"Skipped {len(skipped_keys)} incompatible/nonexistent keys, mostly critic-only weights."
        )
    actor_critic.eval()
    print(
        f"[INFO] Loaded policy checkpoint {args_cli.policy_checkpoint} "
        f"(iter={checkpoint.get('iter', 'unknown')})."
    )
    return actor_critic


def _policy_actions(actor_critic: ActorCritic, obs, num_envs: int, action_dim: int, device: torch.device) -> torch.Tensor:
    if args_cli.policy_stochastic:
        actions = actor_critic.act(obs)
    else:
        actions = actor_critic.act_inference(obs)

    mix_prob = args_cli.action_mix_random_prob
    if mix_prob > 0.0:
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError("`--action_mix_random_prob` must be in [0, 1].")
        random_actions = _sample_actions(num_envs, action_dim, device, "random")
        random_mask = torch.rand((num_envs, 1), device=device) < mix_prob
        actions = torch.where(random_mask, random_actions, actions)
    return actions


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


def _groups_to_rows(groups) -> list[list[float]]:
    state = groups["system_state"].to(torch.float32)
    action = groups["system_action"].to(torch.float32)
    contact = groups["system_contact"].to(torch.float32)
    termination = _compute_effective_termination(groups)
    row = torch.cat([state, action, contact, termination], dim=-1)
    return row.detach().cpu().tolist()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_normalizer_metadata(output_dir)
    file_index = args_cli.start_index if args_cli.start_index is not None else _next_file_index(output_dir)

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped
    obs = _normalize_reset_output(raw_env.reset())

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
    print(f"[INFO] Total CSV files to write: {args_cli.num_files * raw_env.num_envs}")

    actor_critic = None
    if args_cli.action_source == "policy":
        actor_critic = _build_policy(obs, action_dim, agent_cfg, raw_env.device)

    for file_offset in range(args_cli.num_files):
        shard_start = file_index + file_offset * raw_env.num_envs
        file_paths = [output_dir / f"state_action_data_{shard_start + env_id}.csv" for env_id in range(raw_env.num_envs)]
        print(f"[INFO] Writing shard {file_offset + 1}/{args_cli.num_files}: {file_paths[0]} ... {file_paths[-1]}")
        csv_files = [file_path.open("w", newline="", encoding="utf-8") for file_path in file_paths]
        try:
            writers = [csv.writer(csv_file) for csv_file in csv_files]
            for step in range(args_cli.steps_per_file):
                with torch.inference_mode():
                    if args_cli.action_source == "policy":
                        action = _policy_actions(actor_critic, obs, raw_env.num_envs, action_dim, raw_env.device)
                    else:
                        action = _sample_actions(raw_env.num_envs, action_dim, raw_env.device, args_cli.action_source)
                    groups = _pre_reset_step(raw_env, action)
                    rows = _groups_to_rows(groups)
                    for writer, row in zip(writers, rows, strict=True):
                        writer.writerow(row)
                    obs = raw_env.obs_buf
                if (step + 1) % 1000 == 0 or (step + 1) == args_cli.steps_per_file:
                    print(f"[INFO]   {step + 1}/{args_cli.steps_per_file} steps written per env")
        finally:
            for csv_file in csv_files:
                csv_file.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
