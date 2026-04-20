from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an offline policy checkpoint on a simulator reference env.")
parser.add_argument(
	"--task",
	type=str,
	default="Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0",
	help="Simulator gym task used as reference.",
)
parser.add_argument(
	"--offline_task",
	type=str,
	default="lite3_flat",
	help="Offline task key used to build ActorCritic architecture.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to offline policy checkpoint (*.pt).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of simulator envs for reference evaluation.")
parser.add_argument("--num_steps", type=int, default=600, help="Number of rollout steps for evaluation.")
parser.add_argument("--seed", type=int, default=None, help="Optional env seed.")
parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy actions.")
parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON file for metrics.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.modules import ActorCritic

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

from configs import Lite3FlatConfig, make_lite3_flat_config

import isaaclab_tasks  # noqa: F401
import mbrl.tasks  # noqa: F401


def resolve_offline_config(task: str):
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
		return make_lite3_flat_config(lite3_preset_tasks[task])
	if task == "lite3_flat":
		return Lite3FlatConfig()
	# fallback to default Lite3 config for unknown keys
	return Lite3FlatConfig()


def normalize_reset_output(reset_output):
	if isinstance(reset_output, tuple):
		return reset_output[0]
	return reset_output


def sanitize_obs_groups(obs, obs_groups):
	if not isinstance(obs, dict):
		return {"policy": ["policy"], "critic": ["policy"]}
	available = list(obs.keys())
	if len(available) == 0:
		return {"policy": ["policy"], "critic": ["policy"]}
	policy_groups = [k for k in obs_groups.get("policy", []) if k in obs]
	if len(policy_groups) == 0:
		policy_groups = [available[0]]
	critic_groups = [k for k in obs_groups.get("critic", []) if k in obs]
	if len(critic_groups) == 0:
		critic_groups = policy_groups
	return {"policy": policy_groups, "critic": critic_groups}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
	cfg = resolve_offline_config(args_cli.offline_task)
	policy_cfg = cfg.policy_architecture_config

	env_cfg.scene.num_envs = args_cli.num_envs
	if args_cli.seed is not None:
		env_cfg.seed = args_cli.seed
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	env = gym.make(args_cli.task, cfg=env_cfg)
	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)

	obs = normalize_reset_output(env.reset())
	obs_groups = sanitize_obs_groups(obs, policy_cfg.obs_groups)
	actor_critic = ActorCritic(
		obs=obs,
		obs_groups=obs_groups,
		num_actions=policy_cfg.action_dim,
		actor_hidden_dims=policy_cfg.actor_hidden_dims,
		critic_hidden_dims=policy_cfg.critic_hidden_dims,
		activation=policy_cfg.activation,
		init_noise_std=policy_cfg.init_noise_std,
		noise_std_type=policy_cfg.noise_std_type,
	).to(env.unwrapped.device)

	checkpoint = torch.load(args_cli.checkpoint, map_location=env.unwrapped.device)
	actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
	actor_critic.eval()

	episode_reward = torch.zeros(env.unwrapped.num_envs, dtype=torch.float32, device=env.unwrapped.device)
	episode_length = torch.zeros(env.unwrapped.num_envs, dtype=torch.float32, device=env.unwrapped.device)
	finished_rewards = []
	finished_lengths = []
	total_reward_sum = 0.0

	with torch.inference_mode():
		for _ in range(args_cli.num_steps):
			if args_cli.deterministic:
				actions = actor_critic.act_inference(obs)
			else:
				actions = actor_critic.act(obs)

			obs, rewards, terminated, truncated, _ = env.step(actions)
			done = (terminated | truncated).to(torch.bool)

			episode_reward += rewards
			episode_length += 1
			total_reward_sum += rewards.sum().item()

			done_ids = done.nonzero(as_tuple=False).squeeze(-1)
			if done_ids.numel() > 0:
				finished_rewards.extend(episode_reward[done_ids].detach().cpu().tolist())
				finished_lengths.extend(episode_length[done_ids].detach().cpu().tolist())
				episode_reward[done_ids] = 0.0
				episode_length[done_ids] = 0.0

	env.close()

	if len(finished_rewards) > 0:
		mean_ep_reward = float(sum(finished_rewards) / len(finished_rewards))
		mean_ep_length = float(sum(finished_lengths) / len(finished_lengths))
		used_partial_stats = False
	else:
		# If no episode ended in the evaluation horizon, fall back to partial episode means.
		mean_ep_reward = float(episode_reward.mean().item())
		mean_ep_length = float(episode_length.mean().item())
		used_partial_stats = True
	mean_step_reward = total_reward_sum / (args_cli.num_steps * args_cli.num_envs)

	metrics = {
		"offline_task": args_cli.offline_task,
		"sim_task": args_cli.task,
		"checkpoint": args_cli.checkpoint,
		"num_envs": args_cli.num_envs,
		"num_steps": args_cli.num_steps,
		"num_finished_episodes": len(finished_rewards),
		"mean_episode_reward": mean_ep_reward,
		"mean_episode_length": mean_ep_length,
		"mean_step_reward": mean_step_reward,
		"used_partial_episode_stats": used_partial_stats,
	}

	print(json.dumps(metrics, indent=2, ensure_ascii=False))
	if args_cli.output_json is not None:
		output_path = Path(args_cli.output_json)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
	main()
	simulation_app.close()
