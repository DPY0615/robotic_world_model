from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

from configs import AnymalDFlatConfig, Lite3FlatConfig, make_lite3_flat_config


STATE_DIM = 45
ACTION_DIM = 12
EXTENSION_DIM = 0

STATE_GROUPS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "joint_pos": slice(9, 21),
    "joint_vel": slice(21, 33),
    "joint_torque": slice(33, 45),
}

LITE3_PRESET_TASKS = {
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


def _load_system_dynamics_cls(rsl_rl_root: str | None):
    try:
        from rsl_rl.modules import SystemDynamicsEnsemble

        return SystemDynamicsEnsemble
    except ModuleNotFoundError:
        pass

    candidate_roots = []
    if rsl_rl_root is not None:
        candidate_roots.append(Path(rsl_rl_root))
    if os.environ.get("RSL_RL_ROOT"):
        candidate_roots.append(Path(os.environ["RSL_RL_ROOT"]))
    candidate_roots.append(Path.cwd().parent / "rsl_rl_rwm" / "rsl_rl")

    root = next((path for path in candidate_roots if (path / "modules" / "system_dynamics.py").exists()), None)
    if root is None:
        raise ModuleNotFoundError(
            "Could not import rsl_rl SystemDynamicsEnsemble. Install rsl_rl or pass --rsl_rl_root."
        )

    rsl_pkg = types.ModuleType("rsl_rl")
    rsl_pkg.__path__ = [str(root)]
    sys.modules["rsl_rl"] = rsl_pkg

    modules_pkg = types.ModuleType("rsl_rl.modules")
    modules_pkg.__path__ = [str(root / "modules")]
    sys.modules["rsl_rl.modules"] = modules_pkg

    arch_dir = root / "modules" / "architectures"
    arch_spec = importlib.util.spec_from_file_location(
        "rsl_rl.modules.architectures",
        arch_dir / "__init__.py",
        submodule_search_locations=[str(arch_dir)],
    )
    arch_mod = importlib.util.module_from_spec(arch_spec)
    sys.modules["rsl_rl.modules.architectures"] = arch_mod
    arch_spec.loader.exec_module(arch_mod)

    sd_spec = importlib.util.spec_from_file_location(
        "rsl_rl.modules.system_dynamics",
        root / "modules" / "system_dynamics.py",
    )
    sd_mod = importlib.util.module_from_spec(sd_spec)
    sys.modules["rsl_rl.modules.system_dynamics"] = sd_mod
    sd_spec.loader.exec_module(sd_mod)
    return sd_mod.SystemDynamicsEnsemble


def resolve_config(task: str):
    if task == "anymal_d_flat":
        return AnymalDFlatConfig()
    if task == "lite3_flat":
        return Lite3FlatConfig()
    if task in LITE3_PRESET_TASKS:
        return make_lite3_flat_config(LITE3_PRESET_TASKS[task])
    valid = ["anymal_d_flat", "lite3_flat", *sorted(LITE3_PRESET_TASKS.keys())]
    raise ValueError(f"Unknown task {task!r}. Valid tasks: {', '.join(valid)}")


def default_data_path(config) -> Path:
    data_cfg = config.data_config
    return Path(data_cfg.dataset_root) / data_cfg.dataset_folder / "state_action_data_0.csv"


def default_checkpoint_path(config) -> Path:
    path = config.model_architecture_config.resume_path
    if path is None:
        raise ValueError("No checkpoint path in config. Pass --checkpoint.")
    return Path(path)


def load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[None, :] if data.ndim == 1 else data


def make_windows(data: np.ndarray, history_horizon: int, forecast_horizon: int, num_windows: int, seed: int):
    max_start = data.shape[0] - history_horizon - forecast_horizon + 1
    if max_start <= 0:
        raise ValueError("Dataset is shorter than history_horizon + forecast_horizon.")

    starts = np.arange(max_start)
    termination = data[:, STATE_DIM + ACTION_DIM + EXTENSION_DIM + 8 : STATE_DIM + ACTION_DIM + EXTENSION_DIM + 9]
    if termination.shape[1] > 0:
        term = termination[:, 0] > 0.5
        prefix = np.concatenate([[0], np.cumsum(term.astype(np.int64))])
        window_horizon = history_horizon + forecast_horizon
        valid = (prefix[starts + window_horizon - 1] - prefix[starts]) == 0
        starts = starts[valid]
        if starts.shape[0] == 0:
            raise ValueError("No valid windows remain after filtering termination crossings.")

    if num_windows > 0 and starts.shape[0] > num_windows:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, size=num_windows, replace=False))

    indices = starts[:, None] + np.arange(history_horizon + forecast_horizon)[None, :]
    state = torch.from_numpy(data[indices, :STATE_DIM])
    action = torch.from_numpy(data[indices, STATE_DIM : STATE_DIM + ACTION_DIM])

    offset = STATE_DIM + ACTION_DIM + EXTENSION_DIM
    contact = torch.from_numpy(data[indices, offset : offset + 8])
    termination = torch.from_numpy(data[indices, offset + 8 : offset + 9])
    return state, action, contact, termination


def build_model(system_dynamics_cls, config, checkpoint_path: Path, device: str):
    model_cfg = config.model_architecture_config
    model = system_dynamics_cls(
        STATE_DIM,
        ACTION_DIM,
        EXTENSION_DIM,
        model_cfg.contact_dim,
        model_cfg.termination_dim,
        device,
        ensemble_size=model_cfg.ensemble_size,
        history_horizon=model_cfg.history_horizon,
        architecture_config=model_cfg.architecture_config,
        freeze_auxiliary=model_cfg.freeze_auxiliary,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["system_dynamics_state_dict"], strict=True)
    model.eval()
    return model, checkpoint.get("iter")


def normalize(state: torch.Tensor, action: torch.Tensor, config, device: str):
    data_cfg = config.data_config
    state_mean = torch.tensor(data_cfg.state_data_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(data_cfg.state_data_std, dtype=torch.float32, device=device)
    action_mean = torch.tensor(data_cfg.action_data_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(data_cfg.action_data_std, dtype=torch.float32, device=device)
    return (state.to(device) - state_mean) / state_std, (action.to(device) - action_mean) / action_std, state_std


@torch.no_grad()
def evaluate(config, data_path: Path, checkpoint_path: Path, device: str, batch_size: int, num_windows: int, seed: int, rsl_rl_root: str | None):
    system_dynamics_cls = _load_system_dynamics_cls(rsl_rl_root)
    model, checkpoint_iter = build_model(system_dynamics_cls, config, checkpoint_path, device)

    horizon = config.model_architecture_config.history_horizon
    forecast = config.model_architecture_config.forecast_horizon
    state, action, contact, termination = make_windows(load_csv(data_path), horizon, forecast, num_windows, seed)
    state, action, state_std = normalize(state, action, config, device)
    contact = contact.to(device)
    termination = termination.to(device)

    state_squared_error = torch.zeros(STATE_DIM, device=device)
    horizon_squared_error = torch.zeros(forecast, STATE_DIM, device=device)
    contact_correct = torch.zeros(8, device=device)
    contact_positive = torch.zeros(8, device=device)
    contact_pred_positive = torch.zeros(8, device=device)
    contact_true_positive = torch.zeros(8, device=device)
    epistemic = []
    aleatoric = []
    total = 0

    for start in range(0, state.shape[0], batch_size):
        state_batch = state[start : start + batch_size]
        action_batch = action[start : start + batch_size]
        contact_batch = contact[start : start + batch_size]

        model.reset()
        pred, ale, epi, _, contact_logits, _ = model(state_batch[:, :horizon], action_batch[:, 1 : horizon + 1])
        target = state_batch[:, horizon]
        state_squared_error += torch.square(pred - target).sum(dim=0)
        total += pred.shape[0]
        aleatoric.append(ale.detach())
        epistemic.append(epi.detach())

        if contact_logits is not None:
            contact_pred = (torch.sigmoid(contact_logits) >= 0.5).float()
            contact_target = contact_batch[:, horizon]
            contact_correct += (contact_pred == contact_target).float().sum(dim=0)
            contact_positive += contact_target.sum(dim=0)
            contact_pred_positive += contact_pred.sum(dim=0)
            contact_true_positive += (contact_pred * contact_target).sum(dim=0)

        model.reset()
        rollout_state = state_batch[:, :horizon]
        rollout_action = action_batch[:, 1 : horizon + 1]
        for step in range(forecast):
            pred, _, _, _, _, _ = model(rollout_state, rollout_action)
            horizon_squared_error[step] += torch.square(pred - state_batch[:, horizon + step]).sum(dim=0)
            rollout_state = pred.unsqueeze(1)
            if step + 1 < forecast:
                rollout_action = action_batch[:, horizon + step + 1 : horizon + step + 2]

    norm_rmse = torch.sqrt(state_squared_error / total)
    physical_rmse = norm_rmse * state_std
    horizon_norm_rmse = torch.sqrt(horizon_squared_error.sum(dim=1) / (total * STATE_DIM))
    contact_precision = contact_true_positive / contact_pred_positive.clamp_min(1.0)
    contact_recall = contact_true_positive / contact_positive.clamp_min(1.0)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iter": checkpoint_iter,
        "data": str(data_path),
        "num_windows": total,
        "one_step_norm_rmse_all": float(torch.sqrt(state_squared_error.sum() / (total * STATE_DIM)).item()),
        "autoregressive_norm_rmse_by_horizon": [float(v.item()) for v in horizon_norm_rmse],
        "aleatoric_mean": float(torch.cat(aleatoric).mean().item()),
        "epistemic_mean": float(torch.cat(epistemic).mean().item()),
        "state_groups": {},
        "contact": {
            "accuracy": [float(v.item()) for v in contact_correct / total],
            "precision": [float(v.item()) for v in contact_precision],
            "recall": [float(v.item()) for v in contact_recall],
            "target_positive_rate": [float(v.item()) for v in contact_positive / total],
            "pred_positive_rate": [float(v.item()) for v in contact_pred_positive / total],
        },
        "termination_positive_rate": float(termination[:, horizon].mean().item()),
    }

    for name, group_slice in STATE_GROUPS.items():
        group_norm = torch.sqrt(torch.square(norm_rmse[group_slice]).mean())
        group_physical = torch.sqrt(torch.square(physical_rmse[group_slice]).mean())
        metrics["state_groups"][name] = {
            "norm_rmse": float(group_norm.item()),
            "physical_rmse": float(group_physical.item()),
        }
    return metrics


def print_metrics(task: str, metrics: dict):
    print(f"=== {task} ===")
    print(f"checkpoint={metrics['checkpoint']} iter={metrics['checkpoint_iter']}")
    print(f"data={metrics['data']} windows={metrics['num_windows']}")
    print(f"one_step_norm_rmse_all={metrics['one_step_norm_rmse_all']:.5f}")
    horizons = ", ".join(f"{v:.5f}" for v in metrics["autoregressive_norm_rmse_by_horizon"])
    print(f"autoregressive_norm_rmse_by_horizon=[{horizons}]")
    print(f"aleatoric_mean={metrics['aleatoric_mean']:.5f} epistemic_mean={metrics['epistemic_mean']:.5f}")
    for name, values in metrics["state_groups"].items():
        print(f"{name:14s} norm_rmse={values['norm_rmse']:.5f} physical_rmse={values['physical_rmse']:.5f}")
    print("contact_accuracy=" + ", ".join(f"{v:.4f}" for v in metrics["contact"]["accuracy"]))
    print("contact_recall=" + ", ".join(f"{v:.4f}" for v in metrics["contact"]["recall"]))
    print(f"termination_positive_rate={metrics['termination_positive_rate']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline world-model prediction error on a CSV dataset.")
    parser.add_argument("--task", default="lite3_flat", help="Offline task config to use.")
    parser.add_argument("--data", type=Path, default=None, help="CSV dataset path. Defaults to config dataset.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="World-model checkpoint. Defaults to config resume_path.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_windows", type=int, default=2048, help="0 means all possible windows.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rsl_rl_root", type=str, default=None, help="Path to the rsl_rl package root if not importable.")
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    config = resolve_config(args.task)
    data_path = args.data or default_data_path(config)
    checkpoint_path = args.checkpoint or default_checkpoint_path(config)
    metrics = evaluate(
        config,
        data_path,
        checkpoint_path,
        args.device,
        args.batch_size,
        args.num_windows,
        args.seed,
        args.rsl_rl_root,
    )
    print_metrics(args.task, metrics)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
