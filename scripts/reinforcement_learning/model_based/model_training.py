from __future__ import annotations

import os

import matplotlib
import torch
from rsl_rl.modules import plotter
from torch.utils.data import DataLoader, random_split
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ModelTraining:
    def __init__(
        self,
        log_dir,
        history_horizon,
        forecast_horizon,
        dataset,
        system_dynamics,
        device,
        optimizer,
        batch_size=1024,
        eval_traj_config=None,
        eval_traj_noise_scale=None,
        system_dynamics_loss_weights=None,
        save_interval=200,
        max_iterations=1000,
        random_batch_updates=True,
        max_eval_batches=64,
    ):
        if len(dataset) < 2:
            raise ValueError("Model dataset must contain at least two valid windows for train/eval split.")

        self.log_dir = log_dir
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        train_size = max(1, int(0.8 * len(dataset)))
        test_size = len(dataset) - train_size
        if test_size == 0:
            train_size -= 1
            test_size = 1
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.system_dynamics = system_dynamics
        self.device = device
        self.optimizer = optimizer
        self.eval_traj_config = eval_traj_config
        self.plotter = plotter.Plotter()
        self.eval_traj_noise_scale = eval_traj_noise_scale or [0.1, 0.2, 0.4, 0.5, 0.8]
        self.system_dynamics_loss_weights = system_dynamics_loss_weights or {
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        }
        self.save_interval = save_interval
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.random_batch_updates = random_batch_updates
        self.max_eval_batches = max_eval_batches
        if self.random_batch_updates and not hasattr(dataset, "sample_model_batch"):
            print("[WARN] Dataset has no sample_model_batch(); falling back to full dataloader epochs.")
            self.random_batch_updates = False
        if self.random_batch_updates:
            self.train_indices = torch.as_tensor(self.train_dataset.indices, dtype=torch.long, device=dataset.state_data.device)
        else:
            self.train_indices = None
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.current_learning_iteration = 0

    @staticmethod
    def _optional_batch(tensor: torch.Tensor, device: str):
        return tensor.to(device) if tensor.shape[-1] > 0 else None

    def _weighted_loss(self, state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss):
        weights = self.system_dynamics_loss_weights
        return (
            weights["state"] * state_loss
            + weights["sequence"] * sequence_loss
            + weights["bound"] * bound_loss
            + weights["kl"] * kl_loss
            + weights["extension"] * extension_loss
            + weights["contact"] * contact_loss
            + weights["termination"] * termination_loss
        )

    def train(self):
        self.system_dynamics.train()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.max_iterations
        for it in range(start_iter, tot_iter):
            self.log_dict = {}
            mean_losses = {
                "state": 0.0,
                "sequence": 0.0,
                "bound": 0.0,
                "kl": 0.0,
                "extension": 0.0,
                "contact": 0.0,
                "termination": 0.0,
            }
            last_loss = None
            num_batches = 1 if self.random_batch_updates else len(self.train_dataloader)
            if self.random_batch_updates:
                batch_iter = [self.train_dataset.dataset.sample_model_batch(self.batch_size, self.train_indices)]
            else:
                batch_iter = self.train_dataloader
            for state, action, extension, contact, termination in batch_iter:
                self.system_dynamics.reset()
                state = state.to(self.device)
                action = action.to(self.device)
                extension = self._optional_batch(extension, self.device)
                contact = self._optional_batch(contact, self.device)
                termination = self._optional_batch(termination, self.device)

                self.optimizer.zero_grad()
                losses = self.system_dynamics.compute_loss(state, action, extension, contact, termination)
                loss = self._weighted_loss(*losses)
                loss.backward()
                self.optimizer.step()

                last_loss = loss
                for key, value in zip(mean_losses.keys(), losses, strict=True):
                    mean_losses[key] += value.item()

            for key in mean_losses:
                mean_losses[key] /= num_batches
            print(f"[Model Training] Iteration {it}/{tot_iter} | Loss: {last_loss.item() if last_loss is not None else 0.0}")
            self.current_learning_iteration = it
            self.log_dict.update({f"Model/train_{key}_loss": value for key, value in mean_losses.items()})
            if it % self.save_interval == 0:
                self.evaluate()
                self.save(it)
            if wandb.run is not None:
                wandb.log(self.log_dict)

        self.evaluate()
        self.save(self.current_learning_iteration)

    def _autoregressive_prediction(self, state_traj, action_traj, extension_traj, contact_traj, termination_traj):
        state_traj_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        epistemic_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        action_traj_pred = action_traj.clone()
        extension_traj_pred = torch.zeros_like(extension_traj, device=self.device)
        contact_traj_pred = torch.zeros_like(contact_traj, device=self.device)
        termination_traj_pred = torch.zeros_like(termination_traj, device=self.device)

        state_traj_pred[:, : self.start_step] = state_traj[:, : self.start_step]
        extension_traj_pred[:, : self.start_step] = extension_traj[:, : self.start_step]
        contact_traj_pred[:, : self.start_step] = contact_traj[:, : self.start_step]
        termination_traj_pred[:, : self.start_step] = termination_traj[:, : self.start_step]

        self.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(self.start_step, self.eval_traj_config["len_traj"]):
                if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"] and i > self.start_step:
                    state_input = state_traj_pred[:, i - 1 : i]
                    action_input = action_traj_pred[:, i - 1 : i]
                else:
                    state_input = state_traj_pred[:, i - self.start_step : i]
                    action_input = action_traj_pred[:, i - self.start_step : i]
                (
                    state_pred,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                    extension_pred,
                    contact_pred,
                    termination_pred,
                ) = self.system_dynamics.forward(state_input, action_input)
                state_traj_pred[:, i] = state_pred
                aleatoric_uncertainty_traj_pred[:, i] = aleatoric_uncertainty
                epistemic_uncertainty_traj_pred[:, i] = epistemic_uncertainty
                if extension_pred is not None:
                    extension_traj_pred[:, i] = extension_pred
                if contact_pred is not None:
                    contact_traj_pred[:, i] = torch.sigmoid(contact_pred).round().int()
                if termination_pred is not None:
                    termination_traj_pred[:, i] = torch.sigmoid(termination_pred).round().int()
        return (
            state_traj_pred,
            aleatoric_uncertainty_traj_pred,
            epistemic_uncertainty_traj_pred,
            action_traj_pred,
            extension_traj_pred,
            contact_traj_pred,
            termination_traj_pred,
        )

    def evaluate(self):
        self.system_dynamics.eval()
        mean_losses = {
            "state": 0.0,
            "sequence": 0.0,
            "bound": 0.0,
            "kl": 0.0,
            "extension": 0.0,
            "contact": 0.0,
            "termination": 0.0,
        }
        num_batches = 0
        with torch.inference_mode():
            for batch_idx, (state, action, extension, contact, termination) in enumerate(self.test_dataloader):
                if self.max_eval_batches > 0 and batch_idx >= self.max_eval_batches:
                    break
                self.system_dynamics.reset()
                state = state.to(self.device)
                action = action.to(self.device)
                extension = self._optional_batch(extension, self.device)
                contact = self._optional_batch(contact, self.device)
                termination = self._optional_batch(termination, self.device)
                losses = self.system_dynamics.compute_loss(state, action, extension, contact, termination)
                for key, value in zip(mean_losses.keys(), losses, strict=True):
                    mean_losses[key] += value.item()
                num_batches += 1

        for key in mean_losses:
            mean_losses[key] /= max(num_batches, 1)
        print(f"[Model Evaluation] Eval state loss: {mean_losses['state']}")
        self.log_dict.update({f"Model/eval_{key}_loss": value for key, value in mean_losses.items()})

        if self.eval_traj_config is not None:
            self._evaluate_trajectories()
        self.system_dynamics.train()

    def _evaluate_trajectories(self):
        fig, ax = plt.subplots(
            len(self.eval_traj_config["state_idx_dict"]) + 4,
            self.eval_traj_config["num_visualizations"],
            figsize=(10 * self.eval_traj_config["num_visualizations"], 10),
        )
        state_traj, action_traj, extension_traj, contact_traj, termination_traj = self.eval_traj_config["traj_data"]
        self.plotter.plot_trajectories(
            ax,
            None,
            state_traj[: self.eval_traj_config["num_visualizations"]],
            action_traj[: self.eval_traj_config["num_visualizations"]],
            extension_traj[: self.eval_traj_config["num_visualizations"]],
            contact_traj[: self.eval_traj_config["num_visualizations"]],
            termination_traj[: self.eval_traj_config["num_visualizations"]],
            self.eval_traj_config["state_idx_dict"],
        )

        state_traj, action_traj = self.train_dataset.dataset.normalize(state_traj, action_traj)
        self.start_step = self.history_horizon
        (
            state_traj_pred,
            _,
            _,
            action_traj_pred,
            extension_traj_pred,
            contact_traj_pred,
            termination_traj_pred,
        ) = self._autoregressive_prediction(state_traj, action_traj, extension_traj, contact_traj, termination_traj)
        denom = state_traj[:, self.start_step :].abs().sum(dim=-1).clamp_min(1.0e-6)
        traj_autoregressive_error = (
            (state_traj_pred[:, self.start_step :] - state_traj[:, self.start_step :]).abs().sum(dim=-1) / denom
        ).mean().item()
        print(f"[Model Evaluation] Autoregressive Error: {traj_autoregressive_error}")

        state_traj_pred, action_traj_pred = self.train_dataset.dataset.denormalize(state_traj_pred, action_traj_pred)
        self.plotter.plot_trajectories(
            ax,
            self.start_step,
            state_traj_pred[: self.eval_traj_config["num_visualizations"]],
            action_traj_pred[: self.eval_traj_config["num_visualizations"]],
            extension_traj_pred[: self.eval_traj_config["num_visualizations"]],
            contact_traj_pred[: self.eval_traj_config["num_visualizations"]],
            termination_traj_pred[: self.eval_traj_config["num_visualizations"]],
            self.eval_traj_config["state_idx_dict"],
            prediction=True,
        )
        self.log_dict["Model/traj_autoregressive_error"] = traj_autoregressive_error
        if wandb.run is not None:
            self.log_dict["Model/visualize_trajs"] = wandb.Image(fig)

        for noise_scale in self.eval_traj_noise_scale:
            state_traj_noised = state_traj + torch.randn_like(state_traj) * noise_scale
            action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale
            state_traj_pred_noised, _, _, _, _, _, _ = self._autoregressive_prediction(
                state_traj_noised,
                action_traj_noised,
                extension_traj,
                contact_traj,
                termination_traj,
            )
            denom = state_traj_noised[:, self.start_step :].abs().sum(dim=-1).clamp_min(1.0e-6)
            traj_autoregressive_error_noised = (
                (state_traj_pred_noised[:, self.start_step :] - state_traj_noised[:, self.start_step :]).abs().sum(dim=-1)
                / denom
            ).mean().item()
            print(f"[Model Evaluation] Noised Autoregressive Error with Level {noise_scale}: {traj_autoregressive_error_noised}")
            self.log_dict[f"Model/traj_autoregressive_error_noised_{noise_scale}"] = traj_autoregressive_error_noised
        plt.close(fig)

    def save(self, it):
        os.makedirs(self.log_dir, exist_ok=True)
        dataset = self.train_dataset.dataset
        torch.save(
            {
                "system_dynamics_state_dict": self.system_dynamics.state_dict(),
                "iter": it,
                "history_horizon": self.history_horizon,
                "forecast_horizon": self.forecast_horizon,
                "state_data_mean": dataset.state_data_mean.detach().cpu(),
                "state_data_std": dataset.state_data_std.detach().cpu(),
                "action_data_mean": dataset.action_data_mean.detach().cpu(),
                "action_data_std": dataset.action_data_std.detach().cpu(),
            },
            os.path.join(self.log_dir, f"model_{it}.pt"),
        )
