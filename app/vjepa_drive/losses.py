# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from app.vjepa_drive.types import TrajectoryDistribution


@dataclass(frozen=True)
class TrajectoryLossWeights:
    ade_weight: float = 1.0
    fde_weight: float = 1.0
    nll_weight: float = 0.5
    calib_weight: float = 0.1
    smoothness_weight: float = 0.05
    feasibility_weight: float = 0.05
    speed_limit_mps: float = 25.0
    accel_limit_mps2: float = 6.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "TrajectoryLossWeights":
        return cls(
            ade_weight=float(cfg.get("loss_ade_weight", 1.0)),
            fde_weight=float(cfg.get("loss_fde_weight", 1.0)),
            nll_weight=float(cfg.get("loss_nll_weight", 0.5)),
            calib_weight=float(cfg.get("loss_calib_weight", 0.1)),
            smoothness_weight=float(cfg.get("loss_smoothness_weight", 0.05)),
            feasibility_weight=float(cfg.get("loss_feasibility_weight", 0.05)),
            speed_limit_mps=float(cfg.get("feasibility_speed_limit_mps", 25.0)),
            accel_limit_mps2=float(cfg.get("feasibility_accel_limit_mps2", 6.0)),
        )


def schedule_scalar(start, end, progress, schedule="cosine"):
    """Interpolate a scalar value over normalized progress [0, 1]."""
    progress = float(max(0.0, min(1.0, progress)))
    if schedule == "constant":
        mix = 0.0
    elif schedule == "linear":
        mix = progress
    elif schedule == "cosine":
        mix = 0.5 - 0.5 * math.cos(math.pi * progress)
    else:
        raise ValueError(f"Unsupported schedule '{schedule}'")
    return float(start + (end - start) * mix)


def _gather_modes(mode_tensor, mode_idx):
    # mode_tensor: [B, K, ...], mode_idx: [B]
    batch_idx = torch.arange(mode_tensor.size(0), device=mode_tensor.device)
    return mode_tensor[batch_idx, mode_idx]


def _multimodal_likelihood_nll(means, log_stds, mode_logits, gt_trajectory):
    # means/log_stds: [B, K, T, 2], mode_logits: [B, K], gt: [B, T, 2]
    log_stds = log_stds.clamp(-8.0, 4.0)
    diffs = means - gt_trajectory.unsqueeze(1)
    inv_var = torch.exp(-2.0 * log_stds)

    # Independent Gaussian per coordinate/waypoint.
    point_log_prob = -0.5 * (
        diffs.pow(2) * inv_var + 2.0 * log_stds + math.log(2.0 * math.pi)
    )
    mode_log_prob = point_log_prob.sum(dim=(-1, -2))  # [B, K]
    log_pi = F.log_softmax(mode_logits, dim=-1)
    mix_log_prob = torch.logsumexp(log_pi + mode_log_prob, dim=-1)  # [B]
    return -mix_log_prob.mean()


def compute_multimodal_trajectory_losses(
    pred_dist: TrajectoryDistribution,
    gt_trajectory,
    dt,
    weights: TrajectoryLossWeights | None = None,
):
    """Compute multimodal trajectory losses and diagnostics."""
    weights = weights or TrajectoryLossWeights()
    means = pred_dist.means            # [B, K, T, 2]
    log_stds = pred_dist.log_stds      # [B, K, T, 2]
    mode_logits = pred_dist.mode_logits  # [B, K]
    mode_probs = pred_dist.mode_probs  # [B, K]

    diffs = means - gt_trajectory.unsqueeze(1)
    l2 = torch.norm(diffs, dim=-1)  # [B, K, T]
    ade_per_mode = l2.mean(dim=-1)  # [B, K]
    fde_per_mode = l2[:, :, -1]     # [B, K]

    best_mode_idx = torch.argmin(ade_per_mode, dim=-1)  # [B]
    min_ade = _gather_modes(ade_per_mode, best_mode_idx).mean()
    min_fde = _gather_modes(fde_per_mode, best_mode_idx).mean()

    nll = _multimodal_likelihood_nll(
        means=means,
        log_stds=log_stds,
        mode_logits=mode_logits,
        gt_trajectory=gt_trajectory,
    )

    best_means = _gather_modes(means, best_mode_idx)         # [B, T, 2]
    best_log_stds = _gather_modes(log_stds, best_mode_idx)   # [B, T, 2]
    pred_var = torch.exp(2.0 * best_log_stds)
    sq_err = (best_means - gt_trajectory).pow(2)
    # Calibrate predicted uncertainty against observed squared error.
    calib = F.l1_loss(pred_var, sq_err.detach())

    dt = float(max(dt, 1e-3))
    vel = (best_means[:, 1:] - best_means[:, :-1]) / dt
    speed = torch.norm(vel, dim=-1)
    if vel.size(1) > 1:
        acc = (vel[:, 1:] - vel[:, :-1]) / dt
    else:
        acc = vel.new_zeros((vel.size(0), 0, vel.size(2)))

    if acc.size(1) > 1:
        jerk = (acc[:, 1:] - acc[:, :-1]) / dt
        smoothness = torch.norm(jerk, dim=-1).mean()
    else:
        smoothness = vel.new_zeros(())

    if acc.numel() > 0:
        accel = torch.norm(acc, dim=-1)
        accel_penalty = F.relu(accel - weights.accel_limit_mps2).mean()
    else:
        accel_penalty = vel.new_zeros(())
    speed_penalty = F.relu(speed - weights.speed_limit_mps).mean()
    feasibility = speed_penalty + accel_penalty

    loss_traj = (
        weights.ade_weight * min_ade
        + weights.fde_weight * min_fde
        + weights.nll_weight * nll
        + weights.calib_weight * calib
        + weights.smoothness_weight * smoothness
        + weights.feasibility_weight * feasibility
    )

    top_prob = mode_probs.max(dim=-1).values.mean()
    mode_entropy = -(mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=-1).mean()
    avg_std = torch.exp(log_stds).mean()

    return {
        "loss": loss_traj,
        "min_ade": min_ade,
        "min_fde": min_fde,
        "nll": nll,
        "calib": calib,
        "smoothness": smoothness,
        "feasibility": feasibility,
        "top_prob": top_prob,
        "mode_entropy": mode_entropy,
        "avg_std": avg_std,
    }
