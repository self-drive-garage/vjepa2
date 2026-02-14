# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from app.vjepa_drive.types import TrajectoryDistribution


@dataclass
class SafetyEnvelopeConfig:
    min_confidence: float = 0.45
    max_avg_std: float = 2.5
    max_mode_entropy: float = 1.2
    speed_limit_mps: float = 25.0
    accel_limit_mps2: float = 6.0
    fallback_mode: str = "constant_velocity"  # constant_velocity | stop

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "SafetyEnvelopeConfig":
        cfg = cfg or {}
        return cls(
            min_confidence=float(cfg.get("min_confidence", 0.45)),
            max_avg_std=float(cfg.get("max_avg_std", 2.5)),
            max_mode_entropy=float(cfg.get("max_mode_entropy", 1.2)),
            speed_limit_mps=float(cfg.get("speed_limit_mps", 25.0)),
            accel_limit_mps2=float(cfg.get("accel_limit_mps2", 6.0)),
            fallback_mode=str(cfg.get("fallback_mode", "constant_velocity")),
        )


@dataclass
class SafetyEnvelopeOutput:
    trajectory: torch.Tensor
    selected_trajectory: torch.Tensor
    selected_mode: torch.Tensor
    used_fallback: torch.Tensor
    confidence: torch.Tensor
    avg_std: torch.Tensor
    mode_entropy: torch.Tensor
    speed_max: torch.Tensor
    accel_max: torch.Tensor


def _gather_modes(mode_tensor, mode_idx):
    # mode_tensor: [B, K, ...], mode_idx: [B]
    batch_idx = torch.arange(mode_tensor.size(0), device=mode_tensor.device)
    return mode_tensor[batch_idx, mode_idx]


def _kinematics_limits(trajectory, dt):
    dt = float(max(dt, 1e-3))
    vel = (trajectory[:, 1:] - trajectory[:, :-1]) / dt
    speed = torch.norm(vel, dim=-1)
    speed_max = speed.max(dim=-1).values if speed.numel() > 0 else trajectory.new_zeros((trajectory.size(0),))
    if vel.size(1) > 1:
        acc = (vel[:, 1:] - vel[:, :-1]) / dt
        accel = torch.norm(acc, dim=-1)
        accel_max = accel.max(dim=-1).values if accel.numel() > 0 else trajectory.new_zeros((trajectory.size(0),))
    else:
        accel_max = trajectory.new_zeros((trajectory.size(0),))
    return speed_max, accel_max


def _constant_velocity_fallback(ego_history, horizon, dt, max_speed_mps):
    if ego_history is None or ego_history.ndim != 3 or ego_history.size(1) == 0:
        return None
    dt = float(max(dt, 1e-3))
    last = ego_history[:, -1, :]
    v_now = (0.0 - last) / dt
    if ego_history.size(1) > 1:
        prev = ego_history[:, -2, :]
        v_hist = (last - prev) / dt
        velocity = 0.5 * (v_now + v_hist)
    else:
        velocity = v_now
    speed = torch.norm(velocity, dim=-1, keepdim=True).clamp(min=1e-6)
    scale = torch.clamp(max_speed_mps / speed, max=1.0)
    velocity = velocity * scale
    steps = torch.arange(1, horizon + 1, device=ego_history.device, dtype=ego_history.dtype).view(1, horizon, 1)
    return velocity.unsqueeze(1) * steps * dt


def select_trajectory_with_safety_envelope(
    pred_dist: TrajectoryDistribution,
    ego_history,
    dt,
    config: SafetyEnvelopeConfig,
):
    """Select a trajectory and apply confidence/dynamics guardrails."""
    means = pred_dist.means          # [B, K, T, 2]
    log_stds = pred_dist.log_stds    # [B, K, T, 2]
    mode_probs = pred_dist.mode_probs  # [B, K]

    selected_mode = torch.argmax(mode_probs, dim=-1)
    selected_traj = _gather_modes(means, selected_mode)  # [B, T, 2]
    selected_std = torch.exp(_gather_modes(log_stds, selected_mode))

    confidence = mode_probs.max(dim=-1).values
    mode_entropy = -(mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=-1)
    avg_std = selected_std.mean(dim=(-1, -2))

    speed_max, accel_max = _kinematics_limits(selected_traj, dt=dt)
    low_confidence = confidence < config.min_confidence
    high_uncertainty = avg_std > config.max_avg_std
    high_entropy = mode_entropy > config.max_mode_entropy
    infeasible_dyn = (speed_max > config.speed_limit_mps) | (accel_max > config.accel_limit_mps2)
    needs_fallback = low_confidence | high_uncertainty | high_entropy | infeasible_dyn

    if config.fallback_mode == "stop":
        fallback_traj = torch.zeros_like(selected_traj)
    elif config.fallback_mode == "constant_velocity":
        fallback_traj = _constant_velocity_fallback(
            ego_history=ego_history,
            horizon=selected_traj.size(1),
            dt=dt,
            max_speed_mps=config.speed_limit_mps,
        )
        if fallback_traj is None:
            fallback_traj = torch.zeros_like(selected_traj)
    else:
        raise ValueError(f"Unsupported fallback mode '{config.fallback_mode}'")

    final_traj = selected_traj.clone()
    if needs_fallback.any():
        final_traj[needs_fallback] = fallback_traj[needs_fallback]

    return SafetyEnvelopeOutput(
        trajectory=final_traj,
        selected_trajectory=selected_traj,
        selected_mode=selected_mode,
        used_fallback=needs_fallback,
        confidence=confidence,
        avg_std=avg_std,
        mode_entropy=mode_entropy,
        speed_max=speed_max,
        accel_max=accel_max,
    )
