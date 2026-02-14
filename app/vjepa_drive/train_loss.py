# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F

from app.vjepa_drive.losses import TrajectoryLossWeights, compute_multimodal_trajectory_losses
from app.vjepa_drive.types import TrajectoryDistribution
from src.masks.utils import apply_masks


def compute_jepa_loss(
    z_pred: list[list[torch.Tensor]],
    h: list[torch.Tensor],
    masks_pred: list[list[torch.Tensor]],
    loss_exp: float,
) -> torch.Tensor:
    """Compute JEPA loss for post-hoc masked predictions."""
    h_masked = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    loss, count = 0.0, 0
    for zi, hi in zip(z_pred, h_masked):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            count += 1
    return loss / max(count, 1)


def compute_trajectory_stats(
    z_full: list[torch.Tensor],
    gt_trajectories: list[torch.Tensor],
    ego_histories: list[torch.Tensor],
    predict_fn,
    trajectory_dt: float,
    weights: TrajectoryLossWeights,
) -> dict[str, torch.Tensor]:
    """Compute averaged trajectory losses across clip groups."""
    traj_stats = []
    for zi, gt_i, hist_i in zip(z_full, gt_trajectories, ego_histories):
        pred_dist: TrajectoryDistribution = predict_fn(zi, ego_history=hist_i)
        stats = compute_multimodal_trajectory_losses(
            pred_dist=pred_dist,
            gt_trajectory=gt_i,
            dt=trajectory_dt,
            weights=weights,
        )
        traj_stats.append(stats)

    out = {}
    keys = traj_stats[0].keys()
    for key in keys:
        out[key] = sum(stat[key] for stat in traj_stats) / len(traj_stats)
    return out


def forward_target_encoder(target_encoder, clips: list[torch.Tensor]) -> list[torch.Tensor]:
    with torch.no_grad():
        h = target_encoder(clips)
        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
    return h


def forward_encoder_full(encoder, clips: list[torch.Tensor]) -> list[torch.Tensor]:
    return encoder(clips)


def forward_predictor_from_full(
    predictor,
    z_full: list[torch.Tensor],
    masks_enc: list[list[torch.Tensor]],
    masks_pred: list[list[torch.Tensor]],
) -> list[list[torch.Tensor]]:
    z_masked = []
    for zi, mi in zip(z_full, masks_enc):
        z_masked_i = []
        for mij in mi:
            z_masked_i.append(apply_masks(zi, mij))
        z_masked.append(z_masked_i)
    return predictor(z_masked, masks_enc, masks_pred)
