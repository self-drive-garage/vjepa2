# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryDistribution:
    """Typed multimodal trajectory distribution."""

    means: torch.Tensor         # [B, K, T, 2]
    log_stds: torch.Tensor      # [B, K, T, 2]
    mode_logits: torch.Tensor   # [B, K]

    @property
    def mode_probs(self) -> torch.Tensor:
        return torch.softmax(self.mode_logits, dim=-1)

    @property
    def num_modes(self) -> int:
        return int(self.mode_logits.size(-1))

    @staticmethod
    def _gather_modes(mode_tensor: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(mode_tensor.size(0), device=mode_tensor.device)
        return mode_tensor[batch_idx, mode_idx]

    def best_mode_idx(self) -> torch.Tensor:
        return torch.argmax(self.mode_probs, dim=-1)

    def best_means(self) -> torch.Tensor:
        return self._gather_modes(self.means, self.best_mode_idx())

    def best_log_stds(self) -> torch.Tensor:
        return self._gather_modes(self.log_stds, self.best_mode_idx())


@dataclass
class DrivingSample:
    """Structured dataset sample returned by DrivingVideoDataset."""

    buffer: list[torch.Tensor]
    gt_trajectory: torch.Tensor
    ego_history: torch.Tensor
    clip_indices: list[torch.Tensor]


@dataclass
class DrivingBatch:
    """Structured collated batch used by vjepa_drive training."""

    videos: list[torch.Tensor]
    labels: torch.Tensor
    clip_indices: list[torch.Tensor]
    masks_enc: list[torch.Tensor]
    masks_pred: list[torch.Tensor]
    gt_trajectories: torch.Tensor
    ego_histories: torch.Tensor

    def batch_size_and_fpc(self) -> tuple[int, int]:
        if not self.clip_indices:
            raise ValueError("clip_indices is empty; cannot infer frames-per-clip.")
        batch_size, fpc = self.clip_indices[-1].size()
        return int(batch_size), int(fpc)

    def video(self, clip_idx: int = 0) -> torch.Tensor:
        return self.videos[clip_idx]

    def to_device(self, device: torch.device) -> "DrivingBatch":
        return DrivingBatch(
            videos=[v.to(device, non_blocking=True) for v in self.videos],
            labels=self.labels.to(device, non_blocking=True),
            clip_indices=self.clip_indices,
            masks_enc=[m.to(device, non_blocking=True) for m in self.masks_enc],
            masks_pred=[m.to(device, non_blocking=True) for m in self.masks_pred],
            gt_trajectories=self.gt_trajectories.to(device, non_blocking=True),
            ego_histories=self.ego_histories.to(device, non_blocking=True),
        )
