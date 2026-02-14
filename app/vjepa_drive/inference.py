# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from app.vjepa_drive.runtime_safety import (
    SafetyEnvelopeConfig,
    SafetyEnvelopeOutput,
    select_trajectory_with_safety_envelope,
)
from app.vjepa_drive.types import TrajectoryDistribution


class DrivingInferenceModel(torch.nn.Module):
    """Inference wrapper with optional runtime safety envelope."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        trajectory_head: torch.nn.Module,
        trajectory_dt: float,
        safety_cfg: SafetyEnvelopeConfig | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.trajectory_head = trajectory_head
        self.trajectory_dt = float(trajectory_dt)
        self.safety_cfg = safety_cfg
        self._input_dtype = next(encoder.parameters()).dtype

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            video = video.unsqueeze(2)
        video = video.to(dtype=self._input_dtype)
        features = self.encoder(video)
        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise RuntimeError(f"Expected a single sequence output, got {len(features)}")
            features = features[0]
        return features

    def predict_distribution(self, video: torch.Tensor, ego_history: torch.Tensor | None = None) -> TrajectoryDistribution:
        features = self.encode_video(video)
        return self.trajectory_head.predict_distribution(features, ego_history=ego_history)

    def predict(
        self,
        video: torch.Tensor,
        ego_history: torch.Tensor | None = None,
        apply_safety: bool = True,
    ) -> tuple[torch.Tensor, TrajectoryDistribution, SafetyEnvelopeOutput | None]:
        dist = self.predict_distribution(video, ego_history=ego_history)
        if apply_safety and self.safety_cfg is not None:
            safety = select_trajectory_with_safety_envelope(
                pred_dist=dist,
                ego_history=ego_history,
                dt=self.trajectory_dt,
                config=self.safety_cfg,
            )
            return safety.trajectory, dist, safety
        return dist.best_means(), dist, None

    def forward(self, video: torch.Tensor, ego_history: torch.Tensor | None = None) -> torch.Tensor:
        traj, _, _ = self.predict(video, ego_history=ego_history, apply_safety=True)
        return traj
