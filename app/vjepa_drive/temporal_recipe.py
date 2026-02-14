# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TemporalRecipe:
    """Shared temporal recipe used by train + data preparation."""

    frames_per_clip: int
    clip_fps: float
    history_horizon: int
    history_dt: float
    future_horizon: int
    future_dt: float

    @property
    def clip_duration_sec(self) -> float:
        return float(self.frames_per_clip) / max(float(self.clip_fps), 1e-6)

    @property
    def history_margin_sec(self) -> float:
        return float(self.history_horizon) * float(self.history_dt)

    @property
    def future_margin_sec(self) -> float:
        return float(self.future_horizon) * float(self.future_dt)

    @property
    def required_window_sec(self) -> float:
        return self.history_margin_sec + self.clip_duration_sec + self.future_margin_sec

    @classmethod
    def from_data_cfg(cls, data_cfg: dict) -> "TemporalRecipe":
        dataset_fpcs = data_cfg.get("dataset_fpcs")
        if not dataset_fpcs:
            raise ValueError("data.dataset_fpcs must be provided to derive temporal recipe")
        return cls(
            frames_per_clip=int(max(dataset_fpcs)),
            clip_fps=float(data_cfg.get("fps", 4.0)),
            history_horizon=int(data_cfg.get("trajectory_history_horizon", 8)),
            history_dt=float(data_cfg.get("trajectory_history_dt", 0.5)),
            future_horizon=int(data_cfg.get("trajectory_horizon", 12)),
            future_dt=float(data_cfg.get("trajectory_dt", 0.5)),
        )

    @classmethod
    def from_env(cls) -> "TemporalRecipe":
        return cls(
            frames_per_clip=int(os.environ.get("VJEPA_FRAMES_PER_CLIP", "16")),
            clip_fps=float(os.environ.get("VJEPA_CLIP_FPS", "4.0")),
            history_horizon=int(os.environ.get("VJEPA_TRAJ_HISTORY_HORIZON", "8")),
            history_dt=float(os.environ.get("VJEPA_TRAJ_HISTORY_DT", "0.5")),
            future_horizon=int(os.environ.get("VJEPA_TRAJ_FUTURE_HORIZON", "12")),
            future_dt=float(os.environ.get("VJEPA_TRAJ_FUTURE_DT", "0.5")),
        )
