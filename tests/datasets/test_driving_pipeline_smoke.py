# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from app.vjepa.utils import init_video_model
from app.vjepa_drive.losses import TrajectoryLossWeights
from app.vjepa_drive.train_loss import (
    compute_jepa_loss,
    compute_trajectory_stats,
    forward_encoder_full,
    forward_predictor_from_full,
    forward_target_encoder,
)
from app.vjepa_drive.utils import init_trajectory_head
from src.datasets.driving_dataset import DrivingMaskCollator, DrivingVideoDataset
from src.masks.multiseq_multiblock3d import MaskCollator


class _DummyEgoLoader:
    def load(self, ego_path):
        timestamps = np.linspace(0.0, 4.0, 41, dtype=np.float64)
        positions = np.stack([timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)], axis=1)
        headings = np.zeros_like(timestamps)
        return timestamps, positions, headings


class TestDrivingPipelineSmoke(unittest.TestCase):
    def test_dataset_collator_single_train_step_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "train.csv"
            manifest.write_text(f"{tmpdir}/video.mp4 {tmpdir}/ego.csv 0.0 1.0 nvidia_av\n", encoding="utf-8")

            dataset = DrivingVideoDataset(
                data_paths=[str(manifest)],
                frames_per_clip=2,
                dataset_fpcs=[2],
                fps=2,
                num_clips=1,
                random_clip_sampling=False,
                transform=lambda clip: torch.from_numpy(clip).permute(3, 0, 1, 2).float(),
                trajectory_horizon=2,
                trajectory_dt=0.5,
                trajectory_history_horizon=2,
                trajectory_history_dt=0.5,
                dataset_names=["nvidia_av"],
            )

            def fake_loadvideo(fname, fpc, sample=None):
                del fname, sample
                h, w = 64, 64
                frame0 = np.zeros((h, w, 3), dtype=np.uint8)
                frame1 = np.ones((h, w, 3), dtype=np.uint8)
                buffer = np.stack([frame0, frame1], axis=0)
                clip_indices = [np.array([0, 1], dtype=np.int64)]
                sampled_clip_end = 1.0
                return buffer, clip_indices, sampled_clip_end

            dataset._loadvideo_decord = fake_loadvideo
            dataset._get_ego_loader = lambda dataset_name: _DummyEgoLoader()

            sample = dataset[0]
            self.assertIsNotNone(sample)

            mask_cfg = [
                {
                    "spatial_scale": [0.4, 0.4],
                    "temporal_scale": [1.0, 1.0],
                    "aspect_ratio": [0.75, 1.5],
                    "num_blocks": 2,
                }
            ]
            mask_collator = DrivingMaskCollator(
                MaskCollator(
                    cfgs_mask=mask_cfg,
                    dataset_fpcs=[2],
                    crop_size=64,
                    patch_size=16,
                    tubelet_size=2,
                )
            )
            collated = mask_collator([sample])
            self.assertEqual(len(collated), 1)

            device = torch.device("cpu")
            encoder, predictor = init_video_model(
                device=device,
                patch_size=16,
                max_num_frames=2,
                tubelet_size=2,
                model_name="vit_micro",
                crop_size=64,
                pred_depth=2,
                pred_num_heads=3,
                pred_embed_dim=96,
                use_mask_tokens=True,
                num_mask_tokens=1,
                use_activation_checkpointing=False,
            )
            target_encoder = copy.deepcopy(encoder)
            traj_head = init_trajectory_head(
                embed_dim=encoder.backbone.embed_dim,
                num_waypoints=2,
                waypoint_dim=2,
                num_modes=2,
                history_tokens=2,
                history_hidden_dim=64,
                num_heads=2,
                mlp_ratio=2.0,
                pooler_depth=1,
                device=device,
                use_activation_checkpointing=False,
            )

            batch = collated[0].to_device(device)
            clips = [batch.video()]
            masks_enc = [batch.masks_enc]
            masks_pred = [batch.masks_pred]
            gt = [batch.gt_trajectories]
            hist = [batch.ego_histories]

            h = forward_target_encoder(target_encoder, clips)
            z_full = forward_encoder_full(encoder, clips)
            z_pred = forward_predictor_from_full(predictor, z_full, masks_enc, masks_pred)
            loss_jepa = compute_jepa_loss(z_pred, h, masks_pred, loss_exp=1.0)
            traj_stats = compute_trajectory_stats(
                z_full=z_full,
                gt_trajectories=gt,
                ego_histories=hist,
                predict_fn=lambda features, ego_history: traj_head.predict_distribution(features, ego_history=ego_history),
                trajectory_dt=0.5,
                weights=TrajectoryLossWeights(),
            )
            loss = loss_jepa + traj_stats["loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            opt = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()) + list(traj_head.parameters()), lr=1e-4)
            opt.step()


if __name__ == "__main__":
    unittest.main()
