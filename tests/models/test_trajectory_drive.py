# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from app.vjepa_drive.losses import compute_multimodal_trajectory_losses
from app.vjepa_drive.runtime_safety import SafetyEnvelopeConfig, select_trajectory_with_safety_envelope
from src.models.trajectory_head import TrajectoryHead


class TestTrajectoryDrive(unittest.TestCase):
    def test_multimodal_head_shapes_and_losses(self):
        bsz, num_tokens, embed_dim = 2, 128, 64
        num_modes, num_waypoints, history_tokens = 3, 6, 4

        head = TrajectoryHead(
            embed_dim=embed_dim,
            num_waypoints=num_waypoints,
            waypoint_dim=2,
            num_modes=num_modes,
            history_tokens=history_tokens,
            history_hidden_dim=32,
            num_heads=4,
            pooler_depth=1,
        )
        features = torch.randn(bsz, num_tokens, embed_dim)
        ego_history = torch.randn(bsz, history_tokens, 2)

        pred = head.predict_distribution(features, ego_history=ego_history)
        self.assertEqual(pred.means.shape, (bsz, num_modes, num_waypoints, 2))
        self.assertEqual(pred.log_stds.shape, (bsz, num_modes, num_waypoints, 2))
        self.assertEqual(pred.mode_logits.shape, (bsz, num_modes))
        self.assertEqual(pred.mode_probs.shape, (bsz, num_modes))

        best = head(features, ego_history=ego_history)
        self.assertEqual(best.shape, (bsz, num_waypoints, 2))

        gt = torch.randn(bsz, num_waypoints, 2)
        losses = compute_multimodal_trajectory_losses(pred, gt, dt=0.5)
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_runtime_safety_fallback(self):
        bsz, num_modes, num_waypoints = 2, 2, 4
        history_tokens = 2
        head = TrajectoryHead(
            embed_dim=64,
            num_waypoints=num_waypoints,
            waypoint_dim=2,
            num_modes=num_modes,
            history_tokens=history_tokens,
            history_hidden_dim=32,
            num_heads=4,
            pooler_depth=1,
        )
        pred_dist = head.predict_distribution(
            torch.randn(bsz, 32, 64),
            ego_history=torch.randn(bsz, history_tokens, 2),
        )
        ego_history = torch.tensor(
            [
                [[-2.0, 0.0], [-1.0, 0.0]],
                [[-1.0, 0.0], [-0.5, 0.0]],
            ],
            dtype=torch.float32,
        )
        config = SafetyEnvelopeConfig(
            min_confidence=0.9,  # force fallback
            max_avg_std=10.0,
            max_mode_entropy=10.0,
            speed_limit_mps=20.0,
            accel_limit_mps2=20.0,
            fallback_mode="constant_velocity",
        )
        out = select_trajectory_with_safety_envelope(
            pred_dist=pred_dist,
            ego_history=ego_history,
            dt=0.5,
            config=config,
        )
        self.assertEqual(out.trajectory.shape, (bsz, num_waypoints, 2))
        self.assertTrue(torch.all(out.used_fallback))


if __name__ == "__main__":
    unittest.main()
