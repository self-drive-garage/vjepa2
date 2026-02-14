# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from src.datasets.ego_motion_utils import (
    compute_future_waypoints_from_poses,
    compute_history_waypoints_from_poses,
)


class TestEgoMotionUtils(unittest.TestCase):
    def test_future_and_history_waypoints(self):
        # Simple straight-line motion in +x with 1 m/s and zero heading.
        timestamps = np.arange(0.0, 11.0, 1.0, dtype=np.float64)
        positions = np.stack([timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)], axis=1)
        headings = np.zeros_like(timestamps)
        ref = 5.0

        future = compute_future_waypoints_from_poses(
            timestamps=timestamps,
            positions=positions,
            headings=headings,
            ref_timestamp=ref,
            horizon=2,
            dt=1.0,
        )
        history = compute_history_waypoints_from_poses(
            timestamps=timestamps,
            positions=positions,
            headings=headings,
            ref_timestamp=ref,
            horizon=2,
            dt=1.0,
        )

        np.testing.assert_allclose(future.numpy(), np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(history.numpy(), np.array([[-2.0, 0.0], [-1.0, 0.0]], dtype=np.float32), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
