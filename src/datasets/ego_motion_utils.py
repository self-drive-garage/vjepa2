# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ego-motion utilities for computing trajectory waypoints from driving datasets.

Provides a unified interface for extracting future ego-vehicle waypoints
from multiple autonomous driving dataset formats (nuScenes, Argoverse v2,
Waymo, NVIDIA Physical AI AV).
"""

import json
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


def compute_future_waypoints_from_poses(
    timestamps,
    positions,
    headings,
    ref_timestamp,
    horizon=12,
    dt=0.5,
):
    """Compute future (x, y) waypoints in ego frame at ref_timestamp.

    Given a sequence of timestamped ego poses, computes future waypoints
    relative to the ego frame at ref_timestamp.

    Args:
        timestamps: [N] array of timestamps in seconds (float64)
        positions: [N, 3] array of (x, y, z) positions in world frame
        headings: [N] array of yaw angles in radians
        ref_timestamp: reference timestamp (end of observed clip)
        horizon: number of future waypoints to predict
        dt: time step between waypoints in seconds
    Returns:
        waypoints: [horizon, 2] tensor of (x, y) in ego frame, or None if
                   the required future timestamps exceed available data
    """
    # Find reference pose via interpolation
    ref_pos = _interp_position(timestamps, positions, ref_timestamp)
    ref_heading = _interp_heading(timestamps, headings, ref_timestamp)
    if ref_pos is None or ref_heading is None:
        return None

    # Compute rotation matrix for ego frame (inverse of reference heading)
    cos_h = np.cos(-ref_heading)
    sin_h = np.sin(-ref_heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    waypoints = []
    for i in range(1, horizon + 1):
        t_future = ref_timestamp + i * dt
        future_pos = _interp_position(timestamps, positions, t_future)
        if future_pos is None:
            return None
        # Transform to ego frame
        delta = future_pos[:2] - ref_pos[:2]
        wp_ego = rot @ delta
        waypoints.append(wp_ego)

    return torch.tensor(np.stack(waypoints), dtype=torch.float32)


def _interp_position(timestamps, positions, t):
    """Linearly interpolate position at time t."""
    if t < timestamps[0] or t > timestamps[-1]:
        return None
    idx = np.searchsorted(timestamps, t, side='right') - 1
    idx = min(idx, len(timestamps) - 2)
    t0, t1 = timestamps[idx], timestamps[idx + 1]
    if t1 == t0:
        return positions[idx]
    alpha = (t - t0) / (t1 - t0)
    return (1 - alpha) * positions[idx] + alpha * positions[idx + 1]


def _interp_heading(timestamps, headings, t):
    """Linearly interpolate heading (with angle wrapping) at time t."""
    if t < timestamps[0] or t > timestamps[-1]:
        return None
    idx = np.searchsorted(timestamps, t, side='right') - 1
    idx = min(idx, len(timestamps) - 2)
    t0, t1 = timestamps[idx], timestamps[idx + 1]
    if t1 == t0:
        return headings[idx]
    alpha = (t - t0) / (t1 - t0)
    # Handle angle wrapping
    diff = headings[idx + 1] - headings[idx]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return headings[idx] + alpha * diff


class EgoMotionLoader:
    """Base class for dataset-specific ego-motion loading."""

    def load(self, ego_path):
        """Load ego-motion data from a file path.

        Returns:
            timestamps: [N] array of timestamps in seconds (float64)
            positions: [N, 3] array of (x, y, z) world positions
            headings: [N] array of yaw angles in radians
        """
        raise NotImplementedError


class NuScenesEgoMotionLoader(EgoMotionLoader):
    """Load ego-motion from nuScenes ego_pose JSON format.

    Expected format: JSON list of dicts, each with:
        - "timestamp": int (microseconds)
        - "translation": [x, y, z]
        - "rotation": [w, x, y, z] quaternion
    """

    def load(self, ego_path):
        with open(ego_path, "r") as f:
            data = json.load(f)

        timestamps = np.array([d["timestamp"] / 1e6 for d in data], dtype=np.float64)
        positions = np.array([d["translation"] for d in data], dtype=np.float64)
        # Extract yaw from quaternion [w, x, y, z]
        headings = np.array([
            _quat_to_yaw(d["rotation"]) for d in data
        ], dtype=np.float64)

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]


class ArgoverseEgoMotionLoader(EgoMotionLoader):
    """Load ego-motion from Argoverse v2 city_SE3_egovehicle format.

    Expected format: JSON with "timestamps_ns" list and "poses" list of
    dicts each with "translation" [x, y, z] and "rotation" [qw, qx, qy, qz].
    Alternatively, a CSV with columns: timestamp_ns, tx, ty, tz, qw, qx, qy, qz
    """

    def load(self, ego_path):
        if ego_path.endswith(".json"):
            return self._load_json(ego_path)
        else:
            return self._load_csv(ego_path)

    def _load_json(self, ego_path):
        with open(ego_path, "r") as f:
            data = json.load(f)
        timestamps = np.array(data["timestamps_ns"], dtype=np.float64) / 1e9
        positions = np.array([p["translation"] for p in data["poses"]], dtype=np.float64)
        headings = np.array([
            _quat_to_yaw(p["rotation"]) for p in data["poses"]
        ], dtype=np.float64)
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]

    def _load_csv(self, ego_path):
        import pandas as pd
        df = pd.read_csv(ego_path)
        timestamps = df["timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
        positions = df[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)
        quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
        headings = np.array([_quat_to_yaw(q) for q in quats], dtype=np.float64)
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]


class WaymoEgoMotionLoader(EgoMotionLoader):
    """Load ego-motion from Waymo Open Dataset format.

    Expected format: JSON list of dicts with:
        - "timestamp_micros": int
        - "transform": 4x4 flattened list (row-major)
    """

    def load(self, ego_path):
        with open(ego_path, "r") as f:
            data = json.load(f)

        timestamps = np.array([d["timestamp_micros"] / 1e6 for d in data], dtype=np.float64)
        positions = []
        headings = []
        for d in data:
            tf = np.array(d["transform"]).reshape(4, 4)
            positions.append(tf[:3, 3])
            headings.append(np.arctan2(tf[1, 0], tf[0, 0]))
        positions = np.array(positions, dtype=np.float64)
        headings = np.array(headings, dtype=np.float64)
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]


class GenericCSVEgoMotionLoader(EgoMotionLoader):
    """Load ego-motion from a generic CSV format.

    Expected CSV columns: timestamp, x, y, z, yaw
    timestamp in seconds, positions in meters, yaw in radians.
    """

    def load(self, ego_path):
        import pandas as pd
        df = pd.read_csv(ego_path)
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        positions = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        headings = df["yaw"].to_numpy(dtype=np.float64)
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]


class NvidiaAVEgoMotionLoader(EgoMotionLoader):
    """Load ego-motion from NVIDIA Physical AI AV CSV format.

    Expected CSV columns: timestamp, x, y, z, heading
    timestamp in seconds, positions in meters, heading (yaw) in radians.

    Generated by scripts/prepare_nvidia_av_data.py from the NVIDIA
    PhysicalAI-Autonomous-Vehicles dataset on Hugging Face.
    """

    def load(self, ego_path):
        import pandas as pd
        df = pd.read_csv(ego_path)
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        positions = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        headings = df["heading"].to_numpy(dtype=np.float64)
        sort_idx = np.argsort(timestamps)
        return timestamps[sort_idx], positions[sort_idx], headings[sort_idx]


# Registry of ego-motion loaders by dataset name
EGO_MOTION_LOADERS = {
    "nuscenes": NuScenesEgoMotionLoader(),
    "argoverse": ArgoverseEgoMotionLoader(),
    "waymo": WaymoEgoMotionLoader(),
    "nvidia_av": NvidiaAVEgoMotionLoader(),
    "generic": GenericCSVEgoMotionLoader(),
}


def get_ego_motion_loader(dataset_name):
    """Get the ego-motion loader for a given dataset name."""
    dataset_name = dataset_name.lower()
    if dataset_name not in EGO_MOTION_LOADERS:
        logger.warning(
            f"Unknown dataset '{dataset_name}', falling back to generic CSV loader"
        )
        return EGO_MOTION_LOADERS["generic"]
    return EGO_MOTION_LOADERS[dataset_name]


def _quat_to_yaw(q):
    """Convert quaternion to yaw angle.

    Args:
        q: quaternion as [w, x, y, z] or [qw, qx, qy, qz]
    Returns:
        yaw angle in radians
    """
    if len(q) == 4:
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Expected quaternion of length 4, got {len(q)}")
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)
