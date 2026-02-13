#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare NVIDIA Physical AI AV dataset clips for V-JEPA2 driving training.

Downloads a small number of clips from the NVIDIA Physical AI AV dataset
on Hugging Face, extracts video files and ego-motion data, and creates
a training CSV compatible with V-JEPA2's DrivingVideoDataset.

Usage:
    python scripts/prepare_nvidia_av_data.py

Requires:
    - physical_ai_av devkit installed (pip install -e physical_ai_av/)
    - HuggingFace token at /localhome/local-samehm/.hugging_face_token
"""

import io
import logging
import os
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN_PATH = "/localhome/local-samehm/.hugging_face_token"
PROJECT_ROOT = pathlib.Path("/localhome/local-samehm/vjepa2")
DATA_ROOT = PROJECT_ROOT / "data" / "nvidia_av"
VIDEO_DIR = DATA_ROOT / "videos"
EGOMOTION_DIR = DATA_ROOT / "egomotion"
TRAIN_CSV_PATH = DATA_ROOT / "train.csv"
HF_CACHE_DIR = str(PROJECT_ROOT / "hf_cache")

FEATURES_TO_DOWNLOAD = ["camera_front_wide_120fov", "egomotion"]

# Disk-space heuristics (overridable via env vars)
DEFAULT_BYTES_PER_CLIP = int(float(os.environ.get("VJEPA_BYTES_PER_CLIP_MB", "25")) * 1024 * 1024)
DISK_RESERVE_GB = float(os.environ.get("VJEPA_DISK_RESERVE_GB", "50"))
MAX_CLIPS_CAP = int(os.environ.get("VJEPA_MAX_CLIPS", "10000"))
MAX_SAMPLES_CAP = int(os.environ.get("VJEPA_MAX_SAMPLES", "30000"))
CLIP_STRIDE_SEC = float(os.environ.get("VJEPA_CLIP_STRIDE_SEC", "1.0"))
MAX_WINDOWS_PER_CLIP = int(os.environ.get("VJEPA_MAX_WINDOWS_PER_CLIP", "0"))

# Clip metadata derived from configs/train/vitg16/driving-joint-256px-16f.yaml:
#   frames_per_clip=16 sampled at 4 fps => 4-second video windows.
#   Trajectory head predicts 12 waypoints at 0.5s each => need 6s of ego data
#   beyond the clip end to compute training targets.
CLIP_DURATION_SEC = 4.0
FUTURE_MARGIN_SEC = 6.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("prepare_nvidia_av_data")


def _bytes_free_across_paths(paths: Iterable[pathlib.Path]) -> int:
    """Estimate total free bytes across unique filesystems hosting the given paths."""
    total = 0
    seen_devices = set()
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        stat_info = os.stat(resolved)
        if stat_info.st_dev in seen_devices:
            continue
        seen_devices.add(stat_info.st_dev)
        vfs = os.statvfs(resolved)
        total += vfs.f_bavail * vfs.f_frsize
    return total


def estimate_num_clips(avail_clip_count: int) -> tuple[int, int, int]:
    """Estimate maximum clips that fit given disk availability.

    Returns:
        (clip_count, total_free_bytes, usable_bytes_after_reserve)
    """
    reserve_bytes = int(DISK_RESERVE_GB * (1024**3))
    candidates = [
        DATA_ROOT,
        VIDEO_DIR,
        EGOMOTION_DIR,
        pathlib.Path(HF_CACHE_DIR),
    ]
    total_free = _bytes_free_across_paths(candidates)
    usable_bytes = max(total_free - reserve_bytes, 0)
    max_by_disk = usable_bytes // DEFAULT_BYTES_PER_CLIP
    est = min(avail_clip_count, MAX_CLIPS_CAP, max_by_disk)
    return int(est), total_free, usable_bytes


def read_hf_token(token_path: str) -> str:
    """Read the HuggingFace token from a file."""
    with open(token_path, "r") as f:
        token = f.read().strip()
    if not token:
        raise ValueError(f"HuggingFace token file is empty: {token_path}")
    logger.info("Successfully read HuggingFace token.")
    return token


def extract_video_bytes_from_zip(ds, clip_id: str, feature: str) -> bytes:
    """Extract raw video bytes from the dataset's zip file.

    Replicates the logic in dataset.get_clip_feature for camera features,
    but returns the raw video bytes instead of constructing a SeekVideoReader.
    This preserves the original encoding (lossless copy).
    """
    chunk_filename = ds.features.get_chunk_feature_filename(
        ds.get_clip_chunk(clip_id), feature
    )
    clip_files_in_zip = ds.features.get_clip_files_in_zip(clip_id, feature)
    with ds.open_file(chunk_filename) as f:
        with zipfile.ZipFile(f, "r") as zf:
            video_bytes = zf.read(clip_files_in_zip["video"])
    return video_bytes


def save_video(video_bytes: bytes, output_path: pathlib.Path) -> None:
    """Save raw video bytes to an MP4 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    size_mb = len(video_bytes) / (1024 * 1024)
    logger.info(f"  Saved video ({size_mb:.1f} MB): {output_path}")


def select_clip_windows_from_ego_df(
    ego_df: pd.DataFrame,
    clip_id: str,
) -> tuple[list[tuple[float, float]], float, float]:
    """Compute one or more valid clip windows from ego-motion samples."""
    ego_start = float(ego_df["timestamp"].iloc[0])
    ego_end = float(ego_df["timestamp"].iloc[-1])
    available = ego_end - ego_start
    required = CLIP_DURATION_SEC + FUTURE_MARGIN_SEC
    if available < required:
        raise ValueError(
            f"insufficient duration for clip {clip_id}: available={available:.3f}s < required={required:.3f}s"
        )

    min_start_time = ego_start + FUTURE_MARGIN_SEC
    max_start_time = ego_end - FUTURE_MARGIN_SEC - CLIP_DURATION_SEC
    if max_start_time < min_start_time:
        raise ValueError(
            f"cannot satisfy clip+future margin for clip {clip_id}: "
            f"min_start={min_start_time:.3f}s max_start={max_start_time:.3f}s"
        )

    starts = np.arange(min_start_time, max_start_time + 1e-6, CLIP_STRIDE_SEC, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([min_start_time], dtype=np.float64)
    elif abs(starts[-1] - max_start_time) > 1e-3:
        starts = np.append(starts, max_start_time)

    windows = [(float(s), float(s + CLIP_DURATION_SEC)) for s in starts]
    if MAX_WINDOWS_PER_CLIP > 0:
        windows = windows[:MAX_WINDOWS_PER_CLIP]
    return windows, ego_start, ego_end


def make_row_id(clip_id: str, clip_start_time: float, clip_end_time: float) -> str:
    return f"{clip_id}:{clip_start_time:.6f}:{clip_end_time:.6f}"


def load_existing_train_rows(train_csv_path: pathlib.Path) -> OrderedDict[str, dict[str, str]]:
    """Load previously written training rows to support resume."""
    rows: OrderedDict[str, dict[str, str]] = OrderedDict()
    if not train_csv_path.exists():
        logger.info("No existing training CSV found. Starting fresh.")
        return rows

    with open(train_csv_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                if line.strip():
                    logger.warning("  Skipping malformed training CSV line: %s", line.rstrip())
                continue
            video_path, ego_path, clip_start, clip_end, dataset_name = parts
            clip_id = pathlib.Path(video_path).stem
            clip_start_f = float(clip_start)
            clip_end_f = float(clip_end)
            rows[make_row_id(clip_id, clip_start_f, clip_end_f)] = {
                "video_path": video_path,
                "ego_motion_path": ego_path,
                "clip_start_time": f"{clip_start_f:.6f}",
                "clip_end_time": f"{clip_end_f:.6f}",
                "dataset_name": dataset_name,
                "clip_id": clip_id,
            }
    logger.info(
        "Loaded %d existing training entries from %s.",
        len(rows),
        train_csv_path,
    )
    return rows


def feature_zip_cached(ds, clip_id: str, feature: str) -> bool:
    """Return True if the HF cache already contains the chunk zip for this feature."""
    chunk_id = ds.get_clip_chunk(clip_id)
    chunk_filename = ds.features.get_chunk_feature_filename(chunk_id, feature)
    return ds.is_file_cached(chunk_filename)


def extract_egomotion_csv(
    ego_interpolator,
    frame_timestamps: np.ndarray,
    output_path: pathlib.Path,
) -> pd.DataFrame:
    """Interpolate ego-motion at frame timestamps and save to CSV.

    Args:
        ego_interpolator: Interpolator[EgomotionState] from the devkit.
        frame_timestamps: Array of timestamps in microseconds (from video reader).
        output_path: Where to write the ego-motion CSV.

    Returns:
        DataFrame with columns: timestamp, x, y, z, heading
        (timestamp in seconds, heading = yaw in radians)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clamp frame timestamps to the interpolator's valid range
    ego_t_min, ego_t_max = ego_interpolator.time_range
    valid_mask = (frame_timestamps >= ego_t_min) & (frame_timestamps <= ego_t_max)
    if not np.any(valid_mask):
        raise ValueError(
            f"No frame timestamps within ego-motion range "
            f"[{ego_t_min}, {ego_t_max}]"
        )
    valid_timestamps = frame_timestamps[valid_mask]
    logger.info(
        f"  Using {len(valid_timestamps)}/{len(frame_timestamps)} frame timestamps "
        f"within ego-motion range."
    )

    # Interpolate ego-motion at valid frame timestamps
    ego_states = ego_interpolator(valid_timestamps)

    # Extract position (x, y, z)
    positions = ego_states.pose.translation  # shape (N, 3)

    # Extract heading (yaw) from rotation using ZYX Euler angles
    # .as_euler('ZYX') returns [yaw, pitch, roll] per row
    euler_angles = ego_states.pose.rotation.as_euler("ZYX")
    yaw = euler_angles[:, 0]  # first column is yaw (Z-axis rotation)

    # Convert timestamps from microseconds to seconds
    timestamps_sec = valid_timestamps / 1e6

    # Build DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps_sec,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "heading": yaw,
        }
    )
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved ego-motion CSV ({len(df)} rows): {output_path}")
    return df


def main():
    logger.info("=" * 70)
    logger.info("NVIDIA Physical AI AV Data Preparation for V-JEPA2")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Read HuggingFace token
    # ------------------------------------------------------------------
    logger.info("Step 1: Reading HuggingFace token...")
    token = read_hf_token(HF_TOKEN_PATH)

    # ------------------------------------------------------------------
    # Step 2: Initialize the dataset interface
    # ------------------------------------------------------------------
    logger.info("Step 2: Initializing PhysicalAIAVDatasetInterface...")
    from physical_ai_av import PhysicalAIAVDatasetInterface

    ds = PhysicalAIAVDatasetInterface(
        token=token,
        cache_dir=HF_CACHE_DIR,
        confirm_download_threshold_gb=float("inf"),
    )
    logger.info(f"  Dataset interface initialized: {ds}")

    # ------------------------------------------------------------------
    # Step 3: List available clips and select a subset
    # ------------------------------------------------------------------
    logger.info("Step 3: Listing available clips...")
    all_clip_ids = ds.clip_index.index.tolist()
    logger.info(f"  Total clips available: {len(all_clip_ids)}")

    num_clips, total_free_bytes, usable_bytes = estimate_num_clips(len(all_clip_ids))
    if num_clips <= 0:
        logger.error(
            "Insufficient disk space. Free more than %.1f GB to download clips.",
            DISK_RESERVE_GB,
        )
        sys.exit(1)

    selected_clip_ids = all_clip_ids[:num_clips]
    logger.info(
        "  Free space: %.2f GB (usable %.2f GB after reserve %.1f GB) | Clip size %.2f MB | Selecting %d clips",
        total_free_bytes / (1024**3),
        usable_bytes / (1024**3),
        DISK_RESERVE_GB,
        DEFAULT_BYTES_PER_CLIP / (1024 * 1024),
        len(selected_clip_ids),
    )
    logger.debug(f"  Selected clip ids: {selected_clip_ids}")

    # ------------------------------------------------------------------
    # Step 4: Create output directories
    # ------------------------------------------------------------------
    logger.info("Step 4: Creating output directories...")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    EGOMOTION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Video dir:     {VIDEO_DIR}")
    logger.info(f"  Egomotion dir: {EGOMOTION_DIR}")

    # ------------------------------------------------------------------
    # Step 5: Download and process clips
    # ------------------------------------------------------------------
    logger.info("Step 5: Downloading and processing clips...")
    train_row_map = load_existing_train_rows(TRAIN_CSV_PATH)
    if len(train_row_map) >= MAX_SAMPLES_CAP:
        logger.info(
            "Existing CSV already has %d rows (>= sample cap %d). Rewriting and exiting.",
            len(train_row_map),
            MAX_SAMPLES_CAP,
        )
    successful_clips = 0
    generated_samples = 0
    reused_local_assets = 0

    for i, clip_id in enumerate(selected_clip_ids):
        if len(train_row_map) >= MAX_SAMPLES_CAP:
            logger.info("Reached sample cap (%d). Stopping early.", MAX_SAMPLES_CAP)
            break

        logger.info(f"\n--- Clip {i + 1}/{len(selected_clip_ids)}: {clip_id} ---")

        try:
            video_path = VIDEO_DIR / f"{clip_id}.mp4"
            ego_csv_path = EGOMOTION_DIR / f"{clip_id}.csv"

            video_exists = video_path.exists()
            ego_exists = ego_csv_path.exists()
            existing_row_keys = [
                row_id
                for row_id, row in train_row_map.items()
                if row.get("clip_id", pathlib.Path(row["video_path"]).stem) == clip_id
            ]
            for row_id in existing_row_keys:
                train_row_map.pop(row_id, None)
            if video_exists and ego_exists:
                logger.info("  Local video + ego CSV found. Skipping download and extraction.")
                reused_local_assets += 1
            else:
                missing_features = [
                    feature
                    for feature in FEATURES_TO_DOWNLOAD
                    if not feature_zip_cached(ds, clip_id, feature)
                ]
                if missing_features:
                    logger.info("  Downloading missing features: %s", missing_features)
                    ds.download_clip_features(
                        clip_id=clip_id,
                        features=missing_features,
                    )
                    logger.info("  Download complete.")
                else:
                    logger.info("  All requested features already cached. Skipping download.")

                if video_exists:
                    logger.info("  Video already exists. Skipping extraction: %s", video_path)
                else:
                    logger.info("  Extracting video bytes from zip...")
                    video_bytes = extract_video_bytes_from_zip(
                        ds, clip_id, "camera_front_wide_120fov"
                    )
                    save_video(video_bytes, video_path)

            # Extract ego-motion at frame timestamps and save CSV
            if ego_exists:
                logger.info("  Ego-motion CSV already exists. Reusing: %s", ego_csv_path)
                ego_df = pd.read_csv(ego_csv_path)
            else:
                logger.info("  Getting video reader for frame timestamps...")
                video_reader = ds.get_clip_feature(clip_id, "camera_front_wide_120fov")
                frame_timestamps = video_reader.timestamps  # microseconds
                logger.info(
                    f"  Frame timestamps: {len(frame_timestamps)} frames, "
                    f"range [{frame_timestamps[0]}, {frame_timestamps[-1]}] us"
                )
                video_reader.close()
                logger.info("  Getting ego-motion interpolator...")
                ego_interp = ds.get_clip_feature(clip_id, "egomotion")
                logger.info("  Interpolating ego-motion at frame timestamps...")
                ego_df = extract_egomotion_csv(
                    ego_interp, frame_timestamps, ego_csv_path
                )

            try:
                clip_windows, ego_start, ego_end = select_clip_windows_from_ego_df(
                    ego_df, clip_id
                )
            except ValueError as exc:
                logger.warning("  Skipping clip %s: %s", clip_id, exc)
                continue

            logger.info(
                "  Generated %d training windows (stride %.2fs, ego span %.3fs-%.3fs)",
                len(clip_windows),
                CLIP_STRIDE_SEC,
                ego_start,
                ego_end,
            )

            remaining = MAX_SAMPLES_CAP - len(train_row_map)
            if remaining <= 0:
                logger.info("  No sample budget remaining. Skipping window writes.")
                break
            if len(clip_windows) > remaining:
                logger.info(
                    "  Truncating clip windows from %d to %d to honor sample cap.",
                    len(clip_windows),
                    remaining,
                )
                clip_windows = clip_windows[:remaining]

            for clip_start_time, clip_end_time in clip_windows:
                row_id = make_row_id(clip_id, clip_start_time, clip_end_time)
                train_row_map[row_id] = {
                    "video_path": str(video_path),
                    "ego_motion_path": str(ego_csv_path),
                    "clip_start_time": f"{clip_start_time:.6f}",
                    "clip_end_time": f"{clip_end_time:.6f}",
                    "dataset_name": "nvidia_av",
                    "clip_id": clip_id,
                }
            generated_samples += len(clip_windows)

            successful_clips += 1
            logger.info(
                f"  Successfully processed clip {clip_id} "
                f"({len(clip_windows)} windows)"
            )

        except Exception as e:
            logger.error(f"  FAILED to process clip {clip_id}: {e}", exc_info=True)
            logger.info("  Skipping this clip and continuing...")
            continue

    # ------------------------------------------------------------------
    # Step 6: Create training CSV
    # ------------------------------------------------------------------
    logger.info(f"\nStep 6: Creating training CSV at {TRAIN_CSV_PATH}")

    if not train_row_map:
        logger.error("No clips were processed and no existing training data found.")
        sys.exit(1)

    # Write space-delimited CSV without header (as expected by DrivingVideoDataset)
    with open(TRAIN_CSV_PATH, "w") as f:
        for row in train_row_map.values():
            line = " ".join(
                [
                    row["video_path"],
                    row["ego_motion_path"],
                    row["clip_start_time"],
                    row["clip_end_time"],
                    row["dataset_name"],
                ]
            )
            f.write(line + "\n")

    logger.info(f"  Training CSV written with {len(train_row_map)} entries.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Clips attempted:  {len(selected_clip_ids)}")
    logger.info(f"  Clips succeeded (this run):  {successful_clips}")
    logger.info(f"  Clips reused local assets:   {reused_local_assets}")
    logger.info(f"  Samples generated (this run): {generated_samples}")
    logger.info(f"  Sample cap:                 {MAX_SAMPLES_CAP}")
    logger.info(
        "  Clips failed (this run):     %d",
        len(selected_clip_ids) - successful_clips,
    )
    logger.info(f"  Video directory:  {VIDEO_DIR}")
    logger.info(f"  Egomotion dir:    {EGOMOTION_DIR}")
    logger.info(f"  Training CSV:     {TRAIN_CSV_PATH}")
    logger.info("=" * 70)

    # Verify output
    logger.info("\nVerification:")
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    ego_files = list(EGOMOTION_DIR.glob("*.csv"))
    logger.info(f"  Video files:   {len(video_files)}")
    logger.info(f"  Ego CSV files: {len(ego_files)}")
    logger.info(f"  Train CSV exists: {TRAIN_CSV_PATH.exists()}")

    if TRAIN_CSV_PATH.exists():
        logger.info(f"\n  Training CSV contents:")
        with open(TRAIN_CSV_PATH, "r") as f:
            for line in f:
                logger.info(f"    {line.rstrip()}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
