# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Driving video dataset with ego-motion trajectory ground truth.

Extends V-JEPA2's video dataset patterns to additionally return
future ego-vehicle waypoints for joint JEPA + trajectory training.
"""

import math
import os
import pathlib
import warnings
from collections import OrderedDict
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu

from app.vjepa_drive.types import DrivingBatch, DrivingSample
from src.datasets.ego_motion_utils import (
    compute_future_waypoints_from_poses,
    compute_history_waypoints_from_poses,
    get_ego_motion_loader,
)
from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_drivingvideodataset(
    data_paths,
    batch_size,
    frames_per_clip=16,
    dataset_fpcs=None,
    fps=None,
    duration=None,
    frame_step=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    # Driving-specific params
    trajectory_horizon=12,
    trajectory_dt=0.5,
    trajectory_history_horizon=8,
    trajectory_history_dt=0.5,
    dataset_names=None,
    max_resample_attempts=64,
    prefetch_factor=2,
    decord_num_threads=2,
    ego_cache_size=512,
):
    dataset = DrivingVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        duration=duration,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        trajectory_horizon=trajectory_horizon,
        trajectory_dt=trajectory_dt,
        trajectory_history_horizon=trajectory_history_horizon,
        trajectory_history_dt=trajectory_history_dt,
        dataset_names=dataset_names,
        max_resample_attempts=max_resample_attempts,
        decord_num_threads=decord_num_threads,
        ego_cache_size=ego_cache_size,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("DrivingVideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": collator,
        "sampler": dist_sampler,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "pin_memory": pin_mem,
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0) and persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            **loader_kwargs,
        )
    else:
        data_loader = NondeterministicDataLoader(
            **loader_kwargs,
        )
    logger.info("DrivingVideoDataset data loader created")

    return dataset, data_loader, dist_sampler


class DrivingVideoDataset(torch.utils.data.Dataset):
    """Video dataset with ego-motion trajectory ground truth.

    Each sample returns:
    - buffer: list of [C, T, H, W] video clips (same as VideoDataset)
    - gt_trajectory: [horizon, 2] future ego waypoints in ego frame
    - ego_history: [history_horizon, 2] past ego waypoints in ego frame
    - clip_indices: frame indices (same as VideoDataset)

    CSV format (space-delimited):
        video_path ego_motion_path clip_start_time clip_end_time dataset_name

    Where:
        - video_path: path to video file
        - ego_motion_path: path to ego-motion data file
        - clip_start_time: start timestamp of clip in seconds
        - clip_end_time: end timestamp of clip in seconds
        - dataset_name: name of dataset (nuscenes, argoverse, waymo, generic)
    """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=None,
        duration=None,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        trajectory_horizon=12,
        trajectory_dt=0.5,
        trajectory_history_horizon=8,
        trajectory_history_dt=0.5,
        dataset_names=None,
        max_resample_attempts=64,
        decord_num_threads=2,
        ego_cache_size=512,
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.trajectory_horizon = trajectory_horizon
        self.trajectory_dt = trajectory_dt
        self.trajectory_history_horizon = trajectory_history_horizon
        self.trajectory_history_dt = trajectory_history_dt
        self.max_resample_attempts = int(max(1, max_resample_attempts))
        self.decord_num_threads = int(max(1, decord_num_threads))
        self.ego_cache_size = int(max(0, ego_cache_size))

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for data paths")
            self.dataset_fpcs = dataset_fpcs

        if dataset_names is None:
            self.dataset_names_list = ["generic" for _ in data_paths]
        elif isinstance(dataset_names, str):
            self.dataset_names_list = [dataset_names for _ in data_paths]
        else:
            self.dataset_names_list = dataset_names

        # Load samples from CSV files
        # CSV columns: video_path ego_motion_path clip_start_time clip_end_time [dataset_name]
        samples = []
        self.num_samples_per_dataset = []
        for data_path_idx, data_path in enumerate(data_paths):
            data = pd.read_csv(data_path, header=None, delimiter=" ")
            n_cols = data.shape[1]
            for _, row in data.iterrows():
                sample = {
                    "video_path": str(row.iloc[0]),
                    "ego_path": str(row.iloc[1]),
                    "clip_start_time": float(row.iloc[2]),
                    "clip_end_time": float(row.iloc[3]),
                    "dataset_name": str(row.iloc[4]) if n_cols > 4 else self.dataset_names_list[data_path_idx],
                }
                samples.append(sample)
            self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # Weights for weighted sampling
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples

        # Cache ego-motion loaders by dataset name
        self._ego_loaders = {}
        # Repeated clip windows often share the same ego file; memoize parsed state per worker.
        self._ego_state_cache = OrderedDict()

    def _get_ego_loader(self, dataset_name):
        if dataset_name not in self._ego_loaders:
            self._ego_loaders[dataset_name] = get_ego_motion_loader(dataset_name)
        return self._ego_loaders[dataset_name]

    def _load_ego_motion(self, sample):
        ego_path = sample["ego_path"]
        dataset_name = sample["dataset_name"]

        if self.ego_cache_size > 0 and ego_path in self._ego_state_cache:
            self._ego_state_cache.move_to_end(ego_path)
            return self._ego_state_cache[ego_path]

        ego_loader = self._get_ego_loader(dataset_name)
        ego_state = ego_loader.load(ego_path)
        if self.ego_cache_size > 0:
            self._ego_state_cache[ego_path] = ego_state
            if len(self._ego_state_cache) > self.ego_cache_size:
                self._ego_state_cache.popitem(last=False)
        return ego_state

    def __getitem__(self, index):
        if self.__len__() == 0:
            raise RuntimeError("DrivingVideoDataset is empty; check data_paths and CSV contents.")

        cur_index = int(index)
        for _ in range(self.max_resample_attempts):
            loaded_sample = self._get_item(cur_index)
            if loaded_sample is not None:
                return loaded_sample
            cur_index = int(np.random.randint(self.__len__()))

        failed_sample = self.samples[cur_index]
        raise RuntimeError(
            "Exceeded max_resample_attempts when loading driving sample. "
            f"max_resample_attempts={self.max_resample_attempts} "
            f"last_video={failed_sample.get('video_path')} "
            f"last_ego={failed_sample.get('ego_path')}"
        )

    def _get_item(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        # Load video frames and the sampled clip end time used for trajectory supervision.
        buffer, clip_indices, sampled_clip_end_time = self._loadvideo_decord(
            sample["video_path"], frames_per_clip, sample
        )
        if len(buffer) == 0:
            return None

        # Load ego-motion and compute trajectory
        try:
            timestamps, positions, headings = self._load_ego_motion(sample)
            gt_trajectory = compute_future_waypoints_from_poses(
                timestamps=timestamps,
                positions=positions,
                headings=headings,
                # Keep trajectory target aligned with the actual sampled video frames.
                ref_timestamp=sampled_clip_end_time,
                horizon=self.trajectory_horizon,
                dt=self.trajectory_dt,
            )
            ego_history = compute_history_waypoints_from_poses(
                timestamps=timestamps,
                positions=positions,
                headings=headings,
                ref_timestamp=sampled_clip_end_time,
                horizon=self.trajectory_history_horizon,
                dt=self.trajectory_history_dt,
            )
            if gt_trajectory is None or ego_history is None:
                return None
        except Exception as e:
            logger.warning(f"Failed to load ego-motion for {sample['ego_path']}: {e}")
            return None

        # Apply transforms
        def split_into_clips(video):
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return DrivingSample(
            buffer=buffer,
            gt_trajectory=gt_trajectory,
            ego_history=ego_history,
            clip_indices=clip_indices,
        )

    def _loadvideo_decord(self, fname, fpc, sample=None):
        """Load video using Decord (mirrors VideoDataset.loadvideo_decord)."""
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None, None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None, None

        try:
            vr = VideoReader(fname, num_threads=self.decord_num_threads, ctx=cpu(0))
        except Exception:
            return [], None, None

        try:
            video_fps = float(vr.get_avg_fps())
            if video_fps <= 0:
                raise ValueError(f"Invalid FPS {video_fps}")
        except Exception as e:
            logger.warning(e)
            return [], None, None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            if self.duration is not None:
                fstp = int(self.duration * video_fps / fpc)
            else:
                fstp = int(math.ceil(video_fps / self.fps))

        if fstp is None or fstp <= 0:
            return [], None, None
        clip_len = int(fpc * fstp)

        def _frame_idx_to_timestamp_seconds(frame_idx: int) -> float:
            frame_idx = int(frame_idx)
            try:
                ts = vr.get_frame_timestamp(frame_idx)
                if isinstance(ts, np.ndarray):
                    ts = float(ts.reshape(-1)[0])
                elif isinstance(ts, (list, tuple)):
                    ts = float(ts[0])
                else:
                    ts = float(ts)
                if np.isfinite(ts):
                    return ts
            except Exception:
                pass
            return float(frame_idx) / video_fps

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None, None

        vr.seek(0)

        # If CSV provides a clip window, prioritize deterministic time alignment.
        # This keeps video frames and trajectory targets synchronized.
        if (
            sample is not None
            and self.num_clips == 1
            and "clip_start_time" in sample
            and "clip_end_time" in sample
        ):
            clip_start_sec = float(sample["clip_start_time"])
            clip_end_sec = float(sample["clip_end_time"])
            if clip_end_sec > clip_start_sec:
                max_frame_idx = max(len(vr) - 1, 0)
                start_idx = int(round(clip_start_sec * video_fps))
                end_idx = int(round(clip_end_sec * video_fps)) - 1
                start_idx = int(np.clip(start_idx, 0, max_frame_idx))
                end_idx = int(np.clip(max(end_idx, start_idx), start_idx, max_frame_idx))
                if end_idx == start_idx:
                    indices = np.full((fpc,), start_idx, dtype=np.int64)
                else:
                    indices = np.linspace(start_idx, end_idx, num=fpc)
                    indices = np.clip(np.round(indices), start_idx, end_idx).astype(np.int64)
                clip_indices = [indices]
                buffer = vr.get_batch(indices.tolist()).asnumpy()
                sampled_clip_end_time = _frame_idx_to_timestamp_seconds(indices[-1])
                return buffer, clip_indices, sampled_clip_end_time

        partition_len = len(vr) // self.num_clips
        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    ))
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - sample_len // fstp) * sample_len,
                    ))
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        sampled_clip_end_time = _frame_idx_to_timestamp_seconds(clip_indices[-1][-1])
        return buffer, clip_indices, sampled_clip_end_time

    def __len__(self):
        return len(self.samples)


class DrivingMaskCollator:
    """Wraps V-JEPA2's MaskCollator to also handle trajectory ground truth.

    Extends the standard MaskCollator collation to additionally stack
    trajectory ground truth tensors.

    The standard MaskCollator returns:
        fpc_collations: list of (collated_batch, masks_enc, masks_pred)
        where collated_batch is [videos, labels, clip_indices].

    This wrapper returns:
        fpc_collations: list of DrivingBatch objects with explicit fields
        (videos, labels, clip_indices, masks_enc, masks_pred, gt_trajectories, ego_histories)

    where gt_trajectories=[B, horizon, 2] and ego_histories=[B, history_horizon, 2].
    """

    def __init__(self, mask_collator):
        self.mask_collator = mask_collator

    def step(self):
        self.mask_collator.step()

    def __call__(self, batch):
        # batch is a list of DrivingSample objects.
        # We need to separate trajectories/history from the rest before calling MaskCollator

        # Reconstruct the batch in the format MaskCollator expects:
        # (buffer, label, clip_indices) — we use gt_trajectory as the "label"
        # but also keep it separate for stacking

        # Group by fpc (frames per clip) like MaskCollator does
        filtered_batches = {}
        for sample in batch:
            fpc = len(sample.clip_indices[-1])
            if fpc not in filtered_batches:
                filtered_batches[fpc] = []
            filtered_batches[fpc].append(sample)

        fpc_collations = []
        for fpc, fpc_batch in filtered_batches.items():
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue

            # Separate trajectory targets and ego-history conditioning.
            gt_trajectories = torch.stack([s.gt_trajectory for s in fpc_batch])
            ego_histories = torch.stack([s.ego_history for s in fpc_batch])

            # Create a batch in the format MaskCollator expects
            # MaskCollator expects: list of (buffer, label, clip_indices)
            mask_batch = [(s.buffer, 0, s.clip_indices) for s in fpc_batch]

            # Call the underlying MaskCollator on this single-fpc batch
            # MaskCollator internally groups by fpc, but here we've already
            # pre-grouped, so we get exactly one fpc_collation back
            mask_result = self.mask_collator(mask_batch)

            for collated_batch, masks_enc, masks_pred in mask_result:
                videos, labels, clip_indices = collated_batch
                fpc_collations.append(
                    DrivingBatch(
                        videos=list(videos),
                        labels=labels,
                        clip_indices=list(clip_indices),
                        masks_enc=masks_enc,
                        masks_pred=masks_pred,
                        gt_trajectories=gt_trajectories,
                        ego_histories=ego_histories,
                    )
                )

        return fpc_collations
