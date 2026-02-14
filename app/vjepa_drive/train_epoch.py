# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import gc
import time

import numpy as np
import torch

from app.vjepa_drive.losses import TrajectoryLossWeights, schedule_scalar
from app.vjepa_drive.train_logging import (
    EpochMeters,
    log_iteration,
    make_epoch_meters,
    update_meters,
)
from app.vjepa_drive.train_loss import (
    compute_jepa_loss,
    compute_trajectory_stats,
    forward_encoder_full,
    forward_predictor_from_full,
    forward_target_encoder,
)
from src.utils.logging import gpu_timer

GARBAGE_COLLECT_ITR_FREQ = 50


def _autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def _update_target_encoder(encoder, target_encoder, momentum_value: float) -> None:
    with torch.no_grad():
        params_k = []
        params_q = []
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            params_k.append(param_k)
            params_q.append(param_q)
        torch._foreach_mul_(params_k, momentum_value)
        torch._foreach_add_(params_k, params_q, alpha=1 - momentum_value)


def run_training_epoch(
    *,
    epoch: int,
    ipe: int,
    batch_size: int,
    dataset_fpcs: list[int],
    loader,
    unsupervised_loader,
    unsupervised_sampler,
    device: torch.device,
    encoder,
    predictor,
    target_encoder,
    predict_traj_distribution,
    optimizer,
    scaler,
    scheduler,
    wd_scheduler,
    momentum_scheduler,
    mixed_precision: bool,
    dtype: torch.dtype,
    loss_exp: float,
    trajectory_dt: float,
    traj_loss_weights: TrajectoryLossWeights,
    lambda_traj_start: float,
    lambda_traj_end: float,
    lambda_traj_schedule: str,
    lambda_traj_ramp_epochs: float,
    sync_gc: bool,
    csv_logger,
    logger,
    log_freq: int,
) -> tuple[object, EpochMeters]:
    meters = make_epoch_meters(dataset_fpcs)

    for itr in range(ipe):
        itr_start_time = time.time()

        iter_retries = 0
        iter_successful = False
        while not iter_successful:
            try:
                sample = next(loader)
                iter_successful = True
            except StopIteration:
                logger.info("Exhausted data loaders. Refreshing...")
                unsupervised_sampler.set_epoch(epoch)
                loader = iter(unsupervised_loader)
            except Exception as e:
                num_retries = 5
                if iter_retries < num_retries:
                    logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                    iter_retries += 1
                    time.sleep(5)
                else:
                    logger.warning(f"Exceeded max retries ({num_retries}) when loading data. Skipping batch.")
                    raise e

        for fpc_sample in sample:
            bs, fpc = fpc_sample.batch_size_and_fpc()
            meters.mask[fpc].update(bs / batch_size)

        clips, masks_enc, masks_pred, gt_trajectories, ego_histories = [], [], [], [], []
        for fpc_sample in sample:
            fpc_batch = fpc_sample.to_device(device)
            clips.append(fpc_batch.video())
            masks_enc.append(fpc_batch.masks_enc)
            masks_pred.append(fpc_batch.masks_pred)
            gt_trajectories.append(fpc_batch.gt_trajectories)
            ego_histories.append(fpc_batch.ego_histories)

        data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
        if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
            logger.info("Running garbage collection...")
            gc.collect()

        def train_step():
            new_lr = scheduler.step()
            new_wd = wd_scheduler.step()
            ramp_epochs = max(float(lambda_traj_ramp_epochs), 1e-6)
            ramp_progress = (epoch + (itr + 1) / max(ipe, 1)) / ramp_epochs
            lambda_traj = schedule_scalar(
                start=lambda_traj_start,
                end=lambda_traj_end,
                progress=ramp_progress,
                schedule=lambda_traj_schedule,
            )

            with _autocast_context(device=device, dtype=dtype, enabled=mixed_precision):
                h = forward_target_encoder(target_encoder, clips)
                z_full = forward_encoder_full(encoder, clips)
                z_pred = forward_predictor_from_full(predictor, z_full, masks_enc, masks_pred)
                loss_jepa = compute_jepa_loss(z_pred, h, masks_pred, loss_exp)
                traj_stats = compute_trajectory_stats(
                    z_full=z_full,
                    gt_trajectories=gt_trajectories,
                    ego_histories=ego_histories,
                    predict_fn=predict_traj_distribution,
                    trajectory_dt=trajectory_dt,
                    weights=traj_loss_weights,
                )
                loss_traj = traj_stats["loss"]
                loss = loss_jepa + lambda_traj * loss_traj

            if mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            momentum_value = next(momentum_scheduler)
            _update_target_encoder(encoder, target_encoder, momentum_value)

            return {
                "loss": float(loss),
                "loss_jepa": float(loss_jepa),
                "loss_traj": float(loss_traj),
                "lambda_traj": float(lambda_traj),
                "traj_ade": float(traj_stats["min_ade"]),
                "traj_fde": float(traj_stats["min_fde"]),
                "traj_nll": float(traj_stats["nll"]),
                "traj_calib": float(traj_stats["calib"]),
                "traj_smooth": float(traj_stats["smoothness"]),
                "traj_feasible": float(traj_stats["feasibility"]),
                "traj_top_prob": float(traj_stats["top_prob"]),
                "traj_entropy": float(traj_stats["mode_entropy"]),
                "traj_avg_std": float(traj_stats["avg_std"]),
                "wd": float(new_wd),
                "lr": float(new_lr),
            }

        metrics, gpu_etime_ms = gpu_timer(train_step)
        iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
        metrics["is_bad"] = bool(np.isnan(metrics["loss"]) or np.isinf(metrics["loss"]))

        update_meters(
            meters=meters,
            metrics=metrics,
            iter_ms=iter_elapsed_time_ms,
            gpu_ms=gpu_etime_ms,
            data_ms=data_elapsed_time_ms,
        )
        log_iteration(
            logger=logger,
            csv_logger=csv_logger,
            epoch=epoch,
            itr=itr,
            ipe=ipe,
            meters=meters,
            metrics=metrics,
            wd=metrics["wd"],
            lr=metrics["lr"],
            log_freq=log_freq,
        )
        assert not np.isnan(metrics["loss"]), "loss is nan"

    return loader, meters
