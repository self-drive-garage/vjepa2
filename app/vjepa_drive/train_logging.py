# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.logging import AverageMeter, CSVLogger


@dataclass
class EpochMeters:
    loss: AverageMeter
    loss_jepa: AverageMeter
    loss_traj: AverageMeter
    lambda_traj: AverageMeter
    traj_ade: AverageMeter
    traj_fde: AverageMeter
    traj_nll: AverageMeter
    traj_calib: AverageMeter
    traj_smooth: AverageMeter
    traj_feasible: AverageMeter
    traj_top_prob: AverageMeter
    traj_entropy: AverageMeter
    traj_avg_std: AverageMeter
    mask: dict[int, AverageMeter]
    iter_time: AverageMeter
    gpu_time: AverageMeter
    data_time: AverageMeter


def make_csv_logger(log_file: str) -> CSVLogger:
    return CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_jepa"),
        ("%.5f", "loss_traj"),
        ("%.5f", "lambda_traj"),
        ("%.5f", "traj_min_ade"),
        ("%.5f", "traj_min_fde"),
        ("%.5f", "traj_nll"),
        ("%.5f", "traj_calib"),
        ("%.5f", "traj_smooth"),
        ("%.5f", "traj_feasible"),
        ("%.5f", "traj_top_prob"),
        ("%.5f", "traj_entropy"),
        ("%.5f", "traj_avg_std"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )


def make_epoch_meters(dataset_fpcs: list[int]) -> EpochMeters:
    return EpochMeters(
        loss=AverageMeter(),
        loss_jepa=AverageMeter(),
        loss_traj=AverageMeter(),
        lambda_traj=AverageMeter(),
        traj_ade=AverageMeter(),
        traj_fde=AverageMeter(),
        traj_nll=AverageMeter(),
        traj_calib=AverageMeter(),
        traj_smooth=AverageMeter(),
        traj_feasible=AverageMeter(),
        traj_top_prob=AverageMeter(),
        traj_entropy=AverageMeter(),
        traj_avg_std=AverageMeter(),
        mask={fpc: AverageMeter() for fpc in dataset_fpcs},
        iter_time=AverageMeter(),
        gpu_time=AverageMeter(),
        data_time=AverageMeter(),
    )


def update_meters(meters: EpochMeters, metrics: dict[str, float], iter_ms: float, gpu_ms: float, data_ms: float) -> None:
    meters.loss.update(metrics["loss"])
    meters.loss_jepa.update(metrics["loss_jepa"])
    meters.loss_traj.update(metrics["loss_traj"])
    meters.lambda_traj.update(metrics["lambda_traj"])
    meters.traj_ade.update(metrics["traj_ade"])
    meters.traj_fde.update(metrics["traj_fde"])
    meters.traj_nll.update(metrics["traj_nll"])
    meters.traj_calib.update(metrics["traj_calib"])
    meters.traj_smooth.update(metrics["traj_smooth"])
    meters.traj_feasible.update(metrics["traj_feasible"])
    meters.traj_top_prob.update(metrics["traj_top_prob"])
    meters.traj_entropy.update(metrics["traj_entropy"])
    meters.traj_avg_std.update(metrics["traj_avg_std"])
    meters.iter_time.update(iter_ms)
    meters.gpu_time.update(gpu_ms)
    meters.data_time.update(data_ms)


def log_iteration(
    logger,
    csv_logger: CSVLogger,
    epoch: int,
    itr: int,
    ipe: int,
    meters: EpochMeters,
    metrics: dict[str, float],
    wd: float,
    lr: float,
    log_freq: int,
) -> None:
    csv_logger.log(
        epoch + 1,
        itr,
        metrics["loss"],
        metrics["loss_jepa"],
        metrics["loss_traj"],
        metrics["lambda_traj"],
        metrics["traj_ade"],
        metrics["traj_fde"],
        metrics["traj_nll"],
        metrics["traj_calib"],
        metrics["traj_smooth"],
        metrics["traj_feasible"],
        metrics["traj_top_prob"],
        metrics["traj_entropy"],
        metrics["traj_avg_std"],
        meters.iter_time.val,
        meters.gpu_time.val,
        meters.data_time.val,
    )
    if (itr % log_freq == 0) or (itr == ipe - 1) or metrics["is_bad"]:
        logger.info(
            "[%d, %5d] loss: %.3f "
            "[jepa: %.3f] [traj: %.3f] [lambda_traj: %.3f] "
            "[ade: %.3f] [fde: %.3f] [nll: %.3f] [calib: %.3f] "
            "[smooth: %.3f] [feas: %.3f] [pmax: %.3f] [ent: %.3f] [std: %.3f] "
            "masks: %s "
            "[wd: %.2e] [lr: %.2e] "
            "[mem: %.2e] "
            "[iter: %.1f ms] "
            "[gpu: %.1f ms] "
            "[data: %.1f ms]"
            % (
                epoch + 1,
                itr,
                meters.loss.avg,
                meters.loss_jepa.avg,
                meters.loss_traj.avg,
                meters.lambda_traj.avg,
                meters.traj_ade.avg,
                meters.traj_fde.avg,
                meters.traj_nll.avg,
                meters.traj_calib.avg,
                meters.traj_smooth.avg,
                meters.traj_feasible.avg,
                meters.traj_top_prob.avg,
                meters.traj_entropy.avg,
                meters.traj_avg_std.avg,
                "[" + ", ".join([f"{k}: " + "%.1f" % meters.mask[k].avg for k in meters.mask]) + "]",
                wd,
                lr,
                (torch.cuda.max_memory_allocated() / 1024.0**2) if torch.cuda.is_available() else 0.0,
                meters.iter_time.avg,
                meters.gpu_time.avg,
                meters.data_time.avg,
            )
        )


def log_epoch_summary(logger, meters: EpochMeters) -> None:
    logger.info(
        "avg. loss %.3f [jepa: %.3f] [traj: %.3f] [lambda_traj: %.3f] "
        "[ade: %.3f] [fde: %.3f] [nll: %.3f] [calib: %.3f] [smooth: %.3f] [feas: %.3f]"
        % (
            meters.loss.avg,
            meters.loss_jepa.avg,
            meters.loss_traj.avg,
            meters.lambda_traj.avg,
            meters.traj_ade.avg,
            meters.traj_fde.avg,
            meters.traj_nll.avg,
            meters.traj_calib.avg,
            meters.traj_smooth.avg,
            meters.traj_feasible.avg,
        )
    )
