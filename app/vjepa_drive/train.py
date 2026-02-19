# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    if ("CUDA_VISIBLE_DEVICES" not in os.environ) and ("LOCAL_RANK" not in os.environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_drive.losses import TrajectoryLossWeights
from app.vjepa_drive.train_config import parse_train_sections
from app.vjepa_drive.train_epoch import run_training_epoch
from app.vjepa_drive.train_logging import log_epoch_summary, make_csv_logger
from app.vjepa_drive.transforms import make_transforms
from app.vjepa_drive.utils import init_drive_opt, init_trajectory_head, load_checkpoint
from app.vjepa.utils import init_video_model
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger

log_freq = 10
CHECKPOINT_FREQ = 1

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


logger = get_logger(__name__, force=True)


def _resolve_local_cuda_index() -> int:
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is None:
        return 0
    try:
        local_rank = int(local_rank)
    except Exception:
        return 0

    # If the process has already been pinned to one GPU, always use cuda:0.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and ("," not in visible):
        return 0
    return max(local_rank, 0)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PARSE CONFIG
    # ----------------------------------------------------------------------- #
    sections = parse_train_sections(args, resume_preempt=resume_preempt)
    logger.info("dtype=%s", sections.meta.dtype)

    folder = sections.meta.folder
    load_model = sections.meta.load_model
    r_file = sections.meta.read_checkpoint
    seed = sections.meta.seed
    save_every_freq = sections.meta.save_every_freq
    skip_batches = sections.meta.skip_batches
    use_sdpa = sections.meta.use_sdpa
    sync_gc = sections.meta.sync_gc
    dtype = sections.meta.dtype
    mixed_precision = sections.meta.mixed_precision

    cfgs_mask = sections.mask_cfg

    cfgs_traj = sections.model.traj_cfg
    compile_model = sections.model.compile_model
    use_activation_checkpointing = sections.model.use_activation_checkpointing
    predictor_find_unused_parameters = sections.model.predictor_find_unused_parameters
    model_name = sections.model.model_name
    pred_depth = sections.model.pred_depth
    pred_num_heads = sections.model.pred_num_heads
    pred_embed_dim = sections.model.pred_embed_dim
    uniform_power = sections.model.uniform_power
    use_mask_tokens = sections.model.use_mask_tokens
    zero_init_mask_tokens = sections.model.zero_init_mask_tokens
    use_rope = sections.model.use_rope
    use_silu = sections.model.use_silu
    use_pred_silu = sections.model.use_pred_silu
    wide_silu = sections.model.wide_silu

    num_waypoints = int(cfgs_traj.get("num_waypoints", sections.data.temporal_recipe.future_horizon))
    waypoint_dim = int(cfgs_traj.get("waypoint_dim", 2))
    num_modes = int(cfgs_traj.get("num_modes", 4))
    history_dim = int(cfgs_traj.get("history_dim", 2))
    history_hidden_dim = cfgs_traj.get("history_hidden_dim", None)
    history_tokens = int(cfgs_traj.get("history_tokens", sections.data.temporal_recipe.history_horizon))
    min_log_std = float(cfgs_traj.get("min_log_std", -4.0))
    max_log_std = float(cfgs_traj.get("max_log_std", 1.0))
    traj_pooler_depth = int(cfgs_traj.get("pooler_depth", 2))
    traj_num_heads = int(cfgs_traj.get("num_heads", 16))
    traj_mlp_ratio = float(cfgs_traj.get("mlp_ratio", 4.0))

    dataset_type = sections.data.dataset_type
    dataset_paths = sections.data.dataset_paths
    datasets_weights = sections.data.datasets_weights
    dataset_fpcs = sections.data.dataset_fpcs
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = sections.data.batch_size
    tubelet_size = sections.data.tubelet_size
    fps = sections.data.fps
    crop_size = sections.data.crop_size
    patch_size = sections.data.patch_size
    pin_mem = sections.data.pin_mem
    num_workers = sections.data.num_workers
    persistent_workers = sections.data.persistent_workers
    deterministic_loader = sections.data.deterministic_loader
    prefetch_factor = sections.data.prefetch_factor
    decord_num_threads = sections.data.decord_num_threads
    ego_cache_size = sections.data.ego_cache_size
    trajectory_horizon = sections.data.temporal_recipe.future_horizon
    trajectory_dt = sections.data.temporal_recipe.future_dt
    trajectory_history_horizon = sections.data.temporal_recipe.history_horizon
    trajectory_history_dt = sections.data.temporal_recipe.history_dt
    dataset_names = sections.data.dataset_names
    max_resample_attempts = sections.data.max_resample_attempts

    cfgs_data_aug = sections.data_aug
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    loss_exp = sections.loss.loss_exp
    lambda_traj_start = sections.loss.lambda_traj_start
    lambda_traj_end = sections.loss.lambda_traj_end
    lambda_traj_schedule = sections.loss.lambda_traj_schedule
    lambda_traj_ramp_epochs = sections.loss.lambda_traj_ramp_epochs or sections.optimization.num_epochs
    traj_loss_weights = TrajectoryLossWeights.from_cfg(sections.loss.raw_cfg)

    is_anneal = sections.optimization.is_anneal
    anneal_ckpt = sections.optimization.anneal_ckpt
    resume_anneal = sections.optimization.resume_anneal
    ipe = sections.optimization.ipe
    ipe_scale = sections.optimization.ipe_scale
    wd = sections.optimization.wd
    final_wd = sections.optimization.final_wd
    num_epochs = sections.optimization.num_epochs
    warmup = sections.optimization.warmup
    start_lr = sections.optimization.start_lr
    lr = sections.optimization.lr
    final_lr = sections.optimization.final_lr
    ema = sections.optimization.ema
    betas = sections.optimization.betas
    eps = sections.optimization.eps
    encoder_lr_scale = sections.optimization.encoder_lr_scale
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        local_cuda_index = _resolve_local_cuda_index()
        device = torch.device(f"cuda:{local_cuda_index}")
        torch.cuda.set_device(device)

    if device.type == "cpu" and mixed_precision:
        logger.warning("Mixed precision requested but CUDA is unavailable; falling back to float32 training.")
        mixed_precision = False
        dtype = torch.float32

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = make_csv_logger(log_file)

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- init trajectory head
    embed_dim = encoder.backbone.embed_dim
    trajectory_head = init_trajectory_head(
        embed_dim=embed_dim,
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
        num_modes=num_modes,
        history_dim=history_dim,
        history_hidden_dim=history_hidden_dim,
        history_tokens=history_tokens,
        num_heads=traj_num_heads,
        mlp_ratio=traj_mlp_ratio,
        pooler_depth=traj_pooler_depth,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        device=device,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    if compile_model:
        logger.info("Compiling encoder, target_encoder, predictor, and trajectory_head.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
        trajectory_head.compile()

    # -- init mask collator (wrapped with DrivingMaskCollator)
    from src.datasets.driving_dataset import DrivingMaskCollator

    base_mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    mask_collator = DrivingMaskCollator(base_mask_collator)

    transform = make_transforms(
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        deterministic=deterministic_loader,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
        # Driving-specific params
        trajectory_horizon=trajectory_horizon,
        trajectory_dt=trajectory_dt,
        trajectory_history_horizon=trajectory_history_horizon,
        trajectory_history_dt=trajectory_history_dt,
        dataset_names=dataset_names,
        max_resample_attempts=max_resample_attempts,
        prefetch_factor=prefetch_factor,
        decord_num_threads=decord_num_threads,
        ego_cache_size=ego_cache_size,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler (with differential LR)
    optimizer, scaler, scheduler, wd_scheduler = init_drive_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        trajectory_head=trajectory_head,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        amp_dtype=dtype,
        betas=betas,
        eps=eps,
        encoder_lr_scale=encoder_lr_scale,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        logger.info(
            "Wrapping models with DDP (predictor_find_unused_parameters=%s)",
            predictor_find_unused_parameters,
        )
        ddp_kwargs = {
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }
        encoder = DistributedDataParallel(encoder, static_graph=True, **ddp_kwargs)
        predictor = DistributedDataParallel(
            predictor,
            static_graph=not predictor_find_unused_parameters,
            find_unused_parameters=predictor_find_unused_parameters,
            **ddp_kwargs,
        )
        trajectory_head = DistributedDataParallel(trajectory_head, static_graph=True, **ddp_kwargs)
    else:
        logger.info("Distributed process group not initialized; running in single-process mode.")
    for p in target_encoder.parameters():
        p.requires_grad = False

    def predict_traj_distribution(features, ego_history):
        model = trajectory_head.module if isinstance(trajectory_head, DistributedDataParallel) else trajectory_head
        return model.predict_distribution(features, ego_history=ego_history)

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        (
            encoder,
            predictor,
            target_encoder,
            trajectory_head,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            trajectory_head=trajectory_head,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch, path, meters):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "trajectory_head": trajectory_head.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": meters.loss.avg,
            "loss_jepa": meters.loss_jepa.avg,
            "loss_traj": meters.loss_traj.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        loader, epoch_meters = run_training_epoch(
            epoch=epoch,
            ipe=ipe,
            batch_size=batch_size,
            dataset_fpcs=dataset_fpcs,
            loader=loader,
            unsupervised_loader=unsupervised_loader,
            unsupervised_sampler=unsupervised_sampler,
            device=device,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            predict_traj_distribution=predict_traj_distribution,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            mixed_precision=mixed_precision,
            dtype=dtype,
            loss_exp=loss_exp,
            trajectory_dt=trajectory_dt,
            traj_loss_weights=traj_loss_weights,
            lambda_traj_start=lambda_traj_start,
            lambda_traj_end=lambda_traj_end,
            lambda_traj_schedule=lambda_traj_schedule,
            lambda_traj_ramp_epochs=lambda_traj_ramp_epochs,
            sync_gc=sync_gc,
            csv_logger=csv_logger,
            logger=logger,
            log_freq=log_freq,
        )
        log_epoch_summary(logger, epoch_meters)
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path, epoch_meters)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path, epoch_meters)
