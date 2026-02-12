# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_drive.transforms import make_transforms
from app.vjepa_drive.utils import init_drive_opt, init_trajectory_head, load_checkpoint
from app.vjepa.utils import init_video_model
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def min_ade_loss(pred_waypoints, gt_waypoints):
    """Compute minimum Average Displacement Error.

    Args:
        pred_waypoints: [B, T, 2] predicted (x, y) waypoints in ego frame
        gt_waypoints:   [B, T, 2] ground truth from ego-motion data
    Returns:
        scalar loss
    """
    displacement = torch.norm(pred_waypoints - gt_waypoints, dim=-1)  # [B, T]
    ade = displacement.mean(dim=-1)  # [B]
    return ade.mean()


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- TRAJECTORY HEAD
    cfgs_traj = cfgs_model.get("trajectory_head", {})
    num_waypoints = cfgs_traj.get("num_waypoints", 12)
    waypoint_dim = cfgs_traj.get("waypoint_dim", 2)
    traj_pooler_depth = cfgs_traj.get("pooler_depth", 2)
    traj_num_heads = cfgs_traj.get("num_heads", 16)
    traj_mlp_ratio = cfgs_traj.get("mlp_ratio", 4.0)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "drivingvideodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    trajectory_horizon = cfgs_data.get("trajectory_horizon", 12)
    trajectory_dt = cfgs_data.get("trajectory_dt", 0.5)
    dataset_names = cfgs_data.get("dataset_names", None)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    lambda_traj = cfgs_loss.get("lambda_traj", 1.0)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (is_anneal and resume_preempt)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    encoder_lr_scale = cfgs_opt.get("encoder_lr_scale", 0.1)
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
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

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
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_jepa"),
        ("%.5f", "loss_traj"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

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
        num_heads=traj_num_heads,
        mlp_ratio=traj_mlp_ratio,
        pooler_depth=traj_pooler_depth,
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
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
        # Driving-specific params
        trajectory_horizon=trajectory_horizon,
        trajectory_dt=trajectory_dt,
        dataset_names=dataset_names,
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
        betas=betas,
        eps=eps,
        encoder_lr_scale=encoder_lr_scale,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    trajectory_head = DistributedDataParallel(trajectory_head, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
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

    def save_checkpoint(epoch, path):
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
            "loss": loss_meter.avg,
            "loss_jepa": loss_jepa_meter.avg,
            "loss_traj": loss_traj_meter.avg,
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

        loss_meter = AverageMeter()
        loss_jepa_meter = AverageMeter()
        loss_traj_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

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
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred, all_trajectories = [], [], [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred, gt_trajectories = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                    all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                    all_trajectories += [gt_trajectories.to(device, non_blocking=True)]
                return all_clips, all_masks_enc, all_masks_pred, all_trajectories

            clips, masks_enc, masks_pred, gt_trajectories = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # ============================================================
                # OPTION B: Single encoder forward pass, post-hoc masking
                # ============================================================

                # Step 1: Target encoder forward (no grad) on full (unmasked) video
                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                        return h

                # Step 2: Encoder forward on FULL (unmasked) video - SINGLE PASS
                # This gives us features for both JEPA (after masking) and trajectory
                def forward_encoder_full(c):
                    return encoder(c)  # no masks -> returns list of [B, N, D]

                # Step 3: Apply masking post-hoc for JEPA branch, then predict
                def forward_predictor_from_full(z_full):
                    # z_full is a list of [B, N, D] tensors (one per clip length)
                    # Apply masks to get context tokens, then run predictor
                    z_masked = []
                    for zi, mi in zip(z_full, masks_enc):
                        z_masked_i = []
                        for mij in mi:
                            z_masked_i.append(apply_masks(zi, mij))
                        z_masked.append(z_masked_i)
                    # Run predictor on masked tokens
                    z_pred = predictor(z_masked, masks_enc, masks_pred)
                    return z_pred

                # Step 4: JEPA loss
                def loss_fn(z, h):
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    loss /= n
                    return loss

                # Step 5: Trajectory prediction from full features
                def forward_trajectory(z_full):
                    # z_full is list of [B, N, D]; concatenate across clip lengths
                    # then predict waypoints
                    traj_losses = []
                    for zi, gt_i in zip(z_full, gt_trajectories):
                        pred_wp = trajectory_head(zi)  # [B, num_waypoints, 2]
                        traj_losses.append(min_ade_loss(pred_wp, gt_i))
                    return sum(traj_losses) / len(traj_losses)

                # Execute forward pass
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # Target (no grad)
                    h = forward_target(clips)

                    # Single encoder forward on full (unmasked) video
                    z_full = forward_encoder_full(clips)

                    # JEPA branch: post-hoc masking + predictor
                    z_pred = forward_predictor_from_full(z_full)
                    loss_jepa = loss_fn(z_pred, h)

                    # Trajectory branch: from full features
                    loss_traj = forward_trajectory(z_full)

                    # Combined loss
                    loss = loss_jepa + lambda_traj * loss_traj

                # Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_traj),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                loss_jepa_val,
                loss_traj_val,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            loss_jepa_meter.update(loss_jepa_val)
            loss_traj_meter.update(loss_traj_val)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, loss_jepa_val, loss_traj_val,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[jepa: %.3f] [traj: %.3f] "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            loss_jepa_meter.avg,
                            loss_traj_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f [jepa: %.3f] [traj: %.3f]" % (
            loss_meter.avg, loss_jepa_meter.avg, loss_traj_meter.avg))
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
