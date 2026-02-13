# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.utils import init_video_model, load_checkpoint as _load_checkpoint
from src.models.trajectory_head import TrajectoryHead
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _strip_module_prefix(state_dict):
    """Remove a leading 'module.' (DDP) prefix from checkpoint keys when present."""
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    needs_strip = any(k.startswith("module.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict

    cleaned = state_dict.__class__() if isinstance(state_dict, OrderedDict) else OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1)
        cleaned[new_key] = value
    return cleaned


def init_trajectory_head(
    embed_dim,
    num_waypoints=12,
    waypoint_dim=2,
    num_heads=16,
    mlp_ratio=4.0,
    pooler_depth=2,
    device=torch.device("cpu"),
    use_activation_checkpointing=False,
):
    """Initialize the trajectory head."""
    trajectory_head = TrajectoryHead(
        embed_dim=embed_dim,
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        pooler_depth=pooler_depth,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    trajectory_head.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"TrajectoryHead number of parameters: {count_parameters(trajectory_head)}")
    return trajectory_head


def init_drive_opt(
    is_anneal,
    encoder,
    predictor,
    trajectory_head,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    encoder_lr_scale=0.1,
):
    """Initialize optimizer with differential learning rates.

    The encoder and predictor (pretrained) use a lower learning rate,
    while the trajectory head (randomly initialized) uses the full learning rate.
    """
    param_groups = [
        # Encoder weights (lower LR for pretrained params)
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": encoder_lr_scale,
        },
        # Predictor weights (lower LR for pretrained params)
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": encoder_lr_scale,
        },
        # Encoder biases
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": encoder_lr_scale,
        },
        # Predictor biases
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": encoder_lr_scale,
        },
        # Trajectory head weights (full LR for new params)
        {
            "params": (p for n, p in trajectory_head.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        # Trajectory head biases
        {
            "params": (p for n, p in trajectory_head.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    trajectory_head,
    opt,
    scaler,
    is_anneal=False,
):
    """Load checkpoint, handling the case where trajectory_head may not exist in old checkpoints."""
    from src.utils.checkpoint_loader import robust_checkpoint_loader
    def _unwrap(module):
        return module.module if isinstance(module, DistributedDataParallel) else module

    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    # Load encoder
    pretrained_dict = _strip_module_prefix(checkpoint["encoder"])
    msg = _unwrap(encoder).load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # Load predictor
    pretrained_dict = _strip_module_prefix(checkpoint["predictor"])
    msg = _unwrap(predictor).load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # Load target encoder
    if target_encoder is not None:
        pretrained_dict = _strip_module_prefix(checkpoint["target_encoder"])
        msg = _unwrap(target_encoder).load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    # Load trajectory head (may not exist in V-JEPA2 pretrained checkpoints)
    if "trajectory_head" in checkpoint:
        msg = _unwrap(trajectory_head).load_state_dict(_strip_module_prefix(checkpoint["trajectory_head"]))
        logger.info(f"loaded trajectory head from epoch {epoch} with msg: {msg}")
    else:
        logger.info("No trajectory_head in checkpoint, using randomly initialized weights")

    # Load optimizer
    try:
        opt.load_state_dict(checkpoint["opt"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")
    except Exception as e:
        logger.warning(f"Could not load optimizer state (expected when loading V-JEPA2 pretrained): {e}")

    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        trajectory_head,
        opt,
        scaler,
        epoch,
    )
