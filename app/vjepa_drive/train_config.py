# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.vjepa_drive.temporal_recipe import TemporalRecipe


@dataclass(frozen=True)
class MetaConfig:
    folder: str
    load_model: bool
    read_checkpoint: str | None
    seed: int
    save_every_freq: int
    skip_batches: int
    use_sdpa: bool
    sync_gc: bool
    dtype: torch.dtype
    mixed_precision: bool


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    pred_depth: int
    pred_num_heads: int | None
    pred_embed_dim: int
    uniform_power: bool
    use_mask_tokens: bool
    zero_init_mask_tokens: bool
    use_rope: bool
    use_silu: bool
    use_pred_silu: bool
    wide_silu: bool
    compile_model: bool
    use_activation_checkpointing: bool
    predictor_find_unused_parameters: bool
    traj_cfg: dict


@dataclass(frozen=True)
class DataConfig:
    dataset_type: str
    dataset_paths: list[str]
    datasets_weights: list[float] | None
    dataset_fpcs: list[int]
    batch_size: int
    tubelet_size: int
    fps: float
    crop_size: int
    patch_size: int
    pin_mem: bool
    num_workers: int
    persistent_workers: bool
    deterministic_loader: bool
    prefetch_factor: int
    decord_num_threads: int
    ego_cache_size: int
    dataset_names: list[str] | None
    max_resample_attempts: int
    temporal_recipe: TemporalRecipe


@dataclass(frozen=True)
class LossConfig:
    loss_exp: float
    lambda_traj_start: float
    lambda_traj_end: float
    lambda_traj_schedule: str
    lambda_traj_ramp_epochs: float | None
    raw_cfg: dict


@dataclass(frozen=True)
class OptimizationConfig:
    is_anneal: bool
    anneal_ckpt: str | None
    resume_anneal: bool
    ipe: int | None
    ipe_scale: float
    wd: float
    final_wd: float
    num_epochs: int
    warmup: int
    start_lr: float
    lr: float
    final_lr: float
    ema: tuple[float, float]
    betas: tuple[float, float]
    eps: float
    encoder_lr_scale: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class TrainSections:
    meta: MetaConfig
    model: ModelConfig
    data: DataConfig
    data_aug: dict
    loss: LossConfig
    optimization: OptimizationConfig
    mask_cfg: list[dict]


def _resolve_dtype(dtype_name: str) -> tuple[torch.dtype, bool]:
    dtype_name = dtype_name.lower()
    if dtype_name == "bfloat16":
        return torch.bfloat16, True
    if dtype_name == "float16":
        return torch.float16, True
    return torch.float32, False


def parse_train_sections(args: dict, resume_preempt: bool = False) -> TrainSections:
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    cfgs_model = args.get("model")
    cfgs_data = args.get("data")
    cfgs_data_aug = args.get("data_aug")
    cfgs_loss = args.get("loss")
    cfgs_opt = args.get("optimization")
    cfgs_mask = args.get("mask")

    dtype, mixed_precision = _resolve_dtype(cfgs_meta.get("dtype"))
    meta = MetaConfig(
        folder=folder,
        load_model=bool(cfgs_meta.get("load_checkpoint") or resume_preempt),
        read_checkpoint=cfgs_meta.get("read_checkpoint", None),
        seed=int(cfgs_meta.get("seed", 0)),
        save_every_freq=int(cfgs_meta.get("save_every_freq", -1)),
        skip_batches=int(cfgs_meta.get("skip_batches", -1)),
        use_sdpa=bool(cfgs_meta.get("use_sdpa", False)),
        sync_gc=bool(cfgs_meta.get("sync_gc", False)),
        dtype=dtype,
        mixed_precision=mixed_precision,
    )

    model = ModelConfig(
        model_name=cfgs_model.get("model_name"),
        pred_depth=int(cfgs_model.get("pred_depth")),
        pred_num_heads=cfgs_model.get("pred_num_heads", None),
        pred_embed_dim=int(cfgs_model.get("pred_embed_dim")),
        uniform_power=bool(cfgs_model.get("uniform_power", False)),
        use_mask_tokens=bool(cfgs_model.get("use_mask_tokens", False)),
        zero_init_mask_tokens=bool(cfgs_model.get("zero_init_mask_tokens", True)),
        use_rope=bool(cfgs_model.get("use_rope", False)),
        use_silu=bool(cfgs_model.get("use_silu", False)),
        use_pred_silu=bool(cfgs_model.get("use_pred_silu", False)),
        wide_silu=bool(cfgs_model.get("wide_silu", True)),
        compile_model=bool(cfgs_model.get("compile_model", False)),
        use_activation_checkpointing=bool(cfgs_model.get("use_activation_checkpointing", False)),
        predictor_find_unused_parameters=bool(cfgs_model.get("predictor_find_unused_parameters", True)),
        traj_cfg=cfgs_model.get("trajectory_head", {}),
    )

    data = DataConfig(
        dataset_type=cfgs_data.get("dataset_type", "drivingvideodataset"),
        dataset_paths=cfgs_data.get("datasets", []),
        datasets_weights=cfgs_data.get("datasets_weights"),
        dataset_fpcs=cfgs_data.get("dataset_fpcs"),
        batch_size=int(cfgs_data.get("batch_size")),
        tubelet_size=int(cfgs_data.get("tubelet_size")),
        fps=float(cfgs_data.get("fps")),
        crop_size=int(cfgs_data.get("crop_size", 224)),
        patch_size=int(cfgs_data.get("patch_size")),
        pin_mem=bool(cfgs_data.get("pin_mem", False)),
        num_workers=int(cfgs_data.get("num_workers", 1)),
        persistent_workers=bool(cfgs_data.get("persistent_workers", True)),
        deterministic_loader=bool(cfgs_data.get("deterministic_loader", True)),
        prefetch_factor=max(1, int(cfgs_data.get("prefetch_factor", 2))),
        decord_num_threads=max(1, int(cfgs_data.get("decord_num_threads", 2))),
        ego_cache_size=max(0, int(cfgs_data.get("ego_cache_size", 512))),
        dataset_names=cfgs_data.get("dataset_names", None),
        max_resample_attempts=int(cfgs_data.get("max_resample_attempts", 64)),
        temporal_recipe=TemporalRecipe.from_data_cfg(cfgs_data),
    )

    lambda_traj_start = float(cfgs_loss.get("lambda_traj_start", cfgs_loss.get("lambda_traj", 1.0)))
    loss = LossConfig(
        loss_exp=float(cfgs_loss.get("loss_exp", 1.0)),
        lambda_traj_start=lambda_traj_start,
        lambda_traj_end=float(cfgs_loss.get("lambda_traj_end", lambda_traj_start)),
        lambda_traj_schedule=str(cfgs_loss.get("lambda_traj_schedule", "cosine")),
        lambda_traj_ramp_epochs=cfgs_loss.get("lambda_traj_ramp_epochs", None),
        raw_cfg=cfgs_loss,
    )

    is_anneal = bool(cfgs_opt.get("is_anneal", False))
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    optimization = OptimizationConfig(
        is_anneal=is_anneal,
        anneal_ckpt=anneal_ckpt,
        resume_anneal=bool(cfgs_opt.get("resume_anneal", False) or (is_anneal and resume_preempt)),
        ipe=cfgs_opt.get("ipe", None),
        ipe_scale=float(cfgs_opt.get("ipe_scale", 1.0)),
        wd=float(cfgs_opt.get("weight_decay")),
        final_wd=float(cfgs_opt.get("final_weight_decay")),
        num_epochs=int(cfgs_opt.get("epochs")),
        warmup=int(cfgs_opt.get("warmup")),
        start_lr=float(cfgs_opt.get("start_lr")),
        lr=float(cfgs_opt.get("lr")),
        final_lr=float(cfgs_opt.get("final_lr")),
        ema=tuple(cfgs_opt.get("ema")),
        betas=tuple(cfgs_opt.get("betas", (0.9, 0.999))),
        eps=float(cfgs_opt.get("eps", 1.0e-8)),
        encoder_lr_scale=float(cfgs_opt.get("encoder_lr_scale", 0.1)),
        gradient_accumulation_steps=max(1, int(cfgs_opt.get("gradient_accumulation_steps", 1))),
    )

    return TrainSections(
        meta=meta,
        model=model,
        data=data,
        data_aug=cfgs_data_aug,
        loss=loss,
        optimization=optimization,
        mask_cfg=cfgs_mask,
    )
