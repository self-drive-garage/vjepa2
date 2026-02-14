#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TensorRT exporter for the V-JEPA2 driving waypoint model.

This utility loads a trained checkpoint (encoder + trajectory head),
stitches them into a single forward graph (video -> waypoints),
exports ONNX, and optionally builds a TensorRT engine.  It is written
to run inside NVIDIA's TensorRT docker images (e.g.
`nvcr.io/nvidia/tensorrt:24.02-py3`) but also works anywhere the
TensorRT Python bindings are available.

Example usage from the repo root:

    docker run --rm -it --gpus all \\
      -v $PWD:/workspace -w /workspace \\
      nvcr.io/nvidia/tensorrt:24.02-py3 \\
      python scripts/export_trt_waypoints.py \\
        --config configs/train/vitg16/driving-joint-256px-16f.yaml \\
        --checkpoint logs/vjepa_drive/phase1-*/checkpoints/latest.pt \\
        --output-dir artifacts/trt \\
        --precision fp16 \\
        --opt-batch 2 --max-batch 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from app.vjepa.utils import init_video_model
from app.vjepa_drive.inference import DrivingInferenceModel
from app.vjepa_drive.runtime_safety import SafetyEnvelopeConfig
from app.vjepa_drive.utils import _strip_module_prefix, init_trajectory_head

LOGGER = logging.getLogger("export_trt_waypoints")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    LOGGER.warning("CUDA not available; exporting on CPU (TensorRT build will still require GPU).")
    return torch.device("cpu")


class DrivingWaypointModule(torch.nn.Module):
    """Export wrapper supporting best-mode, safe, or full-distribution outputs."""

    def __init__(
        self,
        inference_model: DrivingInferenceModel,
        export_mode: str,
        include_ego_history: bool,
    ):
        super().__init__()
        self.inference_model = inference_model
        self.export_mode = export_mode
        self.include_ego_history = include_ego_history

    def forward(self, video: torch.Tensor, ego_history: torch.Tensor | None = None):
        if self.include_ego_history and ego_history is None:
            raise ValueError("ego_history input is required when include_ego_history=True")
        if self.export_mode == "distribution":
            dist = self.inference_model.predict_distribution(video, ego_history=ego_history)
            return dist.means, dist.log_stds, dist.mode_logits
        apply_safety = self.export_mode == "safe"
        traj, _, _ = self.inference_model.predict(video, ego_history=ego_history, apply_safety=apply_safety)
        return traj


class TensorRTWaypointExporter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_config(Path(args.config))
        self.device = resolve_device()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_index = args.sequence_index
        dataset_fpcs = self.cfg["data"]["dataset_fpcs"]
        if not (0 <= self.sequence_index < len(dataset_fpcs)):
            raise ValueError(
                f"sequence-index {self.sequence_index} is invalid for dataset_fpcs={dataset_fpcs}"
            )
        self.seq_frames = dataset_fpcs[self.sequence_index]
        self.frames = args.frames or self.seq_frames

        crop_size = self.cfg["data"].get("crop_size", 224)
        self.height = args.height or crop_size
        self.width = args.width or crop_size

        if self.frames % self.cfg["data"]["tubelet_size"] != 0:
            raise ValueError(
                f"frames ({self.frames}) must be divisible by tubelet_size "
                f"{self.cfg['data']['tubelet_size']}"
            )

        if not (self.args.min_batch <= self.args.opt_batch <= self.args.max_batch):
            raise ValueError("Must satisfy min_batch <= opt_batch <= max_batch.")

        self.export_mode = args.export_mode
        self.include_ego_history = bool(args.include_ego_history or (self.export_mode == "safe"))
        if self.export_mode == "safe" and not self.include_ego_history:
            raise ValueError("Safe export mode requires ego-history input.")
        self.history_tokens = int(
            self.cfg.get("model", {})
            .get("trajectory_head", {})
            .get("history_tokens", self.cfg.get("data", {}).get("trajectory_history_horizon", 8))
        )
        self.trajectory_dt = float(self.cfg.get("data", {}).get("trajectory_dt", 0.5))
        self.safety_cfg = SafetyEnvelopeConfig.from_cfg(self.cfg.get("runtime_safety", {}))

    def run(self) -> None:
        LOGGER.info("Building PyTorch module...")
        module = self._build_module().eval()
        module.to(self.device)

        dtype = torch.float16 if self.args.precision == "fp16" else torch.float32
        module.to(dtype=dtype)

        onnx_path = self.output_dir / (
            self.args.onnx_name or f"vjepa_drive_waypoints_{self.args.precision}.onnx"
        )
        LOGGER.info("Exporting ONNX -> %s", onnx_path)
        self._export_onnx(module, dtype, onnx_path)

        if self.args.skip_engine:
            LOGGER.info("Skipping TensorRT engine build (per flag).")
            return

        engine_path = self.output_dir / (
            self.args.engine_name or f"vjepa_drive_waypoints_{self.args.precision}.plan"
        )
        LOGGER.info("Building TensorRT engine -> %s", engine_path)
        self._build_trt_engine(onnx_path, engine_path)

    def _build_module(self) -> DrivingWaypointModule:
        data_cfg = self.cfg["data"]
        model_cfg = self.cfg["model"]
        meta_cfg = self.cfg.get("meta", {})

        encoder_wrapper, _ = init_video_model(
            device=self.device,
            patch_size=data_cfg["patch_size"],
            max_num_frames=max(data_cfg["dataset_fpcs"]),
            tubelet_size=data_cfg.get("tubelet_size", 2),
            model_name=model_cfg["model_name"],
            crop_size=data_cfg.get("crop_size", 224),
            pred_depth=model_cfg.get("pred_depth", 12),
            pred_num_heads=model_cfg.get("pred_num_heads"),
            pred_embed_dim=model_cfg.get("pred_embed_dim", 384),
            uniform_power=model_cfg.get("uniform_power", False),
            use_mask_tokens=model_cfg.get("use_mask_tokens", False),
            num_mask_tokens=model_cfg.get("num_mask_tokens", 2),
            zero_init_mask_tokens=model_cfg.get("zero_init_mask_tokens", True),
            use_sdpa=meta_cfg.get("use_sdpa", False),
            use_rope=model_cfg.get("use_rope", False),
            use_silu=model_cfg.get("use_silu", False),
            use_pred_silu=model_cfg.get("use_pred_silu", False),
            wide_silu=model_cfg.get("wide_silu", True),
            use_activation_checkpointing=False,
        )

        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        encoder_state = _strip_module_prefix(checkpoint["encoder"])
        encoder_wrapper.load_state_dict(encoder_state, strict=True)
        encoder = encoder_wrapper.backbone  # VisionTransformer

        traj_cfg = model_cfg.get("trajectory_head", {})
        trajectory_head = init_trajectory_head(
            embed_dim=encoder.embed_dim,
            num_waypoints=traj_cfg.get("num_waypoints", data_cfg.get("trajectory_horizon", 12)),
            waypoint_dim=traj_cfg.get("waypoint_dim", 2),
            num_modes=traj_cfg.get("num_modes", 4),
            history_dim=traj_cfg.get("history_dim", 2),
            history_hidden_dim=traj_cfg.get("history_hidden_dim", None),
            history_tokens=traj_cfg.get(
                "history_tokens",
                self.cfg.get("data", {}).get("trajectory_history_horizon", 8),
            ),
            num_heads=traj_cfg.get("num_heads", 16),
            mlp_ratio=traj_cfg.get("mlp_ratio", 4.0),
            pooler_depth=traj_cfg.get("pooler_depth", 2),
            min_log_std=traj_cfg.get("min_log_std", -4.0),
            max_log_std=traj_cfg.get("max_log_std", 1.0),
            device=self.device,
            use_activation_checkpointing=False,
        )
        if "trajectory_head" in checkpoint:
            traj_state = _strip_module_prefix(checkpoint["trajectory_head"])
            msg = trajectory_head.load_state_dict(traj_state, strict=False)
            LOGGER.info("Loaded trajectory_head with msg: %s", msg)
        else:
            LOGGER.warning("No trajectory_head weights in checkpoint; using randomly initialized head.")

        inference_model = DrivingInferenceModel(
            encoder=encoder,
            trajectory_head=trajectory_head,
            trajectory_dt=self.trajectory_dt,
            safety_cfg=self.safety_cfg if self.export_mode == "safe" else None,
        )
        return DrivingWaypointModule(
            inference_model=inference_model,
            export_mode=self.export_mode,
            include_ego_history=self.include_ego_history,
        )

    def _export_onnx(self, module: DrivingWaypointModule, dtype: torch.dtype, onnx_path: Path) -> None:
        dummy = torch.randn(
            self.args.opt_batch,
            3,
            self.frames,
            self.height,
            self.width,
            device=self.device,
            dtype=dtype,
        )
        inputs = [dummy]
        input_names = ["video"]
        dynamic_axes = {"video": {0: "batch"}}

        if self.include_ego_history:
            dummy_hist = torch.randn(
                self.args.opt_batch,
                self.history_tokens,
                2,
                device=self.device,
                dtype=dtype,
            )
            inputs.append(dummy_hist)
            input_names.append("ego_history")
            dynamic_axes["ego_history"] = {0: "batch", 1: "history_steps"}

        if self.export_mode == "distribution":
            output_names = ["traj_means", "traj_log_stds", "traj_mode_logits"]
            dynamic_axes["traj_means"] = {0: "batch"}
            dynamic_axes["traj_log_stds"] = {0: "batch"}
            dynamic_axes["traj_mode_logits"] = {0: "batch"}
        else:
            output_names = ["waypoints"]
            dynamic_axes["waypoints"] = {0: "batch"}

        torch.onnx.export(
            module,
            tuple(inputs) if len(inputs) > 1 else inputs[0],
            str(onnx_path),
            export_params=True,
            opset_version=self.args.opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    def _build_trt_engine(self, onnx_path: Path, engine_path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT python bindings not available. Install TensorRT or run inside "
                "the NVIDIA TensorRT docker image."
            ) from exc

        logger = trt.Logger(trt.Logger.INFO)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(logger) as builder, builder.create_network(flag) as network, trt.OnnxParser(
            network, logger
        ) as parser:
            with open(onnx_path, "rb") as handle:
                if not parser.parse(handle.read()):
                    error_msgs = "\n".join(
                        f"[{i}] {parser.get_error(i)}" for i in range(parser.num_errors)
                    )
                    raise RuntimeError(f"ONNX parsing failed:\n{error_msgs}")

            config = builder.create_builder_config()
            workspace_bytes = int(self.args.workspace_gb * (1024**3))
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

            profile = builder.create_optimization_profile()
            min_shape = (self.args.min_batch, 3, self.frames, self.height, self.width)
            opt_shape = (self.args.opt_batch, 3, self.frames, self.height, self.width)
            max_shape = (self.args.max_batch, 3, self.frames, self.height, self.width)
            profile.set_shape("video", min_shape, opt_shape, max_shape)
            if self.include_ego_history:
                min_hist = (self.args.min_batch, self.history_tokens, 2)
                opt_hist = (self.args.opt_batch, self.history_tokens, 2)
                max_hist = (self.args.max_batch, self.history_tokens, 2)
                profile.set_shape("ego_history", min_hist, opt_hist, max_hist)
            config.add_optimization_profile(profile)

            if self.args.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    LOGGER.info("Enabled FP16 builder flag.")
                else:
                    LOGGER.warning("FP16 not supported on this platform; falling back to FP32.")

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Failed to build TensorRT engine.")
            engine_path.write_bytes(bytes(serialized))
            LOGGER.info("TensorRT engine written to %s", engine_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V-JEPA2 driving model to TensorRT.")
    parser.add_argument("--config", required=True, help="Training config used for the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing encoder/trajectory_head.")
    parser.add_argument("--output-dir", default="artifacts/trt", help="Directory for ONNX/engine outputs.")
    parser.add_argument("--onnx-name", help="Optional override for the ONNX filename.")
    parser.add_argument("--engine-name", help="Optional override for the TensorRT engine filename.")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--frames", type=int, help="Override frames per clip (defaults to max dataset_fpcs).")
    parser.add_argument("--height", type=int, help="Override input height (defaults to crop_size).")
    parser.add_argument("--width", type=int, help="Override input width (defaults to crop_size).")
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size for TensorRT profile.")
    parser.add_argument("--opt-batch", type=int, default=2, help="Optimal batch size for TensorRT profile.")
    parser.add_argument("--max-batch", type=int, default=4, help="Maximum batch size for TensorRT profile.")
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT workspace size in GB.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--skip-engine", action="store_true", help="Only export ONNX; skip TensorRT build.")
    parser.add_argument(
        "--export-mode",
        choices=("best_mode", "safe", "distribution"),
        default="best_mode",
        help="`best_mode`: top-probability trajectory, `safe`: apply runtime safety envelope, `distribution`: export means/stds/logits.",
    )
    parser.add_argument(
        "--include-ego-history",
        action="store_true",
        help="Export a second input `ego_history` with shape [B, H, 2]. Required for safe mode.",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Index into dataset_fpcs for multi-clip configs (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    exporter = TensorRTWaypointExporter(parse_args())
    exporter.run()


if __name__ == "__main__":
    main()
