# CLAUDE.md — V-JEPA2-Drive Project Context

## Project Overview

This repo is **V-JEPA2** (Meta FAIR's self-supervised video model) extended with **V-JEPA2-Drive** — a joint self-supervised + trajectory prediction training system for end-to-end autonomous driving. The core idea: trajectory prediction is part of the self-supervised learning objective, NOT a frozen-encoder + supervised-head approach.

### Philosophy (Tesla-style E2E)
- Encoder is **jointly trained** with both JEPA loss and trajectory loss
- `L_total = L_jepa + λ * L_traj` — both backpropagate through encoder
- JEPA loss acts as regularizer (prevents collapse to trajectory-only features)
- Cost functions shape feasible trajectories — started with minADE, will add comfort/smoothness
- Leverage massive driving data instead of control-specific parameters
- Future: online learning from watching human driver

### Architecture Diagram
```
Driving Video → Encoder(θ) → z_full (single forward pass, no masking)
                                |
                   +------------+------------+
                   |                         |
             apply_masks(z_full)       TrajectoryHead(ψ)
                   ↓                         ↓
             Predictor(φ)              pred_waypoints
                   ↓                         ↓
             L_jepa = L1(z_pred,       L_traj = minADE(
               sg(z_target))             waypoints, gt_ego)
                   |                         |
                   +------------+------------+
                                |
                   L_total = L_jepa + λ * L_traj
                   Update θ, φ, ψ jointly (encoder NOT frozen)
```

Key: Single encoder forward pass (Option B) — encoder runs once on unmasked video, post-hoc masking for JEPA predictor.

## Repo Structure

### Original V-JEPA2 (Meta)
```
src/models/vision_transformer.py   — ViT encoder (L/H/g variants), 3D patch embed, RoPE
src/models/predictor.py            — Masked latent prediction predictor
src/models/ac_predictor.py         — Action-conditioned predictor (V-JEPA2-AC, not used by us)
src/models/attentive_pooler.py     — AttentivePooler + AttentiveClassifier
src/models/utils/modules.py        — Block, Attention, RoPEAttention, CrossAttention
src/datasets/video_dataset.py      — VideoDataset with Decord loading
src/datasets/data_manager.py       — Dataset dispatch (imagenet/videodataset/drivingvideodataset)
src/masks/multiseq_multiblock3d.py — MaskCollator + _MaskGenerator
src/masks/utils.py                 — apply_masks(x, masks)
src/utils/wrappers.py              — MultiSeqWrapper, PredictorMultiSeqWrapper
src/utils/schedulers.py            — WarmupCosineSchedule, CosineWDSchedule
app/vjepa/train.py                 — Original V-JEPA2 training loop
app/vjepa/transforms.py            — Video augmentation pipeline
app/vjepa/utils.py                 — init_video_model, init_opt, load_checkpoint
app/scaffold.py                    — App dispatch: importlib.import_module(f"app.{app}.train")
app/main.py                        — Entry point (multi-GPU spawn)
```

### V-JEPA2-Drive (our additions)
```
src/models/trajectory_head.py      — TrajectoryHead: AttentivePooler → waypoint_proj
src/datasets/ego_motion_utils.py   — Multi-dataset ego-motion loading + waypoint computation
src/datasets/driving_dataset.py    — DrivingVideoDataset + DrivingMaskCollator
app/vjepa_drive/__init__.py        — Module init
app/vjepa_drive/transforms.py      — Driving transforms (NO horizontal flip)
app/vjepa_drive/utils.py           — init_trajectory_head, init_drive_opt, load_checkpoint
app/vjepa_drive/train.py           — Joint training loop (single forward pass)
configs/train/vitg16/driving-joint-256px-16f.yaml — Training config
PLAN_VJEPA2_E2E_AD.md             — Full architectural plan (v2)
```

### NVIDIA Physical AI AV Devkit (reference)
```
physical_ai_av/src/physical_ai_av/dataset.py     — HF dataset interface
physical_ai_av/src/physical_ai_av/egomotion.py    — EgomotionState dataclass
physical_ai_av/src/physical_ai_av/utils/tf.py     — TransformTree, Transformable
physical_ai_av/src/physical_ai_av/utils/interpolation.py — RigidTransformInterpolator
```

## Key Technical Details

### Encoder
- ViT-g (1B params): embed_dim=1408, depth=40, num_heads=22, 3D-RoPE
- Wrapped in `MultiSeqWrapper` — forward takes lists: `encoder(clips)` or `encoder(clips, masks)`
- Without masks: returns `[backbone(xi) for xi in clips]`
- With masks: returns nested list `[[backbone(xi, masks=mij) for mij in mi] for xi, mi in zip(x, masks)]`

### TrajectoryHead
- Uses `AttentivePooler(num_queries=num_waypoints)` + `nn.Linear(embed_dim, waypoint_dim)`
- Input: `[B, N, D]` encoder features → Output: `[B, num_waypoints, 2]` (x, y) waypoints
- ~80 lines, lightweight by design

### Training Loop (app/vjepa_drive/train.py)
- **Single forward pass**: `z_full = encoder(clips)` (no masks)
- **JEPA branch**: `z_masked = apply_masks(z_full, masks_enc)` → predictor → L_jepa
- **Trajectory branch**: `trajectory_head(z_full)` → minADE loss vs GT waypoints
- **Differential LR**: encoder/predictor at 0.1x, trajectory head at 1.0x
- **EMA target encoder**: momentum update, provides targets for JEPA loss

### DrivingVideoDataset CSV Format
```
video_path ego_motion_path clip_start_time clip_end_time [dataset_name]
```
Space-delimited. dataset_name is optional (nuscenes/argoverse/waymo/generic).

### DrivingMaskCollator
Wraps `MaskCollator`, returns 4-tuples instead of 3-tuples:
`(collated_batch, masks_enc, masks_pred, gt_trajectories)`

### Ego-Motion Utils
- `get_ego_motion_loader(name)` → loader for nuScenes/Argoverse/Waymo/generic
- `compute_future_waypoints_from_poses(timestamps, positions, headings, ref_timestamp, horizon, dt)` → `[horizon, 2]` tensor in ego frame
- Interpolation: linear for position, angle-wrapped linear for heading
- Returns `None` if future timestamps exceed data range (dataset retries)

## Implementation Status

### Sprint 1 (COMPLETE) — Data Pipeline + Training Infrastructure
- [x] TrajectoryHead module
- [x] Ego-motion utilities (multi-dataset)
- [x] DrivingVideoDataset + DrivingMaskCollator
- [x] Joint training loop (single forward pass)
- [x] Driving transforms (no horizontal flip)
- [x] Training config
- [x] data_manager.py registration

### Sprint 2 (TODO) — Joint Training on Real Data
- [ ] Prepare nuScenes data CSVs (video paths + ego-motion extraction)
- [ ] Train on nuScenes with pretrained V-JEPA2 ViT-L (smaller, faster iteration)
- [ ] Validate: trajectory loss decreasing, JEPA loss stable
- [ ] Implement min_ade_loss unit tests

### Sprint 3 (TODO) — Scale Up
- [ ] Add Argoverse v2 and Waymo dataset loaders
- [ ] Scale to ViT-g encoder
- [ ] Add FDE and smoothness cost functions
- [ ] Run ablation experiments

### Sprint 4 (TODO) — Multi-Camera + Analysis
- [ ] Camera ID embeddings for multi-camera
- [ ] Extended trajectory horizon
- [ ] Frozen vs joint training comparison
- [ ] Representation analysis

## Running

```bash
# Training (multi-GPU)
python app/main.py --fname configs/train/vitg16/driving-joint-256px-16f.yaml

# Single GPU debug
python app/main.py --fname configs/train/vitg16/driving-joint-256px-16f.yaml --devices cuda:0 --debugmode True
```

## Important Conventions

- **No horizontal flip** for driving data (flipping inverts trajectory semantics)
- **Encoder is NOT frozen** — both losses backpropagate through it
- scaffold.py dispatches apps via `importlib.import_module(f"app.{app}.train").main()`
- MaskCollator groups batches by frames-per-clip (fpc) — iteration is `for fpc_sample in sample`
- `apply_masks(x, masks)` gathers tokens at mask indices: `[B, N, D]` + `[B, K]` → `[B, K, D]`
- Checkpoints save encoder, predictor, target_encoder, trajectory_head, optimizer, scaler, epoch
- Old V-JEPA2 checkpoints (without trajectory_head) are handled gracefully — head starts random

## Dependencies

See `requirements.txt`. Key: PyTorch, torchvision, decord, numpy, pandas, PyYAML.
Python >= 3.10.
