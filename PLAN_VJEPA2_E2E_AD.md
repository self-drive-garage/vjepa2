# V-JEPA2 for End-to-End Autonomous Driving: Architectural Plan (v2)

## Executive Summary

This plan adapts V-JEPA2 for end-to-end autonomous driving by making **trajectory prediction an intrinsic part of the self-supervised learning objective**. Unlike the standard approach of freezing a pretrained encoder and bolting on supervised planning heads, we **jointly train** the encoder with both the existing masked latent prediction loss AND a trajectory prediction loss. This means the encoder learns latent representations that are inherently useful for planning — not just video understanding.

This is philosophically closer to Tesla's end-to-end approach: the model learns to output feasible trajectories shaped by cost functions (displacement error, comfort, safety), leveraging massive open-source driving datasets (nuScenes, Argoverse v2, Waymo, NVIDIA AV) as self-supervision.

---

## 1. Core Idea: Joint Self-Supervised Learning

### 1.1 Current V-JEPA2 Training

```
Video → Encoder(θ) → Latent z
                         |
                    Predictor(φ) → Predicted z_hat
                         |
                    L_jepa = L1(z_hat, sg(z_target))    ← only loss
                         |
                    Update θ, φ
```

The encoder learns representations optimized purely for predicting masked latent features. These representations capture rich video semantics but are not explicitly shaped for any planning task.

### 1.2 Proposed: Joint JEPA + Trajectory Training

```
Driving Video → Encoder(θ) → Latent z
                                |
                    +-----------+-----------+
                    |                       |
              Predictor(φ)           TrajectoryHead(ψ)
              (masked latent          (waypoint prediction)
               prediction)
                    |                       |
              L_jepa = L1(z_hat,      L_traj = composite(
                sg(z_target))           ADE/FDE/NLL/calib/...
                    |                       |
                    +-----------+-----------+
                                |
                    L_total = L_jepa + λ * L_traj
                                |
                    Update θ, φ, ψ jointly
                           ^^^
                    ENCODER IS NOT FROZEN
```

**The critical difference**: Both losses backpropagate through the encoder. The latent representations learn to simultaneously:
1. Predict masked video features (world understanding)
2. Support trajectory prediction (planning utility)

This creates representations that are **inherently planning-aware** — they encode not just "what is in the scene" but "what matters for driving."

---

## 2. Architecture

### 2.1 Components

```
┌──────────────────────────────────────────────────┐
│                V-JEPA2-Drive                      │
│                                                   │
│  ┌─────────────┐                                  │
│  │  V-JEPA2    │──── z_masked ──→ Predictor(φ)    │
│  │  Encoder(θ) │                   → L_jepa       │
│  │  (ViT-L,    │                                  │
│  │  scale→ViT-g│                                  │
│  │             │──── z_full ────→ TrajectoryHead(ψ)│
│  │  TRAINABLE  │                   → L_traj       │
│  └─────────────┘                                  │
│                                                   │
│  Target Encoder(θ̄)  ← EMA of θ (provides targets)│
└──────────────────────────────────────────────────┘
```

**Encoder** (`VisionTransformer`): We start on the pretrained V-JEPA2 **ViT-L** encoder while keeping it **trainable** end-to-end. The design remains compatible with larger encoders for later scaling.

**Predictor** (`VisionTransformerPredictor`): The existing V-JEPA2 predictor. Takes masked encoder features and predicts target features at masked positions. Unchanged from V-JEPA2.

**TrajectoryHead** (NEW): A lightweight module that takes the encoder's **unmasked** features and predicts future ego-vehicle waypoints. This is the key new component.

**Target Encoder**: EMA of the encoder, as in standard V-JEPA2. Provides prediction targets for the JEPA loss.

### 2.2 TrajectoryHead Design

The trajectory head should be lightweight so it doesn't dominate the encoder's representation learning — we want the JEPA loss and trajectory loss to jointly shape the representations.

```python
# New: src/models/trajectory_head.py

class TrajectoryHead(nn.Module):
    """Predict multimodal future trajectories from V-JEPA2 encoder features.

    Reuses AttentivePooler to aggregate encoder tokens and outputs a
    distribution with K modes:
      means      [B, K, T, 2]
      log_stds   [B, K, T, 2]
      mode_logits [B, K]
    """

    def __init__(
        self,
        embed_dim=1024,        # ViT-L dimension (switch to 1408 for ViT-g)
        num_modes=4,
        num_waypoints=12,      # e.g., 6 seconds at 2Hz
        waypoint_dim=2,        # (x, y) in ego frame
        num_heads=16,
        pooler_depth=2,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_waypoints = num_waypoints

        # Reuse V-JEPA2's AttentivePooler pattern
        self.pooler = AttentivePooler(
            num_queries=num_modes * num_waypoints,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=pooler_depth,
        )
        self.mean_proj = nn.Linear(embed_dim, waypoint_dim)
        self.log_std_proj = nn.Linear(embed_dim, waypoint_dim)
        self.mode_logit_proj = nn.Linear(embed_dim, 1)

    def predict_distribution(self, encoder_features, ego_history=None):
        """
        Args:
            encoder_features: [B, N, D] - full (unmasked) encoder output
            ego_history: [B, H, 2] optional past ego waypoints
        Returns:
            distribution over K trajectory modes
        """
        batch_size = encoder_features.size(0)
        embed_dim = encoder_features.size(-1)
        queries = self.pooler(encoder_features)  # [B, K*T, D]
        queries = queries.view(
            batch_size, self.num_modes, self.num_waypoints, embed_dim
        )
        means = self.mean_proj(queries)
        log_stds = self.log_std_proj(queries)
        mode_logits = self.mode_logit_proj(queries.mean(dim=2)).squeeze(-1)
        return TrajectoryDistribution(means, log_stds, mode_logits)
```

**Why AttentivePooler**: It's already part of V-JEPA2's codebase (`src/models/attentive_pooler.py`), proven to work with V-JEPA2 features, and provides a clean cross-attention mechanism where learnable query tokens attend to the encoder's spatiotemporal features.

### 2.3 Forward Pass During Training

```python
def forward_step(encoder, target_encoder, predictor, trajectory_head,
                 video_clip, masks_enc, masks_pred, gt_trajectory, ego_history, trajectory_dt):
    """
    Single training step for V-JEPA2-Drive.
    Both JEPA loss and trajectory loss backpropagate through encoder.
    """

    # === JEPA Branch (existing V-JEPA2 logic) ===
    # Encode masked context
    z_context = encoder(video_clip, masks=masks_enc)

    # Predict target features at masked positions
    z_pred = predictor(z_context, masks_enc, masks_pred)

    # Target: EMA encoder on full (unmasked) video
    with torch.no_grad():
        z_target = target_encoder(video_clip)
        z_target = F.layer_norm(z_target, (z_target.size(-1),))

    # JEPA loss: L1 between predicted and target at masked positions
    h = apply_masks(z_target, masks_pred)
    loss_jepa = F.l1_loss(z_pred, h)

    # === Trajectory Branch (NEW) ===
    # Encode FULL (unmasked) video for trajectory prediction
    z_full = encoder(video_clip)  # [B, N, D] — no masking

    # Predict multimodal trajectory distribution
    pred_dist = trajectory_head.predict_distribution(z_full, ego_history=ego_history)

    # Trajectory loss: composite multimodal objective
    traj_stats = compute_multimodal_trajectory_losses(
        pred_dist, gt_trajectory, trajectory_dt
    )
    loss_traj = traj_stats["loss"]

    # === Combined Loss ===
    loss = loss_jepa + lambda_traj * loss_traj

    return loss
```

**Important note**: The encoder runs twice per step — once with masking (for JEPA loss) and once without masking (for trajectory loss). This is necessary because:
- The JEPA loss needs masked input (that's the self-supervised objective)
- The trajectory head needs the full scene representation (you can't predict a trajectory from a partially masked scene)

This doubles the encoder compute, but can be mitigated with gradient accumulation or by running the trajectory branch less frequently (e.g., every N steps).

**Optimization**: Alternatively, we can run the encoder once on the unmasked video, then apply masking post-hoc to get the context tokens for the predictor. This is actually what V-JEPA2 already does — the target encoder processes unmasked input. We could share this forward pass:

```python
# More efficient: single encoder forward, branch after
z_full = encoder(video_clip)           # Full features for trajectory
z_context = apply_masks(z_full, masks_enc)  # Masked features for JEPA predictor
```

This avoids the double forward pass entirely.

---

## 3. Loss Functions

### 3.1 Composite Multimodal Trajectory Objective (Current)

Trajectory learning now uses a multimodal distribution objective rather than minADE alone. The head predicts:
- `means`: `[B, K, T, 2]`
- `log_stds`: `[B, K, T, 2]`
- `mode_logits`: `[B, K]`

Loss terms are computed in `app/vjepa_drive/losses.py` and combined as:

```python
loss_traj = (
    w_ade * min_ade
    + w_fde * min_fde
    + w_nll * nll
    + w_calib * calib
    + w_smooth * smoothness
    + w_feas * feasibility
)
```

Where:
- `min_ade`, `min_fde`: best-mode displacement errors
- `nll`: multimodal Gaussian mixture negative log-likelihood
- `calib`: predicted variance calibration against squared error
- `smoothness`: jerk penalty
- `feasibility`: speed/acceleration limit penalty

Ground truth waypoints come from the ego-motion data in driving datasets:
- **NVIDIA AV**: `EgomotionState` with pose/kinematics interpolation
- **nuScenes**: ego-pose annotations
- **Argoverse v2**: ego trajectory logs
- **Waymo**: per-frame vehicle pose

### 3.2 JEPA + Trajectory Coupling

The total training objective remains:

```python
loss_total = loss_jepa + lambda_traj * loss_traj
```

`lambda_traj` is scheduled (constant/linear/cosine ramp) to avoid destabilizing early joint training while the trajectory head is still cold-starting.

---

## 4. Data Pipeline

### 4.1 Multi-Dataset Training

The user correctly identified abundant open-source data with ego-motion:

| Dataset | Clips/Scenes | Ego-Motion | Cameras | Hours |
|---|---|---|---|---|
| NVIDIA Physical AI AV | Large | Pose, vel, accel, curvature | Multi | Large |
| nuScenes | 1,000 scenes | 6-DOF pose at 2Hz | 6 cameras | ~5.5h |
| Argoverse v2 | 1,000 scenes | 6-DOF pose at 10Hz | 7 cameras + 2 stereo | ~4h |
| Waymo Open | 1,150 scenes | 6-DOF pose at 10Hz | 5 cameras | ~6.4h |
| Lyft L5 | 170,000 scenes | 6-DOF pose | 7 cameras | ~1000h |
| ONCE | 1M frames | GPS/IMU | 1 camera | - |

V-JEPA2 already supports multi-dataset training with weighted sampling (`datasets_weights` in config). We extend this pattern:

```yaml
# configs/train/vitg16/driving-joint-256px-16f.yaml
data:
  dataset_type: DrivingVideoDataset
  datasets:
    - /path/to/nvidia_av_clips.csv
    - /path/to/nuscenes_clips.csv
    - /path/to/argoverse2_clips.csv
    - /path/to/waymo_clips.csv
  datasets_weights:
    - 0.4   # NVIDIA AV (largest)
    - 0.2   # nuScenes
    - 0.2   # Argoverse v2
    - 0.2   # Waymo
  batch_size: 6
  crop_size: 256
  dataset_fpcs: [16]
  fps: 4
  patch_size: 16
  tubelet_size: 2
```

### 4.2 Dataset Implementation

```python
# New: src/datasets/driving_dataset.py

class DrivingVideoDataset(torch.utils.data.Dataset):
    """Video dataset with ego-motion trajectory ground truth.

    Each sample returns:
    - video_clip: [C, T, H, W] driving video
    - gt_trajectory: [T_future, 2] future ego waypoints in ego frame
    """

    def __init__(
        self,
        data_paths_csv,        # CSV with video paths + ego-motion paths
        frames_per_clip=16,
        trajectory_horizon=12, # Future waypoints to predict
        trajectory_dt=0.5,     # Seconds between waypoints
        transform=None,
        fps=4,
    ):
        self.samples = self._load_csv(data_paths_csv)
        self.fpc = frames_per_clip
        self.horizon = trajectory_horizon
        self.dt = trajectory_dt
        self.transform = transform
        self.fps = fps

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video clip (reuse V-JEPA2's Decord-based loading)
        video_clip = self._load_video(sample['video_path'])

        # Load ego-motion and compute future trajectory
        ego_poses = self._load_ego_motion(sample['ego_path'])
        clip_end_time = sample['clip_end_timestamp']

        # Future waypoints: positions at t+0.5s, t+1.0s, ..., t+6.0s
        # relative to ego frame at clip_end_time
        gt_trajectory = self._compute_future_waypoints(
            ego_poses, clip_end_time,
            horizon=self.horizon, dt=self.dt
        )

        if self.transform:
            video_clip = self.transform(video_clip)

        return video_clip, gt_trajectory

    def _compute_future_waypoints(self, ego_poses, t_ref, horizon, dt):
        """Compute future (x, y) waypoints in ego frame at t_ref."""
        # Get reference pose (ego position/heading at end of clip)
        ref_pose = ego_poses.interpolate(t_ref)
        ref_inv = ref_pose.inverse()  # Transform to ego frame

        waypoints = []
        for i in range(1, horizon + 1):
            t_future = t_ref + i * dt
            future_pose = ego_poses.interpolate(t_future)
            # Transform to ego frame
            relative = ref_inv @ future_pose
            waypoints.append([relative.x, relative.y])

        return torch.tensor(waypoints, dtype=torch.float32)  # [horizon, 2]
```

### 4.3 Collation with Masking

The existing `MaskCollator` from V-JEPA2 generates masks for the JEPA branch. We extend the collation to also include trajectory ground truth:

```python
# Extension of existing collation in src/masks/multiseq_multiblock3d.py

class DrivingMaskCollator:
    """Wraps V-JEPA2's MaskCollator to also handle trajectory GT."""

    def __init__(self, mask_collator):
        self.mask_collator = mask_collator

    def __call__(self, batch):
        video_clips, gt_trajectories = zip(*batch)

        # Existing mask collation for JEPA branch
        collated_clips, masks_enc, masks_pred = self.mask_collator(
            [(clip,) for clip in video_clips]
        )

        # Stack trajectory GT
        gt_trajectories = torch.stack(gt_trajectories)

        return collated_clips, masks_enc, masks_pred, gt_trajectories
```

---

## 5. Training Loop

### 5.1 Modified Training Loop

The training loop is a modification of `app/vjepa/train.py`. The key changes are:
1. Data loader returns trajectory GT alongside video clips
2. Loss computation includes both JEPA and trajectory terms
3. Encoder gradients come from both losses

```python
# New: app/vjepa_drive/train.py

def train_step(
    encoder, target_encoder, predictor, trajectory_head,
    optimizer, scaler,
    video_clips, masks_enc, masks_pred, gt_trajectories, ego_histories,
    trajectory_dt, traj_loss_weights,
    lambda_traj=1.0, loss_exp=1.0,
):
    """Single training step for V-JEPA2-Drive."""

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        # ──── JEPA Branch (existing V-JEPA2 logic) ────

        # Target: EMA encoder on full video (no grad)
        with torch.no_grad():
            h = target_encoder(video_clips)
            h = F.layer_norm(h, (h.size(-1),))

        # Context: encoder on masked video
        z = encoder(video_clips, masks=masks_enc)

        # Predict masked targets
        z_pred = predictor(z, masks_enc, masks_pred)

        # JEPA loss
        h_masked = apply_masks(h, masks_pred)
        loss_jepa = (torch.abs(z_pred - h_masked) ** loss_exp).mean() / loss_exp

        # ──── Trajectory Branch (NEW) ────

        # Full (unmasked) encoder features for trajectory prediction
        z_full = encoder(video_clips)  # [B, N, D]

        # Predict multimodal trajectory distribution
        pred_dist = trajectory_head.predict_distribution(
            z_full, ego_history=ego_histories
        )
        traj_stats = compute_multimodal_trajectory_losses(
            pred_dist=pred_dist,
            gt_trajectory=gt_trajectories,
            dt=trajectory_dt,
            weights=traj_loss_weights,
        )
        loss_traj = traj_stats["loss"]

        # ──── Combined Loss ────
        loss = loss_jepa + lambda_traj * loss_traj

    # Backward + optimize (gradients flow through encoder from BOTH losses)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # EMA update of target encoder
    with torch.no_grad():
        update_ema(target_encoder, encoder, momentum)

    return loss_jepa.item(), loss_traj.item()
```

### 5.2 Optimizer Groups

Following V-JEPA2's existing pattern, but now including trajectory head parameters:

```python
param_groups = [
    # Encoder params (updated by both losses)
    {'params': encoder_weights, 'weight_decay': wd},
    {'params': encoder_biases, 'weight_decay': 0},
    # Predictor params (updated by JEPA loss only, but that's automatic)
    {'params': predictor_weights, 'weight_decay': wd},
    {'params': predictor_biases, 'weight_decay': 0},
    # Trajectory head params (updated by trajectory loss only)
    {'params': trajectory_head_weights, 'weight_decay': wd},
    {'params': trajectory_head_biases, 'weight_decay': 0},
]
```

### 5.3 Training Configuration

```yaml
# configs/train/vitg16/driving-joint-256px-16f.yaml
app: vjepa_drive

data:
  dataset_type: DrivingVideoDataset
  datasets:
    - /path/to/nvidia_av_clips.csv
    - /path/to/nuscenes_clips.csv
    - /path/to/argoverse2_clips.csv
  datasets_weights: [0.4, 0.3, 0.3]
  batch_size: 6
  crop_size: 256
  dataset_fpcs: [16]
  fps: 4
  patch_size: 16
  tubelet_size: 2

model:
  model_name: vit_giant_xformers
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
  use_mask_tokens: true
  use_rope: true
  # Trajectory head config
  trajectory_head:
    num_modes: 4
    num_waypoints: 12
    waypoint_dim: 2
    pooler_depth: 2

loss:
  loss_exp: 1.0
  lambda_traj_start: 0.25
  lambda_traj_end: 1.0
  lambda_traj_schedule: cosine
  lambda_traj_ramp_epochs: 10
  loss_ade_weight: 1.0
  loss_fde_weight: 1.0
  loss_nll_weight: 0.5
  loss_calib_weight: 0.1
  loss_smoothness_weight: 0.05
  loss_feasibility_weight: 0.05

mask:
  # Same masking config as V-JEPA2 pretraining
  - spatial_scale: [0.15, 0.15]
    temporal_scale: [1.0, 1.0]
    num_blocks: 8
  - spatial_scale: [0.7, 0.7]
    temporal_scale: [1.0, 1.0]
    num_blocks: 2

optimization:
  lr: 0.000525
  start_lr: 0.0001
  final_lr: 0.000001
  weight_decay: 0.04
  warmup: 40
  epochs: 50
  ema: [0.99925, 0.99925]

meta:
  dtype: bfloat16
  use_sdpa: true
```

---

## 6. Training Strategy

### 6.1 Phase 1: Initialize from Pretrained V-JEPA2

Start from the pretrained V-JEPA2 ViT-L checkpoint (Meta release `vitl.pt`) to retain strong video priors, then scale model/data once the joint objective is stable.
- An encoder already trained on 22M internet videos with strong temporal understanding
- A predictor already trained for masked latent prediction

We add only the `TrajectoryHead` (randomly initialized) and begin joint training.

**Warmup strategy**: Use a lower learning rate for the encoder initially (since it's already well-trained) and a higher rate for the trajectory head (randomly initialized). This prevents the trajectory loss from initially destabilizing the encoder's learned representations.

```python
param_groups = [
    {'params': encoder_params, 'lr': base_lr * 0.1},      # Lower LR for pretrained
    {'params': predictor_params, 'lr': base_lr * 0.1},     # Lower LR for pretrained
    {'params': trajectory_head_params, 'lr': base_lr},     # Full LR for new head
]
```

### 6.2 Phase 2: Scale Up Data and Loss Tuning

Once Phase 1 validates that joint training works (encoder learns trajectory-useful features):

1. Scale to all available driving datasets (nuScenes + Argoverse + Waymo + NVIDIA AV + Lyft)
2. Tune the composite trajectory objective weights (`ADE/FDE/NLL/calibration/smoothness/feasibility`)
3. Experiment with `lambda_traj` scheduling to find the right JEPA/trajectory balance
4. Validate runtime safety thresholds against uncertainty and dynamics signals

### 6.3 Phase 3: Multi-Camera + Extended Horizon

1. Extend to multi-camera input (per-camera encoding + camera ID embeddings)
2. Extend trajectory horizon (12 → 24 waypoints, 6s → 12s)
3. Add multi-modal trajectory prediction (K modes with selection)

### 6.4 Phase 4 (Future): Online Learning

The user's vision of online learning from watching a human driver:
- Deploy trained model on vehicle
- Model observes human driver's actions in real-time
- Continuously fine-tune trajectory head on streaming ego-motion data
- The JEPA branch continues self-supervised learning on live video

---

## 7. File-Level Implementation Plan

### 7.1 New Files

| File | Purpose | Complexity |
|---|---|---|
| `src/models/trajectory_head.py` | TrajectoryHead using AttentivePooler | Small (~80 lines) |
| `src/datasets/driving_dataset.py` | Multi-dataset driving video + ego-motion loader | Medium (~200 lines) |
| `src/datasets/ego_motion_utils.py` | Utilities for loading ego-motion from nuScenes/Argo/Waymo/NVIDIA formats | Medium (~150 lines) |
| `app/vjepa_drive/__init__.py` | Module init | Tiny |
| `app/vjepa_drive/train.py` | Joint training loop (modified from `app/vjepa/train.py`) | Large (~500 lines) |
| `configs/train/vitg16/driving-joint-256px-16f.yaml` | Training config | Small |

### 7.2 Files to Modify

| File | Modification | Reason |
|---|---|---|
| `app/main.py` | Register `vjepa_drive` app | Entry point |
| `app/vjepa/transforms.py` | Add driving-specific transform (no horizontal flip) | Data augmentation |

### 7.3 Files Reused Unchanged

| File | Why |
|---|---|
| `src/models/vision_transformer.py` | Encoder architecture unchanged |
| `src/models/predictor.py` | JEPA predictor unchanged |
| `src/models/attentive_pooler.py` | Used by TrajectoryHead |
| `src/models/utils/modules.py` | Block, Attention, etc. unchanged |
| `src/masks/` | Masking strategy unchanged |
| `src/utils/` | Distributed training, schedulers reused |
| `src/hub/backbones.py` | Model loading reused |

---

## 8. Key Design Decisions

### 8.1 Why Joint Training (Not Frozen Encoder + Head)?

Frozen encoder approaches (V-JEPA2-AC, attentive probing) assume the pretrained representations are already optimal for the downstream task. For trajectory prediction, this is unlikely:

- V-JEPA2's representations are optimized for **video understanding** (predicting masked patches)
- Trajectory prediction requires **spatial precision** (exact lane positions, distances to objects)
- Joint training allows the encoder to develop **dual-purpose** representations: good for both video prediction AND spatial planning

The JEPA loss acts as a regularizer preventing the encoder from collapsing to trajectory-only features — it must still predict masked video content, maintaining rich scene understanding.

### 8.2 Why Single-Camera First?

Starting with a single front camera (before multi-camera):
- Simplest integration with V-JEPA2's existing architecture
- Most driving datasets have a dominant front camera
- Tesla's early FSD versions also started single-camera
- Validates the joint training concept before adding multi-camera complexity

### 8.3 Why Composite Trajectory Loss?

- Displacement terms (`minADE/minFDE`) keep geometric accuracy anchored.
- NLL trains the model to represent uncertainty across modes.
- Calibration ties predicted uncertainty to observed error.
- Smoothness/feasibility reduce dynamically implausible trajectories.
- Weighted terms provide explicit tradeoff controls without changing architecture.

### 8.4 Encoder Double Forward Pass vs. Shared Forward

Option A: Run encoder twice (masked + unmasked) — simple but 2x encoder compute
Option B: Run encoder once unmasked, apply masking to features post-hoc — more efficient

```python
# Option B (recommended):
z_full = encoder(video_clip)           # One forward pass, no masking
z_context = apply_masks(z_full, masks_enc)  # Post-hoc masking for JEPA
pred_dist = trajectory_head.predict_distribution(z_full, ego_history=ego_history)
```

This is valid because V-JEPA2's masking is applied at the **token level** (selecting which patch tokens to keep), not at the input pixel level. The encoder can process all patches, and then we select the subset for the predictor.

**Note**: This changes the training slightly from standard V-JEPA2, where the encoder only sees masked tokens (efficiency gain from fewer tokens). We'd need to verify this doesn't degrade JEPA training quality. If it does, we fall back to Option A.

---

## 9. Evaluation

### 9.1 Trajectory Metrics (Standard AD)

- **minADE / minFDE**: Best-mode geometric accuracy
- **NLL**: Likelihood quality of multimodal distribution
- **Calibration error**: Match between predicted variance and realized error
- **Smoothness / feasibility penalties**: Dynamic plausibility
- **Miss Rate**: Fraction of predictions > 2m from GT at final step
- **Collision Rate**: (if obstacle info available)

### 9.2 Representation Quality (JEPA Metrics)

- **SSv2 probe accuracy**: Does the encoder still work well for video understanding?
- **K400 probe accuracy**: Appearance understanding preservation
- **JEPA loss on held-out video**: Has the prediction quality degraded?

This dual evaluation ensures the joint training improves trajectory prediction **without sacrificing** the encoder's general video understanding.

### 9.3 Ablations

| Experiment | Purpose |
|---|---|
| V-JEPA2 frozen + trajectory head | Baseline: is joint training actually better? |
| V-JEPA2-Drive (joint) | Main approach |
| V-JEPA2-Drive without JEPA loss | Does the JEPA loss help as regularizer? |
| Lambda_traj sweep | Find optimal loss weighting |
| Trajectory head depth sweep | How complex should the head be? |

---

## 10. Implementation Priority

### Sprint 1 (Weeks 1-2): Data Pipeline
1. Implement `DrivingVideoDataset` for nuScenes (best documented, most accessible)
2. Implement ego-motion → trajectory waypoint extraction
3. Implement `DrivingMaskCollator` extending V-JEPA2's collation
4. Validate data loading end-to-end

### Sprint 2 (Weeks 3-4): Joint Training
5. Implement `TrajectoryHead` using `AttentivePooler`
6. Implement composite multimodal trajectory losses (`ADE/FDE/NLL/calibration/smoothness/feasibility`)
7. Implement `app/vjepa_drive/train.py` (modify existing train.py)
8. Train on nuScenes with pretrained V-JEPA2 ViT-L (smaller, faster iteration)
9. Validate: does trajectory loss decrease? Does JEPA loss remain stable?

### Sprint 3 (Weeks 5-6): Scale Up
10. Add Argoverse v2 and Waymo dataset loaders
11. Scale to ViT-g encoder
12. Tune loss weights and lambda schedules; evaluate uncertainty calibration
13. Run ablation experiments

### Sprint 4 (Weeks 7-8): Multi-Camera + Analysis
14. Add camera ID embeddings for multi-camera
15. Extended trajectory horizon experiments
16. Compare frozen vs. joint training
17. Analyze learned representations (what changed from vanilla V-JEPA2?)

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Trajectory loss destabilizes JEPA training | Use lower LR for encoder; tune lambda_traj carefully; warmup trajectory head before joint training |
| Encoder loses general video understanding | Monitor SSv2/K400 probing accuracy; use JEPA loss as regularizer |
| Composite loss over-regularizes trajectory accuracy | Tune per-term weights; track ADE/FDE/NLL/calibration together instead of a single scalar |
| Double encoder forward too expensive | Use shared forward pass (Option B in Section 8.4) |
| Different datasets have different ego-motion formats | Abstract behind `ego_motion_utils.py` with per-dataset adapters |
| Single camera insufficient for planning | Start here to validate approach; multi-camera in Sprint 4 |
