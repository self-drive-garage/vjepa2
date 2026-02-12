# V-JEPA2 for End-to-End Autonomous Driving: Architectural Plan

## Executive Summary

This document proposes adapting V-JEPA2's self-supervised video understanding architecture for end-to-end autonomous driving (E2E AD), leveraging the NVIDIA Physical AI Autonomous Vehicles dataset. The key insight is that **V-JEPA2 already implements the core machinery needed** — a video world model with action-conditioned prediction (V-JEPA2-AC) — and our task is to adapt it from robot manipulation to autonomous driving while remaining as close as possible to the original approach.

---

## 1. Why V-JEPA2 Is Uniquely Suited for E2E AD

### 1.1 Existing Capabilities That Transfer Directly

| V-JEPA2 Capability | AD Application |
|---|---|
| Self-supervised video pretraining on 22M videos | Learn driving scene dynamics from internet driving video |
| Action-conditioned predictor (V-JEPA2-AC) | Predict future driving scene states given steering/acceleration |
| Latent-space planning via CEM optimization | Plan trajectories by optimizing actions toward goal states |
| Block-causal attention masks | Respect temporal causality in driving sequences |
| 3D-RoPE for spatio-temporal reasoning | Handle multi-frame driving sequences |
| AttentivePooler for downstream task heads | Extract compact representations for planning heads |
| EMA target encoder | Stable training objective |

### 1.2 Key Gaps to Address

1. **Single-camera → Multi-camera**: V-JEPA2 processes one video stream; AD uses 6-8 surround cameras
2. **Robot actions (7D) → Vehicle actions (steering, acceleration, brake)**: Different action space
3. **Robot state (7D) → Vehicle state (velocity, yaw rate, etc.)**: Different state representation
4. **No ego-trajectory output**: V-JEPA2-AC predicts future representations, not explicit waypoints
5. **No BEV (Bird's Eye View) representation**: Current architecture operates in perspective view only
6. **No multi-sensor fusion**: No LiDAR/radar integration in current architecture

---

## 2. Proposed Architecture: V-JEPA2-Drive

### 2.1 High-Level Architecture

```
Phase 1: Self-Supervised Pretraining (unchanged V-JEPA2)
    Internet video → Frozen ViT-g encoder with rich scene understanding

Phase 2: Driving-Specific Action-Conditioned Post-Training
    NVIDIA AV dataset → V-JEPA2-AC-Drive predictor (modified AC predictor)

Phase 3: Driving Task Heads (trajectory, occupancy, etc.)
    Frozen encoder + AC predictor → Lightweight task-specific heads
```

### 2.2 Detailed Component Design

#### Component A: Multi-Camera Encoder (Modified `VisionTransformer`)

**Strategy: Per-Camera Encoding with Camera-ID Tokens**

This is the most V-JEPA2-faithful approach. Rather than introducing complex BEV projections (which would deviate significantly from the JEPA paradigm), we:

1. **Encode each camera view independently** using the frozen V-JEPA2 ViT-g encoder
2. **Add camera-ID embeddings** to distinguish which camera produced each token
3. **Concatenate all camera tokens** into a single sequence for the predictor

**Rationale**: V-JEPA2-AC already handles multi-token sequences with interleaved conditioning tokens (actions, states). We extend this pattern to include camera identity.

```python
# Modification to: src/models/vision_transformer.py
# New: CameraAwareVisionTransformer

class CameraAwareVisionTransformer(VisionTransformer):
    """Extends ViT with camera-ID embeddings for multi-camera AD."""

    def __init__(self, num_cameras=7, **kwargs):
        super().__init__(**kwargs)
        # Learnable camera embeddings added to patch tokens
        self.camera_embed = nn.Parameter(
            torch.zeros(num_cameras, 1, 1, self.embed_dim)
        )  # [num_cams, 1, 1, D] → broadcast over B, N

    def forward(self, x, camera_id=0, masks=None):
        # x: [B, C, T, H, W] from one camera
        features = super().forward(x, masks=masks)
        # Add camera embedding
        features = features + self.camera_embed[camera_id]
        return features
```

**File changes**: `src/models/vision_transformer.py` — add `CameraAwareVisionTransformer` subclass

**Alternative (more complex, for future work)**: Cross-camera attention layers between camera token sequences, similar to how UniAD uses BEV queries to attend across cameras. This would be a deeper modification.

#### Component B: Driving Action-Conditioned Predictor (Modified `VisionTransformerPredictorAC`)

**Strategy: Replace robot action/state encoders with vehicle action/state encoders**

The existing V-JEPA2-AC predictor in `src/models/ac_predictor.py` already implements the exact pattern we need:
- Action tokens interleaved with frame tokens
- State tokens for conditioning
- Extrinsics tokens (camera parameters)
- Block-causal attention mask for temporal ordering

**Modifications**:

```python
# Modification to: src/models/ac_predictor.py
# New: VisionTransformerPredictorAC_Drive

class VisionTransformerPredictorAC_Drive(nn.Module):
    def __init__(
        self,
        # ... existing params ...
        vehicle_action_dim=3,      # [steering_angle, acceleration, brake]
        vehicle_state_dim=8,       # [vx, vy, vz, ax, ay, az, yaw_rate, curvature]
        num_cameras=7,             # Number of surround cameras
        use_camera_extrinsics=True,  # Camera pose relative to ego
        camera_extrinsics_dim=7,   # [qx,qy,qz,qw, tx,ty,tz] per camera
        **kwargs
    ):
        super().__init__()

        # Vehicle-specific encoders (replacing robot 7D encoders)
        self.action_encoder = nn.Linear(vehicle_action_dim, predictor_embed_dim)
        self.state_encoder = nn.Linear(vehicle_state_dim, predictor_embed_dim)

        # Camera extrinsics encoder (leverages existing extrinsics_encoder pattern)
        if use_camera_extrinsics:
            self.extrinsics_encoder = nn.Linear(
                camera_extrinsics_dim * num_cameras, predictor_embed_dim
            )

        # Multi-camera token merging
        self.num_cameras = num_cameras

        # Rest follows existing AC predictor architecture exactly
        # (predictor_blocks, causal attention mask, etc.)
```

**Token interleaving for driving** (analogous to existing `[action, state, frame_tokens]` pattern):

```
Per timestep t:
    [action_t, state_t, extrinsics_t, cam0_tokens_t, cam1_tokens_t, ..., camN_tokens_t]
```

**Attention mask**: Extend `build_action_block_causal_attention_mask` (from `src/models/utils/modules.py`) to handle multi-camera token sequences. Each timestep's tokens can attend to same and previous timesteps.

**File changes**:
- `src/models/ac_predictor.py` — add `VisionTransformerPredictorAC_Drive`
- `src/models/utils/modules.py` — extend `build_action_block_causal_attention_mask` for multi-camera

#### Component C: Driving Task Heads

**Strategy: Leverage existing AttentivePooler pattern with driving-specific outputs**

The existing `AttentivePooler` in `src/models/attentive_pooler.py` provides the exact abstraction needed — learnable query tokens that cross-attend to encoder features. We create driving-specific poolers:

##### C.1: Trajectory Planning Head (Primary Output)

```python
# New file: src/models/driving_heads.py

class TrajectoryPlanningHead(nn.Module):
    """Predicts future ego-vehicle waypoints from V-JEPA2 features."""

    def __init__(
        self,
        embed_dim=1408,       # ViT-g dimension
        num_waypoints=12,     # 6 seconds at 2Hz
        waypoint_dim=3,       # (x, y, heading) in ego frame
        num_heads=16,
        depth=4,
    ):
        super().__init__()
        # Attentive pooler with num_waypoints query tokens
        self.pooler = AttentivePooler(
            num_queries=num_waypoints,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
        # Project each pooled query to a waypoint
        self.waypoint_proj = nn.Linear(embed_dim, waypoint_dim)

    def forward(self, encoder_features):
        # encoder_features: [B, N_total_tokens, D] (all cameras, all frames)
        queries = self.pooler(encoder_features)  # [B, num_waypoints, D]
        waypoints = self.waypoint_proj(queries)   # [B, num_waypoints, 3]
        return waypoints
```

##### C.2: Occupancy Prediction Head (Safety-Critical)

```python
class OccupancyHead(nn.Module):
    """Predicts future BEV occupancy grid from V-JEPA2 features."""

    def __init__(
        self,
        embed_dim=1408,
        bev_h=200, bev_w=200,  # BEV grid resolution
        num_future_frames=6,
        num_heads=16,
        depth=2,
    ):
        super().__init__()
        self.num_bev_queries = bev_h * bev_w  # Too many for full attention
        # Use a smaller set of queries + deconv
        self.pooler = AttentivePooler(
            num_queries=256,  # Compressed BEV queries
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            # Reshape to spatial grid + upsample
            # ... deconvolution layers to bev_h x bev_w
        )
```

##### C.3: World Model Planning Head (Most V-JEPA2-Native)

This is the most faithful adaptation — using the V-JEPA2-AC planning approach directly:

```python
class LatentPlanningHead(nn.Module):
    """Plans by optimizing actions in V-JEPA2's latent space.

    Directly analogous to V-JEPA2-AC's CEM planning for robotics,
    but with vehicle actions and driving goals.
    """

    def __init__(self, ac_predictor, encoder, planning_horizon=12):
        super().__init__()
        self.predictor = ac_predictor  # V-JEPA2-AC-Drive predictor
        self.encoder = encoder          # Frozen V-JEPA2 encoder
        self.horizon = planning_horizon

    def plan(self, current_obs, goal_obs, num_iterations=5, num_samples=512):
        """CEM-based planning in latent space (from V-JEPA2-AC paper)."""
        # Encode current and goal observations
        with torch.no_grad():
            z_current = self.encoder(current_obs)
            z_goal = self.encoder(goal_obs)

        # Initialize action distribution
        action_mean = torch.zeros(self.horizon, 3)  # [steering, accel, brake]
        action_std = torch.ones(self.horizon, 3)

        for _ in range(num_iterations):
            # Sample action sequences
            actions = action_mean + action_std * torch.randn(num_samples, self.horizon, 3)

            # Rollout in latent space using AC predictor
            z_predicted = self.predictor.rollout(z_current, actions)

            # Score: L1 distance to goal in latent space
            scores = -torch.mean(torch.abs(z_predicted - z_goal), dim=-1)

            # CEM update: fit to top-k samples
            top_k = scores.topk(num_samples // 10).indices
            action_mean = actions[top_k].mean(0)
            action_std = actions[top_k].std(0)

        return action_mean  # Best action sequence
```

**File changes**: New `src/models/driving_heads.py`

---

## 3. Data Pipeline: NVIDIA Physical AI AV Dataset Integration

### 3.1 New Dataset Class

```python
# New file: src/datasets/driving_dataset.py

class DrivingVideoDataset(torch.utils.data.Dataset):
    """Loads driving data from NVIDIA Physical AI AV dataset.

    Leverages the physical_ai_av devkit for data access.
    """

    def __init__(
        self,
        dataset_interface,   # PhysicalAIAVDatasetInterface instance
        clip_ids,            # List of clip IDs to use
        num_frames=16,
        cameras=None,        # List of camera names, or None for all
        transform=None,
    ):
        self.interface = dataset_interface
        self.clip_ids = clip_ids
        self.num_frames = num_frames
        self.cameras = cameras or self._get_all_cameras()
        self.transform = transform

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]

        # Load multi-camera video frames
        camera_frames = {}
        for cam_name in self.cameras:
            feature = getattr(self.interface.features.CAMERA, cam_name)
            video_reader, timestamps = self.interface.get_clip_feature(
                clip_id, feature
            )
            # Sample num_frames uniformly
            frame_indices = np.linspace(0, len(timestamps)-1, self.num_frames, dtype=int)
            frames = video_reader[frame_indices]  # [T, H, W, C]
            if self.transform:
                frames = self.transform(frames)
            camera_frames[cam_name] = frames

        # Load egomotion (pose, velocity, acceleration)
        ego_feature = self.interface.features.LABELS.EGOMOTION
        egomotion = self.interface.get_clip_feature(clip_id, ego_feature)

        # Extract vehicle states at frame timestamps
        states = self._extract_vehicle_states(egomotion, timestamps[frame_indices])

        # Compute actions (delta between consecutive states)
        actions = self._compute_actions(states)

        return {
            'camera_frames': camera_frames,  # Dict[cam_name, Tensor[T, C, H, W]]
            'states': states,                 # Tensor[T, state_dim]
            'actions': actions,               # Tensor[T-1, action_dim]
            'timestamps': timestamps[frame_indices],
        }

    def _extract_vehicle_states(self, egomotion_interpolator, timestamps):
        """Extract [vx, vy, vz, ax, ay, az, yaw_rate, curvature] at each timestamp."""
        states = []
        for t in timestamps:
            pose = egomotion_interpolator(t)
            states.append(torch.tensor([
                pose.velocity.x, pose.velocity.y, pose.velocity.z,
                pose.acceleration.x, pose.acceleration.y, pose.acceleration.z,
                pose.curvature,  # or yaw rate
                0.0,  # placeholder
            ]))
        return torch.stack(states)

    def _compute_actions(self, states):
        """Compute actions as deltas between consecutive states."""
        # For driving: actions = [delta_x, delta_y, delta_heading]
        # Derived from egomotion differences
        return states[1:] - states[:-1]
```

**File changes**: New `src/datasets/driving_dataset.py`

### 3.2 Data Augmentation Considerations

The existing `app/vjepa/transforms.py` handles spatial augmentations. For driving:
- **Keep**: Random resize/crop (with consistent crop across all cameras within a frame)
- **Remove**: Horizontal flip (would reverse left/right turning semantics)
- **Add**: Photometric augmentations (brightness, contrast — consistent across cameras)
- **Add**: Temporal jittering within clips

---

## 4. Training Pipeline

### 4.1 Three-Phase Training Strategy (Following V-JEPA2's Staged Approach)

#### Phase 1: Self-Supervised Pretraining on Driving Video (Optional Enhancement)

**Goal**: Extend V-JEPA2's video understanding to driving-specific scenes

**Approach**: Continue V-JEPA2's masked prediction pretraining on driving video data
- Use the existing `app/vjepa/train.py` training loop **unchanged**
- Simply point the data config to driving video datasets (e.g., nuScenes raw video, YouTube driving compilations)
- This is optional because the pretrained V-JEPA2 already has strong video understanding

```yaml
# configs/train/vitg16/driving-pretrain-256px-16f.yaml
data:
  dataset_type: VideoDataset
  datasets:
    - /path/to/nvidia_av_video_paths.csv    # Camera videos from NVIDIA dataset
    - /path/to/nuscenes_video_paths.csv      # Optional: nuScenes raw video
  batch_size: 6
  crop_size: 256
  dataset_fpcs: [16]
  fps: 4
  patch_size: 16
  tubelet_size: 2
# ... rest follows existing cooldown config pattern
```

#### Phase 2: Action-Conditioned Driving Predictor Training

**Goal**: Train the V-JEPA2-AC-Drive predictor on ego-vehicle actions

**Approach**: Follows V-JEPA2-AC training exactly, but with driving actions/states

```yaml
# configs/train/vitg16/driving-ac-256px-16f.yaml
app: vjepa_drive  # New app entry point
data:
  dataset_type: DrivingVideoDataset
  datasets:
    - nvidia_av_clip_ids.csv
  batch_size: 4
  crop_size: 256
  dataset_fpcs: [16]
  fps: 4
  num_cameras: 7
model:
  model_name: vit_giant_xformers
  pred_depth: 24
  pred_embed_dim: 1024
  pred_num_heads: 16
  vehicle_action_dim: 3
  vehicle_state_dim: 8
  use_camera_extrinsics: true
  is_frame_causal: true
optimization:
  lr: 0.0003
  epochs: 50
  ema: [0.999, 0.999]
```

**Training loop** (new file `app/vjepa_drive/train.py`):
- Freeze V-JEPA2 encoder (load pretrained weights)
- Train AC-Drive predictor with teacher-forcing + rollout loss (following V-JEPA2-AC exactly)
- Loss: L1 in latent space between predicted and actual future frame representations

```python
def loss_fn(predictor, encoder, batch):
    """Driving AC predictor loss — follows V-JEPA2-AC paper exactly."""
    camera_frames = batch['camera_frames']  # Multi-camera video
    actions = batch['actions']               # Vehicle actions
    states = batch['states']                 # Vehicle states

    with torch.no_grad():
        # Encode all camera views for all frames
        all_features = []
        for cam_name, frames in camera_frames.items():
            features = encoder(frames, camera_id=cam_name)
            all_features.append(features)
        z_target = torch.cat(all_features, dim=1)  # [B, N_total, D]

    # Teacher-forcing loss
    z_pred = predictor(z_target[:, :T-1], actions, states)
    loss_tf = F.l1_loss(z_pred, z_target[:, 1:T])

    # Rollout loss (2-step autoregressive)
    z_pred_1 = predictor(z_target[:, :T-2], actions[:, :T-2], states[:, :T-2])
    z_pred_2 = predictor(z_pred_1, actions[:, T-2:T-1], states[:, T-2:T-1])
    loss_rollout = F.l1_loss(z_pred_2, z_target[:, T-1:T])

    return loss_tf + loss_rollout
```

#### Phase 3: Task-Specific Head Training

**Goal**: Train lightweight heads for trajectory prediction, occupancy, etc.

**Approach**: Freeze encoder + AC predictor, train only task heads

```python
# Training trajectory planning head
def train_trajectory_head(encoder, ac_predictor, trajectory_head, dataloader):
    """Frozen backbone, trainable trajectory head only."""
    encoder.eval()
    ac_predictor.eval()

    for batch in dataloader:
        with torch.no_grad():
            features = encode_multi_camera(encoder, batch['camera_frames'])
            # Optionally: run AC predictor for future-aware features
            future_features = ac_predictor(features, batch['actions'], batch['states'])

        # Train trajectory head
        predicted_waypoints = trajectory_head(
            torch.cat([features, future_features], dim=1)
        )

        # Loss: L2 on waypoints + heading angle loss
        gt_trajectory = batch['future_ego_trajectory']  # From egomotion
        loss = F.mse_loss(predicted_waypoints, gt_trajectory)
        loss.backward()
```

---

## 5. File-Level Implementation Plan

### 5.1 New Files to Create

| File | Purpose |
|---|---|
| `src/models/driving_heads.py` | TrajectoryPlanningHead, OccupancyHead, LatentPlanningHead |
| `src/datasets/driving_dataset.py` | DrivingVideoDataset using NVIDIA AV devkit |
| `app/vjepa_drive/train.py` | Training loop for driving AC predictor |
| `app/vjepa_drive/__init__.py` | Module init |
| `configs/train/vitg16/driving-ac-256px-16f.yaml` | AC predictor training config |
| `configs/train/vitg16/driving-pretrain-256px-16f.yaml` | Optional driving pretraining config |
| `configs/eval/driving/trajectory.yaml` | Trajectory evaluation config |

### 5.2 Existing Files to Modify

| File | Modification |
|---|---|
| `src/models/ac_predictor.py` | Add `VisionTransformerPredictorAC_Drive` subclass |
| `src/models/vision_transformer.py` | Add `CameraAwareVisionTransformer` subclass with camera embeddings |
| `src/models/utils/modules.py` | Extend `build_action_block_causal_attention_mask` for multi-camera tokens |
| `app/vjepa/transforms.py` | Add driving-specific transform (no horizontal flip, consistent multi-cam augmentation) |
| `app/main.py` | Register `vjepa_drive` app |
| `src/hub/backbones.py` | Add `vjepa2_drive_vit_giant` model loading function |

### 5.3 No Changes Needed (Reuse As-Is)

| File | Why It Works Unchanged |
|---|---|
| `src/models/utils/pos_embs.py` | 3D-RoPE and sincos embeddings work directly |
| `src/models/utils/patch_embed.py` | PatchEmbed3D tokenization is camera-agnostic |
| `src/masks/` | Masking strategy unchanged for pretraining phase |
| `src/utils/` | Distributed training, schedulers, optimizers all reusable |
| `src/models/attentive_pooler.py` | CrossAttention pooling used directly in driving heads |

---

## 6. Key Design Decisions

### 6.1 Why NOT use BEV explicitly?

Traditional E2E AD models (UniAD, VAD, SparseDrive) project camera features into a Bird's Eye View (BEV) grid. We **deliberately avoid this** because:

1. **V-JEPA2's philosophy is representation learning, not spatial reconstruction**. The model should learn whatever spatial representation is most useful, not be constrained to a human-designed grid.
2. **The V-JEPA2 encoder already captures 3D structure** through self-supervised video pretraining — objects move in 3D, and the model must predict their trajectories.
3. **BEV requires camera intrinsics/extrinsics at training time**, adding complexity. V-JEPA2's approach of learning from raw video is more general.
4. **The latent planning approach** (Component C.3) plans in learned representation space, which may capture richer information than BEV occupancy.

**Escape hatch**: If explicit BEV is needed for safety-critical occupancy prediction, the OccupancyHead (Component C.2) can learn a BEV projection implicitly through the attention queries.

### 6.2 Multi-Camera Handling: Concatenation vs. Cross-Attention

We chose **token concatenation** (all camera tokens in one sequence) over cross-camera attention because:
- It's the simplest extension of V-JEPA2-AC's existing token interleaving
- The block-causal attention mask already handles heterogeneous token types
- V-JEPA2's transformer can handle long sequences (2048+ tokens per frame × 7 cameras = manageable with activation checkpointing)
- Camera extrinsics tokens provide the geometric context needed for spatial reasoning

### 6.3 Action Representation

For the NVIDIA AV dataset, actions are derived from egomotion:
- **Action = delta ego-pose between frames**: `[delta_x, delta_y, delta_heading]` in ego frame
- Computed from the `RigidTransformInterpolator` in the devkit's egomotion module
- This is directly analogous to V-JEPA2-AC's robot actions (delta end-effector state)

---

## 7. Comparison with Existing E2E AD Approaches

| Approach | Architecture | Inputs | Outputs | Relationship to V-JEPA2 |
|---|---|---|---|---|
| **UniAD** | BEV + detection + tracking + planning cascade | Cameras + LiDAR | BEV, trajectories, planning | Very different — modular pipeline |
| **VAD** | Vectorized scene + planning | Cameras | Vectorized map + trajectory | Partially related — uses learned queries |
| **SparseDrive** | Sparse representation + planning | Cameras | Sparse agents + trajectory | Related — sparse token approach |
| **GAIA-1** | Video generation world model | Camera + actions | Future video pixels | Related philosophy, but generative (pixels) |
| **MILE** | Latent world model + planning | Camera | Latent states + actions | **Most similar** — latent space planning |
| **V-JEPA2-Drive (ours)** | Latent JEPA world model + planning | Multi-camera video + ego actions | Latent representations + trajectory | Direct extension of V-JEPA2-AC |

**Key differentiator**: V-JEPA2-Drive operates **entirely in learned latent space** (not pixel space, not BEV space), which is computationally efficient and captures abstract scene semantics rather than low-level details.

---

## 8. Computational Requirements

### Estimated GPU Requirements

| Phase | GPUs | Duration | Notes |
|---|---|---|---|
| Phase 1 (optional driving pretraining) | 32-64 × A100/H100 | 2-4 days | Cooldown-style fine-tuning |
| Phase 2 (AC-Drive predictor) | 8-16 × A100/H100 | 1-2 days | Encoder frozen, only predictor trains |
| Phase 3 (task heads) | 4-8 × A100/H100 | Hours | Very lightweight |
| CEM Planning (inference) | 1 × A100 | ~15 sec/action | Following V-JEPA2-AC benchmark |

### Memory Optimization

- **Activation checkpointing**: Already implemented in V-JEPA2 (`use_activation_checkpointing=True`)
- **bfloat16 training**: Already supported (`meta.dtype: bfloat16`)
- **Multi-camera sequence**: 7 cameras × 256 tokens/frame × 8 frames = 14,336 tokens per timestep. This is large but manageable with:
  - Per-camera encoding (separate forward passes, no memory scaling issue)
  - Token compression via pooling before concatenation (reduce 256 → 64 tokens per camera)

---

## 9. Evaluation Strategy

### 9.1 Open-Loop Metrics (Following AD benchmarks)

- **L2 displacement error** at 1s, 2s, 3s horizons
- **Collision rate** against GT objects
- **Planning score** (nuScenes planning benchmark format)

### 9.2 World Model Quality Metrics

- **Feature prediction error** (L1 in latent space, following V-JEPA2-AC)
- **Temporal consistency** of predicted representations
- **Action sensitivity** — does the model respond meaningfully to different action inputs?

### 9.3 Qualitative Evaluation

- **Latent space visualization** via t-SNE/UMAP of predicted features
- **Goal-conditioned planning visualization** — does CEM find reasonable trajectories?
- **Failure case analysis** — when does latent planning produce unsafe trajectories?

---

## 10. Implementation Priority Order

### Sprint 1 (Weeks 1-2): Data Pipeline
1. Implement `DrivingVideoDataset` using NVIDIA AV devkit
2. Extract and validate egomotion data (states, actions)
3. Create data loading configs and verify with existing V-JEPA2 training loop

### Sprint 2 (Weeks 3-4): Multi-Camera Encoder + AC Predictor
4. Implement `CameraAwareVisionTransformer` with camera embeddings
5. Implement `VisionTransformerPredictorAC_Drive` with vehicle action/state encoders
6. Extend causal attention mask for multi-camera tokens
7. Verify forward pass with dummy data

### Sprint 3 (Weeks 5-6): AC Predictor Training
8. Implement driving AC predictor training loop (`app/vjepa_drive/train.py`)
9. Train on NVIDIA AV dataset with teacher-forcing + rollout loss
10. Evaluate latent prediction quality

### Sprint 4 (Weeks 7-8): Task Heads + Planning
11. Implement `TrajectoryPlanningHead` and train on GT trajectories
12. Implement CEM-based latent planning (following V-JEPA2-AC)
13. Evaluate open-loop planning performance
14. Ablation studies and optimization

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Multi-camera token sequence too long for attention | Use per-camera token compression (AttentivePooler with 64 queries per camera) |
| Latent planning doesn't produce safe trajectories | Add explicit trajectory regularization (smoothness, lane-keeping priors) |
| NVIDIA AV dataset too small for pretraining | Phase 1 uses internet driving video; NVIDIA data only for Phase 2-3 |
| No explicit 3D geometry in latent space | Camera extrinsics tokens + egomotion conditioning provide geometric grounding |
| Planning too slow for real-time | Amortize planning with learned policy distillation (train MLP to mimic CEM output) |
