# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_drive.types import TrajectoryDistribution
from src.models.attentive_pooler import AttentivePooler


class TrajectoryHead(nn.Module):
    """Predict multimodal ego trajectories from video + ego-history context.

    The head explicitly encodes past ego-motion waypoints (history branch),
    fuses them with V-JEPA2 video tokens, then predicts a trajectory
    distribution with K modes:
        means      [B, K, T, 2]
        log_stds   [B, K, T, 2]
        mode_logits [B, K]
    """

    def __init__(
        self,
        embed_dim=1408,        # ViT-g dimension
        num_waypoints=12,      # e.g., 6 seconds at 2Hz
        waypoint_dim=2,        # (x, y) in ego frame
        num_modes=4,           # multimodal hypotheses
        history_dim=2,         # ego-history waypoint dim
        history_hidden_dim=None,
        history_tokens=8,      # e.g., 4 seconds at dt=0.5
        num_heads=16,
        mlp_ratio=4.0,
        pooler_depth=2,
        init_std=0.02,
        qkv_bias=True,
        min_log_std=-4.0,
        max_log_std=1.0,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.num_modes = num_modes
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        history_hidden_dim = history_hidden_dim or embed_dim
        self.history_tokens = history_tokens
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, history_hidden_dim),
            nn.SiLU(),
            nn.Linear(history_hidden_dim, embed_dim),
        )
        self.history_norm = nn.LayerNorm(embed_dim)
        self.history_pos_embed = nn.Parameter(torch.zeros(1, history_tokens, embed_dim))

        # Reuse V-JEPA2's attentive pooling, now with K*T queries.
        self.pooler = AttentivePooler(
            num_queries=num_modes * num_waypoints,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=pooler_depth,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=True,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.waypoint_proj = nn.Linear(embed_dim, waypoint_dim)
        self.log_std_proj = nn.Linear(embed_dim, waypoint_dim)
        self.mode_logit_proj = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.history_pos_embed, std=init_std)
        nn.init.trunc_normal_(self.waypoint_proj.weight, std=init_std)
        nn.init.constant_(self.waypoint_proj.bias, 0)
        nn.init.trunc_normal_(self.log_std_proj.weight, std=init_std)
        nn.init.constant_(self.log_std_proj.bias, -1.0)
        nn.init.trunc_normal_(self.mode_logit_proj.weight, std=init_std)
        nn.init.constant_(self.mode_logit_proj.bias, 0)

    def _interpolate_history_pos_embed(self, num_tokens, dtype, device):
        pos = self.history_pos_embed
        if pos.size(1) == num_tokens:
            return pos.to(device=device, dtype=dtype)
        pos = pos.transpose(1, 2)  # [1, D, H]
        pos = F.interpolate(pos, size=num_tokens, mode="linear", align_corners=False)
        pos = pos.transpose(1, 2)
        return pos.to(device=device, dtype=dtype)

    def _encode_history(self, ego_history, batch_size, dtype, device):
        if ego_history is None:
            return None
        if ego_history.ndim != 3:
            raise ValueError(f"Expected ego_history with shape [B, H, C], got {tuple(ego_history.shape)}")
        if ego_history.size(0) != batch_size:
            raise ValueError(
                f"Ego history batch ({ego_history.size(0)}) must match encoder batch ({batch_size})"
            )
        history_tokens = self.history_encoder(ego_history.to(device=device, dtype=dtype))
        pos = self._interpolate_history_pos_embed(history_tokens.size(1), dtype=dtype, device=device)
        history_tokens = self.history_norm(history_tokens + pos)
        return history_tokens

    @staticmethod
    def _gather_modes(mode_tensor, mode_idx):
        # mode_tensor: [B, K, ...], mode_idx: [B]
        batch_idx = torch.arange(mode_tensor.size(0), device=mode_tensor.device)
        return mode_tensor[batch_idx, mode_idx]

    def predict_distribution(self, encoder_features, ego_history=None) -> TrajectoryDistribution:
        """
        Args:
            encoder_features: [B, N, D] - full (unmasked) encoder output
            ego_history: [B, H, 2] past ego waypoints in current ego frame
        Returns:
            TrajectoryDistribution with:
              means: [B, K, T, waypoint_dim]
              log_stds: [B, K, T, waypoint_dim]
              mode_logits: [B, K]
        """
        if encoder_features.ndim != 3:
            raise ValueError(
                f"Expected encoder features with shape [B, N, D], got {tuple(encoder_features.shape)}"
            )
        bsz, _, _ = encoder_features.shape
        dtype = encoder_features.dtype
        device = encoder_features.device

        history_tokens = self._encode_history(
            ego_history=ego_history,
            batch_size=bsz,
            dtype=dtype,
            device=device,
        )
        if history_tokens is not None:
            fused_tokens = torch.cat([encoder_features, history_tokens], dim=1)
        else:
            fused_tokens = encoder_features

        # [B, K*T, D]
        queries = self.pooler(fused_tokens)
        queries = queries.reshape(
            bsz,
            self.num_modes,
            self.num_waypoints,
            queries.size(-1),
        )
        means = self.waypoint_proj(queries)
        log_stds = self.log_std_proj(queries).clamp(self.min_log_std, self.max_log_std)

        mode_context = queries.mean(dim=2)  # [B, K, D]
        mode_logits = self.mode_logit_proj(mode_context).squeeze(-1)

        return TrajectoryDistribution(
            means=means,
            log_stds=log_stds,
            mode_logits=mode_logits,
        )

    def forward(self, encoder_features, ego_history=None):
        """Backward-compatible API: return the most likely single trajectory."""
        pred = self.predict_distribution(encoder_features, ego_history=ego_history)
        mode_idx = torch.argmax(pred.mode_probs, dim=-1)
        return self._gather_modes(pred.means, mode_idx)
