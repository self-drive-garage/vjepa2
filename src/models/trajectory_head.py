# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


class TrajectoryHead(nn.Module):
    """Predicts ego-vehicle waypoints from V-JEPA2 encoder features.

    Uses AttentivePooler (existing V-JEPA2 component) to aggregate
    spatiotemporal tokens into a compact representation, then
    projects to waypoints.
    """

    def __init__(
        self,
        embed_dim=1408,        # ViT-g dimension
        num_waypoints=12,      # e.g., 6 seconds at 2Hz
        waypoint_dim=2,        # (x, y) in ego frame
        num_heads=16,
        mlp_ratio=4.0,
        pooler_depth=2,
        init_std=0.02,
        qkv_bias=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim

        # Reuse V-JEPA2's AttentivePooler pattern
        self.pooler = AttentivePooler(
            num_queries=num_waypoints,
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

        # Initialize projection layer
        nn.init.trunc_normal_(self.waypoint_proj.weight, std=init_std)
        nn.init.constant_(self.waypoint_proj.bias, 0)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [B, N, D] - full (unmasked) encoder output
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
        """
        queries = self.pooler(encoder_features)      # [B, num_waypoints, D]
        waypoints = self.waypoint_proj(queries)       # [B, num_waypoints, waypoint_dim]
        return waypoints
