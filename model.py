"""The single PenroseAim Direct Transformer architecture."""

from __future__ import annotations

import math

import torch
from torch import nn

from config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        index = torch.arange(half, device=time.device, dtype=time.dtype)
        frequencies = (1000.0 * math.pi) ** (-index / (half - 1))
        phase = time[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class DirectTransformer(nn.Module):
    """Predict geometry velocity and clean identified-probe logits."""

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        num_vertex_in: int,
    ):
        super().__init__()
        if num_vertex_in <= 0:
            raise ValueError("num_vertex_in must be positive")
        self.num_global_tokens = config.num_global_tokens
        self.num_vertex_in = num_vertex_in
        state_dim = 3 + num_vertex_in
        self.input_projection = nn.Linear(state_dim, config.d_model)
        self.color_embedding = nn.Embedding(2, config.d_model)
        self.class_embedding = nn.Embedding(num_classes, config.class_embed_dim)
        self.class_projection = nn.Linear(config.class_embed_dim, config.d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim * 2, config.d_model),
        )
        self.global_tokens = nn.Parameter(
            torch.randn(1, config.num_global_tokens, config.d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        colors: torch.Tensor,
        time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        tiles = self.input_projection(state)
        tiles = tiles + self.color_embedding(colors.long())
        global_tokens = self.global_tokens.expand(batch_size, -1, -1)
        hidden = torch.cat((global_tokens, tiles), dim=1)

        condition = self.time_embedding(time).unsqueeze(1)
        condition = condition + self.class_projection(
            self.class_embedding(class_labels.long())
        ).unsqueeze(1)
        hidden = self.encoder(hidden + condition)
        hidden = hidden[:, self.num_global_tokens :]
        return self.output_projection(self.output_norm(hidden))
