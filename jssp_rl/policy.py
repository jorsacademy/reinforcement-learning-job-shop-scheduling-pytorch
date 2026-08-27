from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Categorical


class JobTransformerActorCritic(nn.Module):
    """Masked Transformer actor-critic over per-job scheduling features."""

    def __init__(
        self,
        *,
        job_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        self.job_embed = nn.Sequential(
            nn.Linear(job_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.global_embed = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, job_features, global_features, action_mask):
        x = self.encoder(self.job_embed(job_features))
        g = self.global_embed(global_features)
        pooled = x.mean(dim=1)

        context = torch.cat([pooled, g], dim=-1)
        context_jobs = context[:, None, :].expand(-1, x.shape[1], -1)
        logits = self.actor(torch.cat([x, context_jobs], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        value = self.critic(context).squeeze(-1)
        return logits, value

    def distribution_and_value(self, job_features, global_features, action_mask):
        logits, value = self(job_features, global_features, action_mask)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, job_features, global_features, action_mask, *, greedy=False):
        dist, value = self.distribution_and_value(
            job_features, global_features, action_mask
        )
        action = torch.argmax(dist.logits, dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value
