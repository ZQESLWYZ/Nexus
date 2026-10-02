import torch
from torch import nn

from nexus.models.gated_deltanet import GatedDeltaNet
from nexus.models.moe import MoEFeedForward
from nexus.models.normalization import AdaptiveLayerNorm


class NexusBlock(nn.Module):
    def __init__(self, dim: int, heads: int, experts: int, top_k: int, condition_dim: int, bits: int):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(dim, condition_dim)
        self.attn = GatedDeltaNet(dim, heads)
        self.norm2 = AdaptiveLayerNorm(dim, condition_dim)
        self.moe = MoEFeedForward(dim, experts, top_k, bits)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x, condition))
        moe_output, balance = self.moe(self.norm2(x, condition))
        return x + moe_output, balance
