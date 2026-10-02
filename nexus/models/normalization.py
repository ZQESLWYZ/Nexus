import torch
from torch import nn


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Linear(condition_dim, dim * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
