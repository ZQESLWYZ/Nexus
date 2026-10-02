import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, channels: int, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x)
        return tokens.flatten(2).transpose(1, 2)


class PatchUnembed(nn.Module):
    def __init__(self, dim: int, channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(dim, channels, patch_size, patch_size)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)
        return self.proj(grid)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.net(timestep.reshape(-1, 1))
