import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)
