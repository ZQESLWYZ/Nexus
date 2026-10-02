import torch
from torch import nn


class TinyDecoder(nn.Module):
    def __init__(self, latent_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
