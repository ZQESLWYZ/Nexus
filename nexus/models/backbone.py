import torch
from torch import nn

from nexus.config import NexusConfig
from nexus.models.blocks import NexusBlock
from nexus.models.embeddings import PatchEmbed, PatchUnembed, TimestepEmbedding


class NexusBackbone(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        self.patch = PatchEmbed(config.latent_channels, config.dim, config.patch_size)
        self.time = TimestepEmbedding(config.dim)
        self.text = nn.Linear(config.text_dim, config.dim)
        self.blocks = nn.ModuleList(
            [
                NexusBlock(
                    config.dim,
                    config.heads,
                    config.experts,
                    config.top_k,
                    config.dim,
                    config.quant_bits,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.unpatch = PatchUnembed(config.dim, config.latent_channels, config.patch_size)
        self.output = nn.Conv2d(config.latent_channels, config.latent_channels, 1)

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height = latent.shape[-2] // self.config.patch_size
        width = latent.shape[-1] // self.config.patch_size
        tokens = self.patch(latent)
        condition = self.time(timestep) + self.text(text)
        balance = latent.new_zeros(())
        for block in self.blocks:
            tokens, current_balance = block(tokens, condition)
            balance = balance + current_balance
        tokens = self.norm(tokens)
        velocity = self.unpatch(tokens, height, width)
        return self.output(velocity), balance / len(self.blocks)
