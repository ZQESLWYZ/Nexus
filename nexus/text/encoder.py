import torch
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(self, vocabulary: int = 8192, dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids).mean(dim=1))
