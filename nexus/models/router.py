import torch
from torch import nn


class TopKRouter(nn.Module):
    def __init__(self, dim: int, experts: int, top_k: int):
        super().__init__()
        self.proj = nn.Linear(dim, experts)
        self.experts = experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(x)
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        return indices, weights
