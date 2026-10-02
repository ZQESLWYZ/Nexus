import torch
from torch import nn

from nexus.models.quantization import QuantizedLinear
from nexus.models.router import TopKRouter


class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bits: int):
        super().__init__()
        self.up = QuantizedLinear(dim, hidden_dim, bits=bits)
        self.down = QuantizedLinear(hidden_dim, dim, bits=bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.up(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, dim: int, experts: int, top_k: int, bits: int = 4, expansion: int = 4):
        super().__init__()
        self.router = TopKRouter(dim, experts, top_k)
        self.experts = nn.ModuleList([Expert(dim, dim * expansion, bits) for _ in range(experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, weights = self.router(x)
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                mask = indices[..., slot] == expert_id
                if mask.any():
                    selected = x[mask]
                    output[mask] = output[mask] + weights[..., slot][mask].unsqueeze(-1) * expert(selected)
        load_balance = weights.mean()
        return output, load_balance
