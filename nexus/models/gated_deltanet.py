import torch
from torch import nn


class GatedDeltaNet(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.alpha = nn.Linear(dim, heads)
        self.beta = nn.Linear(dim, heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, self.heads, self.head_dim)
        k = self.k(x).view(batch, length, self.heads, self.head_dim)
        v = self.v(x).view(batch, length, self.heads, self.head_dim)
        alpha = torch.sigmoid(self.alpha(x)).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)
        state = x.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        outputs = []
        identity = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).expand(batch, self.heads, -1, -1)
        for step in range(length):
            kt = k[:, step]
            vt = v[:, step]
            qt = q[:, step]
            outer = kt.unsqueeze(-1) @ kt.unsqueeze(-2)
            update = beta[:, :, step].unsqueeze(-1).unsqueeze(-1)
            decay = alpha[:, :, step].unsqueeze(-1).unsqueeze(-1)
            state = state @ (decay * (identity - update * outer)) + update * (vt.unsqueeze(-1) @ kt.unsqueeze(-2))
            outputs.append((state @ qt.unsqueeze(-1)).squeeze(-1))
        return self.out(torch.stack(outputs, dim=1).reshape(batch, length, -1))
