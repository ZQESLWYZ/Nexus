import torch
from torch import nn


def fake_quantize(x: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
    levels = 2 ** bits - 1
    if symmetric:
        scale = x.detach().abs().amax().clamp_min(1e-6) / (levels / 2)
        quantized = torch.clamp(torch.round(x / scale), -(levels // 2), levels // 2) * scale
    else:
        minimum = x.detach().amin()
        maximum = x.detach().amax()
        scale = (maximum - minimum).clamp_min(1e-6) / levels
        quantized = torch.round((x - minimum) / scale).clamp(0, levels) * scale + minimum
    return x + (quantized - x).detach()


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 4):
        super().__init__(in_features, out_features, bias)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize(self.weight, self.bits, True)
        activation = fake_quantize(x, self.bits, False)
        return nn.functional.linear(activation, weight, self.bias)
