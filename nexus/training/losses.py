import torch


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def router_regularization(balance: torch.Tensor, coefficient: float = 0.01) -> torch.Tensor:
    return coefficient * (balance - 0.5).square()
