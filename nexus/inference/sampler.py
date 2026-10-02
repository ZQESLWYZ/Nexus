import torch

from nexus.flow.solver import euler_integrate


class EulerSampler:
    def __init__(self, steps: int = 8):
        self.steps = steps

    def sample(self, model, latent: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return euler_integrate(model, latent, text, self.steps)
