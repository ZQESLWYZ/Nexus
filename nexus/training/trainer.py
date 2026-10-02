import torch

from nexus.flow.objectives import conditional_flow_matching_loss
from nexus.training.optim import build_optimizer


class NexusTrainer:
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = build_optimizer(model, learning_rate)

    def step(self, latent: torch.Tensor, text: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = conditional_flow_matching_loss(self.model, latent, text)
        loss.backward()
        self.optimizer.step()
        return metrics
