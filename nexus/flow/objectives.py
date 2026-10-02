import torch

from nexus.flow.schedule import interpolate, sample_time


def conditional_flow_matching_loss(model, data: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(data)
    time = sample_time(data.shape[0], data.device)
    state = interpolate(noise, data, time)
    target = data - noise
    prediction, balance = model(state, time, text)
    loss = torch.mean((prediction - target) ** 2)
    return loss, {"flow_loss": float(loss.detach()), "router_balance": float(balance.detach())}
