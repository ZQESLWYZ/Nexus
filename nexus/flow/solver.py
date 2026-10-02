import torch


@torch.no_grad()
def euler_integrate(model, state: torch.Tensor, text: torch.Tensor, steps: int) -> torch.Tensor:
    step = 1.0 / steps
    for index in range(steps):
        time = torch.full((state.shape[0],), index / steps, device=state.device, dtype=state.dtype)
        velocity, _ = model(state, time, text)
        state = state + step * velocity
    return state
