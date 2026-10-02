import torch


def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def interpolate(noise: torch.Tensor, data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    shape = (time.shape[0],) + (1,) * (noise.ndim - 1)
    return (1 - time.reshape(shape)) * noise + time.reshape(shape) * data
