import torch


def normalize_latent(latent: torch.Tensor) -> torch.Tensor:
    return latent / latent.std(dim=tuple(range(1, latent.ndim)), keepdim=True).clamp_min(1e-6)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(-1, 1).add(1).div(2)
