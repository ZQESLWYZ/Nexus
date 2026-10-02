import torch


def image_statistics(images: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(images.mean()),
        "std": float(images.std()),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
    }
