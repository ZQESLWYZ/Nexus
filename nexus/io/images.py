from pathlib import Path

import torch
from PIL import Image


def save_grid(images: torch.Tensor, path: str | Path) -> None:
    images = images.detach().cpu().clamp(0, 1)
    row = torch.cat([image for image in images], dim=-1)
    array = row.permute(1, 2, 0).mul(255).byte().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
