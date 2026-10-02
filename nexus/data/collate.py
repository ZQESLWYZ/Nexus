import torch


def collate_latents(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "latent": torch.stack([item["latent"] for item in batch]),
        "text": torch.stack([item["text"] for item in batch]),
    }
