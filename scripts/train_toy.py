import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.config import NexusConfig
from nexus.data.synthetic import SyntheticTextImageDataset
from nexus.models.backbone import NexusBackbone
from nexus.training.trainer import NexusTrainer


def main() -> None:
    config = NexusConfig(image_size=32, dim=64, depth=2, heads=4, text_dim=64, device="cuda")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = NexusBackbone(config).to(device)
    dataset = SyntheticTextImageDataset(8, 32, 64)
    trainer = NexusTrainer(model)
    for index in range(3):
        item = dataset[index]
        metrics = trainer.step(item["latent"].unsqueeze(0).to(device), item["text"].unsqueeze(0).to(device))
        print(index, metrics)


if __name__ == "__main__":
    main()
