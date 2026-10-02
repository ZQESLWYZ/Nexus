import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.config import NexusConfig
from nexus.inference.pipeline import NexusPipeline
from nexus.metrics.benchmark import benchmark_pipeline
from nexus.metrics.quality import image_statistics
from nexus.utils import count_parameters, seed_everything


def main() -> None:
    seed_everything(7)
    config = NexusConfig(image_size=32, dim=64, depth=2, heads=4, text_dim=64, device="cuda")
    pipeline = NexusPipeline(config)
    prompts = ["a red kite above a blue lake", "a small robot in a garden"]
    images = pipeline(prompts, steps=3, seed=11)
    benchmark = benchmark_pipeline(pipeline, prompts, steps=3, warmup=1, repeats=2)
    result = {
        "device": str(pipeline.device),
        "parameters": count_parameters(pipeline.backbone),
        "image_shape": list(images.shape),
        "image_statistics": image_statistics(images),
        "benchmark": benchmark,
        "status": "passed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
