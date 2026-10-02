import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.config import NexusConfig
from nexus.inference.pipeline import NexusPipeline
from nexus.io.images import save_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--output", default="outputs/sample.png")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pipeline = NexusPipeline(NexusConfig(device=args.device))
    image = pipeline([args.prompt], steps=args.steps)
    save_grid(image, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
