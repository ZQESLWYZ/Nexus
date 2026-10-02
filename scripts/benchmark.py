import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.config import NexusConfig
from nexus.inference.pipeline import NexusPipeline
from nexus.metrics.benchmark import benchmark_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pipeline = NexusPipeline(NexusConfig(device=args.device))
    result = benchmark_pipeline(pipeline, ["a watercolor mountain landscape"], args.steps, 1, args.repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
