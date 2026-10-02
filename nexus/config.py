from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class NexusConfig:
    image_size: int = 32
    latent_channels: int = 4
    patch_size: int = 2
    dim: int = 128
    depth: int = 4
    heads: int = 4
    experts: int = 4
    top_k: int = 2
    text_dim: int = 128
    text_length: int = 32
    quant_bits: int = 4
    device: str = "cuda"

    @classmethod
    def from_json(cls, path: str | Path) -> "NexusConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2
