import torch

from nexus.config import NexusConfig
from nexus.inference.sampler import EulerSampler
from nexus.models.backbone import NexusBackbone
from nexus.text.encoder import PromptEncoder
from nexus.text.tokenizer import HashTokenizer
from nexus.utils import resolve_device
from nexus.vae.decoder import TinyDecoder


class NexusPipeline:
    def __init__(self, config: NexusConfig | None = None, device: str | None = None):
        self.config = config or NexusConfig()
        self.device = resolve_device(device or self.config.device)
        self.tokenizer = HashTokenizer(self.config.text_length)
        self.encoder = PromptEncoder(dim=self.config.text_dim).to(self.device)
        self.backbone = NexusBackbone(self.config).to(self.device)
        self.decoder = TinyDecoder(self.config.latent_channels).to(self.device)
        self.sampler = EulerSampler()
        self.eval()

    def eval(self):
        self.encoder.eval()
        self.backbone.eval()
        self.decoder.eval()
        return self

    @torch.no_grad()
    def __call__(self, prompts: list[str], steps: int = 8, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        batch = len(prompts)
        size = self.config.image_size // 4
        latent = torch.randn(batch, self.config.latent_channels, size, size, generator=generator, device=self.device)
        token_ids = self.tokenizer.encode(prompts, self.device)
        text = self.encoder(token_ids)
        self.sampler.steps = steps
        latent = self.sampler.sample(self.backbone, latent, text)
        return (self.decoder(latent) + 1).div(2).clamp(0, 1)
