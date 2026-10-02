import torch

from nexus.config import NexusConfig
from nexus.models.gated_deltanet import GatedDeltaNet
from nexus.models.moe import MoEFeedForward


def test_gated_deltanet_shape():
    module = GatedDeltaNet(16, 4)
    output = module(torch.randn(2, 5, 16))
    assert output.shape == (2, 5, 16)


def test_moe_shape():
    module = MoEFeedForward(16, 4, 2)
    output, balance = module(torch.randn(2, 5, 16))
    assert output.shape == (2, 5, 16)
    assert balance.ndim == 0


def test_config_tokens():
    assert NexusConfig(image_size=32, patch_size=2).tokens == 256
