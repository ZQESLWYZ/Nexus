from nexus.config import NexusConfig
from nexus.inference.pipeline import NexusPipeline


def test_pipeline_shape():
    pipeline = NexusPipeline(NexusConfig(image_size=32, dim=32, depth=1, heads=4, text_dim=32, device="cpu"))
    images = pipeline(["a tree"], steps=2, seed=3)
    assert tuple(images.shape) == (1, 3, 32, 32)
