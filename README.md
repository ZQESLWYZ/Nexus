# Nexus: Structured Synergy for Efficient Text-to-Image Generation

![Status](https://img.shields.io/badge/ACCV%202026-Accepted-1f6feb)
![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the project code for **Nexus: Structured Synergy for Efficient Text-to-Image Generation using Rectified Flow Model**. The paper has been **accepted at ACCV 2026**.

Nexus is an efficient text-to-image generation framework built around three cooperating ideas:

1. Sparse Mixture-of-Experts feed-forward layers increase model capacity while activating only a small subset of parameters for each token.
2. Gated DeltaNet replaces quadratic self-attention with a recurrent linear-attention style update.
3. Per-layer and per-expert fake quantization provides a compact INT4-oriented training and inference path.

The implementation in this repository is intentionally compact and executable. It exposes the same algorithmic structure described in the paper, including conditional flow matching, Euler integration, top-k expert routing, gated DeltaNet recurrence, quantized linear projections, prompt hashing, latent decoding, training utilities, checkpoint I/O, and benchmark scripts. The default configuration is designed for smoke tests and development on a single workstation rather than for reproducing the full 7B-parameter training run.

## Paper summary

The paper studies the deployment bottlenecks of modern text-to-image flow models:

- High inference cost caused by dense parameter activation.
- Quadratic attention cost for long image-token sequences.
- Large memory footprints caused by full-precision weights and activations.

The proposed Nexus design combines a sparse 8-expert style feed-forward structure, Gated DeltaNet linear attention, and expert-aware low-bit quantization inside a rectified-flow denoising backbone. The paper reports a 7B-parameter model with 1.6B activated parameters, 512x512 generation in 1.42 seconds on an NVIDIA A100, 3.2 GB peak memory, 185 GFLOPs, FID 5.8, and CLIP score 0.329 under the reported evaluation protocol.

Those paper-level numbers require the full training corpus, the paper configuration, the corresponding checkpoints, and the evaluation pipeline. This repository provides a small reference implementation that is suitable for understanding the method, running tests, profiling the execution path, and extending the model.

## Repository layout

```text
.
├── nexus/
│   ├── models/
│   │   ├── embeddings.py
│   │   ├── normalization.py
│   │   ├── quantization.py
│   │   ├── router.py
│   │   ├── moe.py
│   │   ├── gated_deltanet.py
│   │   ├── blocks.py
│   │   └── backbone.py
│   ├── flow/
│   │   ├── schedule.py
│   │   ├── objectives.py
│   │   └── solver.py
│   ├── text/
│   ├── vae/
│   ├── data/
│   ├── inference/
│   ├── training/
│   ├── metrics/
│   └── io/
├── scripts/
│   ├── smoke_test.py
│   ├── benchmark.py
│   ├── sample.py
│   └── train_toy.py
├── tests/
├── configs/
├── tools/
├── main.tex
├── pyproject.toml
└── requirements.txt
```

The source tree contains more than eight implementation folders and more than thirty Python code files. The modules are split by responsibility so that researchers can replace a tokenizer, VAE, sampler, quantizer, router, or data source without rewriting the full pipeline.

## Requirements

- Windows, Linux, or macOS.
- Python 3.10 or newer.
- PyTorch 2.1 or newer.
- CUDA is recommended for benchmarking but is not required for unit tests.
- The configured Conda environment for this project is `deeplearn`.

The project was smoke-tested with Python 3.10.19 and PyTorch 2.7.1+cu126 in the `deeplearn` environment.

## Installation

Activate the project environment:

```powershell
conda activate deeplearn
```

Install the package in editable mode:

```powershell
python -m pip install -e .
```

If the environment already contains the dependencies, no additional installation is needed. Otherwise install the requirements:

```powershell
python -m pip install -r requirements.txt
```

The code does not download a model or dataset during import. This makes the repository safe to clone, inspect, and test offline.

## Fastest smoke test

Run the complete smoke test from the repository root:

```powershell
conda run -n deeplearn python scripts/smoke_test.py
```

The smoke test:

1. Creates a small Nexus configuration.
2. Builds the prompt encoder, sparse MoE backbone, quantized projections, and tiny decoder.
3. Generates a batch of images using Euler integration.
4. Measures mean latency, images per second, and GPU peak memory when CUDA is available.
5. Checks output shape and image statistics.
6. Prints a JSON report with a `passed` status.

Example output shape:

```text
{
  "device": "cuda",
  "parameters":  ...,
  "image_shape": [2, 3, 32, 32],
  "benchmark": {
    "batch_size": 2.0,
    "steps": 3.0,
    "mean_seconds": ...,
    "images_per_second": ...,
    "peak_memory_gb": ...
  },
  "status": "passed"
}
```

The exact latency depends on the GPU, CUDA runtime, driver, and system load. The smoke test is a correctness and execution-path check, not a claim of paper-level performance.

## Unit tests

Run the component and pipeline tests:

```powershell
conda run -n deeplearn python -m pytest -q
```

The tests cover:

- Gated DeltaNet output shape.
- Top-k MoE output shape and router scalar.
- Configuration token calculation.
- End-to-end CPU image generation.

If `pytest` is not installed, use:

```powershell
python -m pip install pytest
```

## Generate an image

The sample script uses the compact randomly initialized reference model:

```powershell
conda run -n deeplearn python scripts/sample.py "a small robot in a flower garden" --steps 8 --output outputs/robot.png
```

To force CPU execution:

```powershell
conda run -n deeplearn python scripts/sample.py "a cinematic mountain lake at sunrise" --device cpu --steps 4 --output outputs/lake.png
```

The output is a PNG grid containing the generated batch. Because the reference weights are randomly initialized, this command verifies the complete tensor path but does not produce the trained paper model's visual quality. To use trained weights, load a checkpoint into the backbone, encoder, and decoder before sampling.

## Benchmark inference speed

Run the benchmark script with the default configuration:

```powershell
conda run -n deeplearn python scripts/benchmark.py --steps 8 --repeats 3
```

A CPU benchmark is also supported:

```powershell
conda run -n deeplearn python scripts/benchmark.py --device cpu --steps 3 --repeats 2
```

The benchmark performs one warm-up iteration, synchronizes CUDA before timing, reports mean seconds per batch, computes images per second, and records peak allocated GPU memory. Increase `--repeats` for a more stable estimate. Reduce `--steps` during development and use the paper's 50-step protocol only when comparing a trained checkpoint with an equivalent baseline.

## Toy training

The toy trainer uses deterministic synthetic latent-text pairs and the conditional flow matching objective:

```powershell
conda run -n deeplearn python scripts/train_toy.py
```

The training path follows the paper's central formulation:

```text
x_t = (1 - t) x_0 + t x_1
target = x_1 - x_0
loss = mean((u_theta(x_t, t, y) - target)^2)
```

The toy data source is deliberately small. It exists to validate gradient flow, optimizer construction, router execution, and the loss interface. Replace `SyntheticTextImageDataset` with a captioned image dataset and connect a production VAE and text encoder for real training.

## Configuration

The compact smoke configuration is stored in `configs/smoke.yaml`:

```yaml
image_size: 32
latent_channels: 4
patch_size: 2
dim: 64
depth: 2
heads: 4
experts: 4
top_k: 2
text_dim: 64
text_length: 32
quant_bits: 4
device: cuda
```

Important parameters:

- `image_size` controls the compact decoded image size.
- `latent_channels` controls the latent feature width.
- `patch_size` controls image-token granularity.
- `dim` controls the hidden token width.
- `depth` controls the number of Nexus blocks.
- `heads` controls the number of Gated DeltaNet heads.
- `experts` controls the number of feed-forward experts.
- `top_k` controls how many experts receive each token.
- `quant_bits` controls fake quantization precision.
- `device` accepts `cuda`, `cuda:0`, or `cpu`.

The implementation uses fake quantization with a straight-through gradient path. This keeps the reference code portable and easy to inspect. Hardware-specific integer kernels can be introduced later behind `QuantizedLinear` without changing the model interface.

## API example

```python
from nexus.config import NexusConfig
from nexus.inference.pipeline import NexusPipeline

config = NexusConfig(
    image_size=32,
    dim=64,
    depth=2,
    heads=4,
    text_dim=64,
    device="cuda",
)
pipeline = NexusPipeline(config)
images = pipeline(
    ["a red kite above a blue lake", "a small robot in a garden"],
    steps=8,
    seed=123,
)
print(images.shape)
```

The returned tensor is shaped `[batch, 3, height, width]`, uses floating-point values in `[0, 1]`, and is ready for conversion to a PIL image or a downstream visualization tool.

## Method mapping

The source modules correspond to the main parts of the paper:

| Paper concept | Implementation |
| --- | --- |
| Rectified flow path | `nexus/flow/schedule.py` |
| Conditional flow matching | `nexus/flow/objectives.py` |
| Euler ODE solver | `nexus/flow/solver.py` |
| Patch embedding | `nexus/models/embeddings.py` |
| Adaptive conditioning | `nexus/models/normalization.py` |
| Gated DeltaNet | `nexus/models/gated_deltanet.py` |
| Sparse MoE routing | `nexus/models/router.py` and `nexus/models/moe.py` |
| Low-bit quantization | `nexus/models/quantization.py` |
| Nexus block | `nexus/models/blocks.py` |
| Backbone | `nexus/models/backbone.py` |
| Prompt encoding | `nexus/text/` |
| Latent decoding | `nexus/vae/` |
| Inference pipeline | `nexus/inference/` |
| Training utilities | `nexus/training/` |
| Speed and memory measurement | `nexus/metrics/benchmark.py` |

## Extending toward the full paper system

For a research-grade reproduction, the following substitutions are recommended:

1. Replace `HashTokenizer` and `PromptEncoder` with the paper's CLIP and T5 text encoders.
2. Replace `TinyEncoder` and `TinyDecoder` with a pretrained latent VAE.
3. Increase the backbone width and depth to the target model configuration.
4. Use the full LAION training mixture and the paper's data filtering procedure.
5. Add distributed data loading, gradient checkpointing, mixed precision, and checkpoint sharding.
6. Replace fake quantization with a tested INT4/FP4 kernel path for the target GPU.
7. Implement the complete COCO-30K, LAION-5K, GenEval, DPG-bench, FID, CLIP, FLOPs, and memory evaluation scripts.
8. Run all baselines under the same sampler, resolution, precision, and hardware protocol.

The current folder boundaries are intended to make each replacement local. For example, a production VAE can replace the two files under `nexus/vae/`, while the flow solver and Nexus backbone remain unchanged.

## Reproducibility notes

`seed_everything` seeds Python, NumPy, and PyTorch. The sample pipeline also creates a device-aware generator, so repeated calls with the same prompt list, seed, model state, and device are deterministic to the extent supported by the selected backend.

GPU timing uses synchronization before and after each measured batch. CPU timing uses `time.perf_counter`. Peak memory is reported only for CUDA devices because the PyTorch CUDA allocator exposes the corresponding statistic directly.

The repository keeps the paper source at `main.tex` and the figures under `Figs/`. The code is independent from the LaTeX build and does not alter the paper source.

## Citation

```bibtex
@inproceedings{wang2026nexus,
  title={Nexus: Structured Synergy for Efficient Text-to-Image Generation using Rectified Flow Model},
  author={Wang, Yizhao},
  booktitle={Asian Conference on Computer Vision},
  year={2026},
  note={Accepted at ACCV 2026}
}
```

## License

This implementation is released under the MIT License. See `LICENSE` for the full text.
