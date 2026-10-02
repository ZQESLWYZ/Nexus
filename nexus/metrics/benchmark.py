import time

import torch


@torch.no_grad()
def benchmark_pipeline(pipeline, prompts: list[str], steps: int = 4, warmup: int = 1, repeats: int = 3) -> dict[str, float]:
    for index in range(warmup):
        pipeline(prompts, steps=steps, seed=index)
    if pipeline.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(pipeline.device)
        torch.cuda.synchronize(pipeline.device)
    durations = []
    for index in range(repeats):
        start = time.perf_counter()
        pipeline(prompts, steps=steps, seed=100 + index)
        if pipeline.device.type == "cuda":
            torch.cuda.synchronize(pipeline.device)
        durations.append(time.perf_counter() - start)
    result = {
        "batch_size": float(len(prompts)),
        "steps": float(steps),
        "mean_seconds": sum(durations) / len(durations),
        "images_per_second": len(prompts) / (sum(durations) / len(durations)),
    }
    if pipeline.device.type == "cuda":
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated(pipeline.device) / 1024**3
    else:
        result["peak_memory_gb"] = 0.0
    return result
