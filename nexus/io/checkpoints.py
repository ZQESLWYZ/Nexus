from pathlib import Path

import torch


def save_checkpoint(model: torch.nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})
