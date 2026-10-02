from pathlib import Path

import torch


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.int16)
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    low = flat[0::2] & 15
    high = (flat[1::2] & 15) << 4
    return (low | high).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    values = packed.reshape(-1)
    low = values & 15
    high = (values >> 4) & 15
    unpacked = torch.stack((low, high), dim=-1).reshape(-1)[: int(torch.tensor(shape).prod())]
    unpacked = unpacked.to(torch.int8)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.reshape(shape).float() * scale


def load_packed_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device)
    for name, parameter in model.named_parameters():
        packed = payload["weights"][name]
        shape = tuple(payload["shapes"][name])
        scale = float(payload["scales"][name])
        parameter.data.copy_(unpack_int4(packed, shape, scale).to(parameter.device, parameter.dtype))
    return payload.get("metadata", {})
