import hashlib

import torch


class HashTokenizer:
    def __init__(self, length: int = 32, vocabulary: int = 8192):
        self.length = length
        self.vocabulary = vocabulary

    def encode(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        values = []
        for prompt in prompts:
            words = prompt.lower().split()
            ids = [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.vocabulary for word in words]
            ids = (ids + [0] * self.length)[: self.length]
            values.append(ids)
        return torch.tensor(values, device=device, dtype=torch.long)
