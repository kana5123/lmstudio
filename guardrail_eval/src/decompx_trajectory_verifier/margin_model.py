"""M0: base 최종 score 만 쓰는 대조군 (§10).

입력 [z_benign, z_attack, z_attack - z_benign] -> 3 -> 16 -> 1
DecompX evidence 를 전혀 쓰지 않는다.
"""
import torch
import torch.nn as nn


class MarginOnly(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        self.std.copy_(torch.as_tensor(std, dtype=self.std.dtype).clamp_min(1e-6))

    def forward(self, logits):
        """logits [B,2] (0=benign,1=attack) -> [B]"""
        z = torch.stack([logits[:, 0], logits[:, 1], logits[:, 1] - logits[:, 0]], -1)
        return self.net((z - self.mean) / self.std).squeeze(-1)
