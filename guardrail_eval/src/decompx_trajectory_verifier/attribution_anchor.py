"""§15 final attribution anchor.  Y_benign / Y_attack / a 세 scalar 를 128 차원으로.

표준화 통계는 train split 에서만 구한다.  test 통계를 쓰지 않는다.
"""
import torch
import torch.nn as nn


class AttributionAnchor(nn.Module):
    def __init__(self, hidden=32, out=128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, out))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        self.std.copy_(torch.as_tensor(std, dtype=self.std.dtype).clamp_min(1e-6))

    def forward(self, f_attr):
        """f_attr [B,T,3] = (Y_benign, Y_attack, a) -> e_attr [B,T,out]"""
        return self.mlp((f_attr - self.mean) / self.std)
