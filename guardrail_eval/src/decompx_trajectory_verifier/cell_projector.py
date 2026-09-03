"""§12 Cell projection.  모든 layer, 모든 source token 에 같은 선형 사영을 쓴다.

C 를 norm scalar 로 축약하지 않는다.  d 차원 벡터를 그대로 d_v 로 보낸다.
"""
import torch
import torch.nn as nn


class CellProjector(nn.Module):
    def __init__(self, d, d_v, n_layers):
        super().__init__()
        self.P_C = nn.Linear(d, d_v, bias=True)
        self.E_depth = nn.Parameter(torch.zeros(n_layers, d_v))
        nn.init.normal_(self.E_depth, std=0.02)

    def forward(self, C):
        """C [B,L,T,d] -> U [B,L,T,d_v]"""
        return self.P_C(C) + self.E_depth[None, :, None, :]
