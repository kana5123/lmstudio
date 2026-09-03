"""§13-§14 source-token 별 depth trajectory encoder.

모든 source token 에 같은 Depth Transformer 파라미터를 쓴다.
표준 learned self-attention 만 사용한다(별도 Q/K/V 설계 없음).
"""
import torch
import torch.nn as nn


class DepthTrajectoryEncoder(nn.Module):
    def __init__(self, d_v, d_model=128, nhead=4, num_layers=2, dim_feedforward=512,
                 dropout=0.1, norm_first=True, activation="gelu"):
        super().__init__()
        assert d_v == d_model, "cell projection 차원과 depth d_model 을 맞춘다"
        self.tcls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.tcls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation=activation, batch_first=True,
                                           norm_first=norm_first)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, U):
        """U [B,L,T,d_v] -> tau [B,T,d_model]"""
        B, L, T, dv = U.shape
        x = U.permute(0, 2, 1, 3).reshape(B * T, L, dv)          # [B*T, L, d_v]
        x = torch.cat([self.tcls.expand(B * T, -1, -1), x], 1)   # [B*T, L+1, d_v]
        return self.enc(x)[:, 0].reshape(B, T, dv)
