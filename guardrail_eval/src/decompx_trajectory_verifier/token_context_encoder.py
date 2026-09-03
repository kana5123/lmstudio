"""§17-§18 token position + token-context encoder.

raw token embedding 이나 input word embedding 을 feature 로 넣지 않는다.
input_ids 는 정렬 확인 / 패딩 마스크 / 사람이 읽는 분석에만 쓴다.
"""
import torch
import torch.nn as nn


class TokenContextEncoder(nn.Module):
    def __init__(self, d_model=128, max_len=512, nhead=4, num_layers=2, dim_feedforward=512,
                 dropout=0.1, norm_first=True, activation="gelu"):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.vcls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.vcls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation=activation, batch_first=True,
                                           norm_first=norm_first)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, v, mask):
        """v [B,T,d_model], mask [B,T] (1=실토큰) -> z [B,d_model]"""
        B, T, _ = v.shape
        x = v + self.pos[:, :T]
        x = torch.cat([self.vcls.expand(B, -1, -1), x], 1)                # [B,T+1,d]
        pad = torch.cat([torch.zeros(B, 1, device=mask.device, dtype=mask.dtype), mask], 1)
        return self.enc(x, src_key_padding_mask=(pad == 0))[:, 0]
