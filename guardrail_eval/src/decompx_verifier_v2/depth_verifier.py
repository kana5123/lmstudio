"""DepthVerifier (§21-§22).  층 증거 K 개를 깊이 순서열로 처리한다.

토큰 Transformer 는 구현하지 않는다.  깊이 방향만 본다.
"""
import math

import torch
import torch.nn as nn

from src.decompx_verifier_v2.layer_encoder import LayerEvidenceEncoder


def sinusoidal_depth(K, d_model, device):
    """정규화 깊이 l/K 기반 sinusoidal 인코딩."""
    t = (torch.arange(K, device=device, dtype=torch.float32) + 1) / K
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    ang = t[:, None] / (10000 ** (i / d_model))[None, :]
    pe = torch.zeros(K, d_model, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


def head(d_model, hidden, dropout):
    return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))


class DepthVerifier(nn.Module):
    def __init__(self, d, n_transitions, proj_dim=128, fusion_hidden=512, d_model=256,
                 nhead=4, num_layers=2, dim_feedforward=1024, dropout=0.1, norm_first=True,
                 pos_encoding="learned"):
        super().__init__()
        self.K, self.d_model, self.pos_encoding = n_transitions, d_model, pos_encoding
        self.layer_encoder = LayerEvidenceEncoder(d, proj_dim, fusion_hidden, d_model, dropout)
        self.vcls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.vcls, std=0.02)
        if pos_encoding == "learned":
            self.depth_pos = nn.Parameter(torch.zeros(1, n_transitions + 1, d_model))
            nn.init.normal_(self.depth_pos, std=0.02)
        elif pos_encoding != "sinusoidal":
            raise ValueError(pos_encoding)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward, dropout=dropout,
                                         activation="gelu", batch_first=True,
                                         norm_first=norm_first)
        self.depth = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head_attack = head(d_model, 128, dropout)    # TP=0 / FP=1
        self.head_benign = head(d_model, 128, dropout)    # TN=0 / FN=1

    def forward(self, ev):
        """ev: q, zD_pos, zD_neg, zH_pos, zH_neg, mass_pos, mass_neg
        -> dict(z_verifier [B,d_model], attack_error_logit [B], benign_error_logit [B])"""
        r = self.layer_encoder(ev["q"], ev["zD_pos"], ev["zD_neg"],
                               ev["zH_pos"], ev["zH_neg"], ev["mass_pos"], ev["mass_neg"])
        B = r.shape[0]
        x = torch.cat([self.vcls.expand(B, -1, -1), r], dim=1)          # [B,K+1,d_model]
        if self.pos_encoding == "learned":
            x = x + self.depth_pos
        else:
            pe = sinusoidal_depth(self.K, self.d_model, x.device)
            x = x + torch.cat([torch.zeros(1, self.d_model, device=x.device), pe], 0)[None]
        z = self.depth(x)[:, 0]                                          # VCLS
        return dict(z_verifier=z,
                    attack_error_logit=self.head_attack(z).squeeze(-1),
                    benign_error_logit=self.head_benign(z).squeeze(-1))
