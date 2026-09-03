"""LayerEvidenceEncoder (§18).  층 하나의 증거를 r_l in R^256 으로 요약한다.

같은 evidence type 의 pos/neg 에는 동일 projector 가중치를 쓴다(Pd, Ph 공유).
"""
import torch
import torch.nn as nn


def projector(d, out, dropout):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out), nn.GELU(), nn.Dropout(dropout))


class LayerEvidenceEncoder(nn.Module):
    def __init__(self, d, proj_dim=128, fusion_hidden=512, d_model=256, dropout=0.1):
        super().__init__()
        self.Pq = projector(d, proj_dim, dropout)     # q_l
        self.Pd = projector(d, proj_dim, dropout)     # zD_pos, zD_neg 공유
        self.Ph = projector(d, proj_dim, dropout)     # zH_pos, zH_neg 공유
        self.in_dim = 5 * proj_dim + 2
        self.fusion = nn.Sequential(
            nn.Linear(self.in_dim, fusion_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, q, zD_pos, zD_neg, zH_pos, zH_neg, mass_pos, mass_neg):
        """모두 [B,K,d] (mass 는 [B,K]) -> r [B,K,d_model]"""
        x = torch.cat([self.Pq(q), self.Pd(zD_pos), self.Pd(zD_neg),
                       self.Ph(zH_pos), self.Ph(zH_neg),
                       torch.log1p(mass_pos.clamp_min(0)).unsqueeze(-1),
                       torch.log1p(mass_neg.clamp_min(0)).unsqueeze(-1)], dim=-1)
        assert x.shape[-1] == self.in_dim, f"{x.shape[-1]} != {self.in_dim}"
        return self.fusion(x)
