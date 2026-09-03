"""DecompXTransitionExtractor: C -> D, 그리고 층 정렬.

기존 포트(src/pg2_decompx)를 그대로 import 해서 쓴다.  복사하지 않는다.

층 정렬(§10) -- 0-기반 인덱스 j = 0..K-1 에 대해
    C_idx[j]   = C^(L(j+1))          포트가 주는 cls_encoder 의 j 번째
    D[j]       = C_idx[j+1] - C_idx[j]      = D_(L(j+1) -> L(j+2))
    q[j]       = q_(L(j+2))                 목적지 층
    H_pre[j]   = hidden[j+1]                = H^(L(j+1)), transition 직전 상태
"""
import torch

from src.pg2_decompx.decompx_utils import DecompXConfig
from src.pg2_decompx.modeling_deberta_v2_decompx import DecompXDebertaV2

DCFG = DecompXConfig(output_all_layers=True, output_encoder=None,
                     include_classifier_w_pooler=False, output_classifier=False)


class DecompXTransitionExtractor:
    def __init__(self, adapter):
        self.a = adapter
        self.dx = DecompXDebertaV2(adapter.model)
        self.L = adapter.get_num_layers()

    @property
    def K(self):
        return self.L - 1

    @torch.no_grad()
    def extract(self, input_ids, attention_mask):
        """-> dict(D [B,K,T,d], H_pre [B,K,T,d], hidden tuple, logits [B,2], C [B,L,T,d])"""
        logits, _, hs, out = self.dx.forward(input_ids, attention_mask, DCFG,
                                             output_hidden_states=True)
        C = out.cls_encoder                                  # (B, L, T, d) = C_L1..C_L(L)
        assert C.shape[1] == self.L, f"C 층수 {C.shape[1]} != {self.L}"
        m = attention_mask[:, None, :, None].to(C.dtype)
        D = (C[:, 1:] - C[:, :-1]) * m                       # (B, K, T, d)
        H_pre = torch.stack(hs[1:self.L], dim=1)             # (B, K, T, d) = H_L1..H_L(L-1)
        assert D.shape[1] == self.K and H_pre.shape[1] == self.K
        return dict(D=D, H_pre=H_pre, hidden=hs, logits=logits, C=C)
