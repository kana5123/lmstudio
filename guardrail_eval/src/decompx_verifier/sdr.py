"""Signed Directional Retrieval (SDR).

학습 파라미터가 없는 결정론적 연산이다.  learned cross-attention 이 아니다.
Q 와 K 는 둘 다 base 모델의 native hidden space R^d 에 있고 같은 transition 을 표현한다.

  Q = g_l   = h_CLS[l] - h_CLS[l-1]      전체 CLS 가 어느 방향으로 움직였는가
  K = D_lk  = C[l,k] - C[l-1,k]          토큰 k 에 귀속된 기여가 어느 방향으로 변했는가

대수적으로 g = sum_k D 이다(독립된 두 양식이 아니다).

용어: p 는 causal importance 가 아니라 'global CLS movement 로의 부호 있는 사영'
      (signed projection onto global CLS movement) 이다.
"""
import torch

EPS = 1e-12


def transitions(cls_C, hidden_states, mask):
    """C, 은닉상태 -> transition 단위 텐서.

    입력:
      cls_C          (B, L, N, d)  DecompX cls_encoder = C[1]..C[L]
      hidden_states  길이 L+1 의 (B, N, d) 리스트.  hs[0]=임베딩 출력
      mask           (B, N) 1=실토큰
    출력:
      g      (B, K, d)      K = L-1
      D      (B, K, N, d)
      H_pre  (B, K, N, d)   transition l 직전 상태 hs[l-1], l=2..L
      E      (B, N, d)      임베딩 출력 (position_biased_input=False 이므로 순수 어휘)
    """
    L = cls_C.shape[1]
    assert len(hidden_states) == L + 1, f"은닉상태 {len(hidden_states)} != L+1={L+1}"
    hs = torch.stack(hidden_states, 1)                    # (B, L+1, N, d)
    g = hs[:, 2:L + 1, 0] - hs[:, 1:L, 0]                 # (B, K, d)   l=2..L 의 CLS 이동
    D = cls_C[:, 1:L] - cls_C[:, 0:L - 1]                 # (B, K, N, d)
    H_pre = hs[:, 1:L]                                    # (B, K, N, d) transition 직전 상태
    E = hs[:, 0]                                          # (B, N, d)
    m = mask[:, None, :, None].to(D.dtype)
    return g, D * m, H_pre, E


def signed_projection(g, D, mask):
    """§14/§16: p_lk = (g^T D_lk) / (||g|| + eps).  수치적으로 안전한 형태.

    반환 p (B, K, N).  패딩 위치는 0.
    """
    gn = g.norm(dim=-1, keepdim=True)                     # (B, K, 1)
    p = torch.einsum("bkd,bknd->bkn", g, D) / (gn + EPS)
    return p * mask[:, None, :].to(p.dtype)


def signed_retrieval(p, mask):
    """§17: softmax 가 아니라 부호 성분이 있을 때만 검색한다."""
    m = mask[:, None, :].to(p.dtype)
    w_pos, w_neg = torch.relu(p) * m, torch.relu(-p) * m
    mass_pos, mass_neg = w_pos.sum(-1), w_neg.sum(-1)      # (B, K)
    a_pos = w_pos / (mass_pos.unsqueeze(-1) + EPS)
    a_neg = w_neg / (mass_neg.unsqueeze(-1) + EPS)
    # mass 가 0 이면 pooled 결과를 영벡터로 둔다
    a_pos = a_pos * (mass_pos > 0).unsqueeze(-1).to(a_pos.dtype)
    a_neg = a_neg * (mass_neg > 0).unsqueeze(-1).to(a_neg.dtype)
    return a_pos, a_neg, mass_pos, mass_neg


def pool(alpha, V):
    """alpha (B,K,N) 와 값 V (B,K,N,d) 또는 (B,N,d) -> (B,K,d)."""
    if V.dim() == 3:
        return torch.einsum("bkn,bnd->bkd", alpha, V)
    return torch.einsum("bkn,bknd->bkd", alpha, V)


def sdr_evidence(cls_C, hidden_states, mask, low_motion_thr=None):
    """전체 SDR.  §11/§19 의 증거를 한 번에 만든다."""
    g, D, H_pre, E = transitions(cls_C, hidden_states, mask)
    p = signed_projection(g, D, mask)
    a_pos, a_neg, mass_pos, mass_neg = signed_retrieval(p, mask)
    ev = dict(
        g=g,
        zD_pos=pool(a_pos, D),   zD_neg=pool(a_neg, D),
        zH_pos=pool(a_pos, H_pre), zH_neg=pool(a_neg, H_pre),
        zE_pos=pool(a_pos, E),   zE_neg=pool(a_neg, E),
        mass_pos=mass_pos, mass_neg=mass_neg,
    )
    gn = g.norm(dim=-1)
    ev["g_norm"] = gn
    ev["low_motion"] = (gn < low_motion_thr).to(g.dtype) if low_motion_thr is not None \
        else torch.zeros_like(gn)
    return ev, p, D, H_pre, E
